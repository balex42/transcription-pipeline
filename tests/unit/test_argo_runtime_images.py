"""Runtime-oriented Argo image-parameter checks for the WorkflowTemplate."""

from pathlib import Path

import yaml

from speech_transcriber.config import ASR_BACKENDS

TEMPLATE_PATH = Path(__file__).parents[2] / "deploy/argo/transcription-workflowtemplate.yaml"

# backend -> runtime image parameter
BACKEND_IMAGES = {
    "parakeet": "nemo_image",
    "primeline": "nemo_image",
    "canary": "nemo_image",
    "qwen": "transformers_image",
    "nemotron": "transformers_image",
    "voxtral": "transformers_image",
    "faster-whisper": "ctranslate2_image",
}

# Runtime images share one offline environment block. TRANSFORMERS_OFFLINE may
# also be present in the CTranslate2 image, where it is simply not consumed.
SHARED_RECOGNITION_ENV = {
    "HF_HOME": "/models/huggingface",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def load() -> tuple[dict[str, dict], dict[str, str]]:
    document = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
    templates = {template["name"]: template for template in document["spec"]["templates"]}
    parameters = {
        parameter["name"]: parameter["value"]
        for parameter in document["spec"]["arguments"]["parameters"]
    }
    return templates, parameters


def load_dag_tasks() -> dict[str, dict]:
    templates, _ = load()
    return {task["name"]: task for task in templates["transcription"]["dag"]["tasks"]}


def resolve(value: object, backend: str) -> str:
    """Resolve a template placeholder to the concrete value a task would pass."""
    text = str(value)
    _, workflow_parameters = load()
    for name, parameter_value in workflow_parameters.items():
        text = text.replace(f"{{{{workflow.parameters.{name}}}}}", parameter_value)
    text = text.replace("{{inputs.parameters.backend}}", backend)
    text = text.replace(
        "{{inputs.parameters.image}}",
        f"image-for-{BACKEND_IMAGES[backend]}",
    )
    return text


def test_runtime_image_parameters_replaced_backend_image_parameters() -> None:
    _, parameters = load()

    assert set(parameters) == {
        "utility_image",
        "transformers_image",
        "nemo_image",
        "ctranslate2_image",
        "backends",
        "language",
    }


def test_workflow_language_flows_only_into_prepare() -> None:
    templates, parameters = load()

    assert parameters["language"] == "de-DE"
    prepare_args = templates["prepare"]["container"]["args"]
    assert prepare_args[prepare_args.index("--language") + 1] == (
        "{{workflow.parameters.language}}"
    )
    assert "--language" not in templates["recognize"]["container"]["args"]
    assert "--language" not in templates["finalize"]["container"]["args"]


def test_one_shared_recognition_template_exists() -> None:
    """The seven per-backend container templates collapsed into one worker template."""
    templates, _ = load()

    assert "recognize" in templates
    assert not any(name.startswith("recognize-") for name in templates)
    assert templates["recognize"]["container"]["image"] == "{{inputs.parameters.image}}"
    args = templates["recognize"]["container"]["args"]
    assert args[0] == "recognize"
    assert args[args.index("--backend") + 1] == "{{inputs.parameters.backend}}"
    assert set(
        item["name"] for item in templates["recognize"]["inputs"]["parameters"]
    ) == {"backend", "image"}


def test_recognition_dag_tasks_map_backends_to_runtime_images() -> None:
    templates, _ = load()
    dag_tasks = load_dag_tasks()

    for backend, parameter in BACKEND_IMAGES.items():
        task = dag_tasks[f"recognize-{backend}"]
        assert task["template"] == "recognize", backend
        params = {
            item["name"]: item["value"] for item in task["arguments"]["parameters"]
        }
        assert params["backend"] == backend, backend
        assert params["image"] == f"{{{{workflow.parameters.{parameter}}}}}", backend


def test_prepare_and_finalize_use_the_transformers_image() -> None:
    templates, _ = load()

    assert templates["prepare"]["container"]["image"] == (
        "{{workflow.parameters.transformers_image}}"
    )
    assert templates["finalize"]["container"]["image"] == (
        "{{workflow.parameters.transformers_image}}"
    )


def test_recognized_template_keeps_one_gpu_and_read_only_model_cache() -> None:
    templates, _ = load()
    recognize = templates["recognize"]

    assert recognize["container"]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert recognize["volumes"] == [
        {"name": "model-cache", "persistentVolumeClaim": {"claimName": "speech-model-cache"}}
    ]
    assert recognize["container"]["volumeMounts"][0] == {
        "name": "model-cache",
        "mountPath": "/models",
        "readOnly": True,
    }


def test_finalizer_remains_gpu_free_without_model_cache() -> None:
    templates, _ = load()

    finalizer = templates["finalize"]
    assert "resources" not in finalizer["container"]
    assert "model-cache" not in str(finalizer)


def test_prepare_uses_one_gpu_and_the_read_only_model_cache() -> None:
    templates, _ = load()

    prepare = templates["prepare"]
    assert prepare["container"]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert prepare["container"]["volumeMounts"][0] == {
        "name": "model-cache",
        "mountPath": "/models",
        "readOnly": True,
    }


def test_backend_artifact_paths_and_dag_branches_remain_unchanged() -> None:
    templates, _ = load()
    dag_tasks = load_dag_tasks()

    # One shared template parameterizes the ASR artifact S3 key per backend.
    shared_key = templates["recognize"]["outputs"]["artifacts"][0]["s3"]["key"]
    assert shared_key == "runs/{{workflow.uid}}/asr/{{inputs.parameters.backend}}/"

    for backend in BACKEND_IMAGES:
        # The explicit per-backend DAG branch is preserved with a concrete artifact key.
        finalize_task = dag_tasks[f"finalize-{backend}"]
        assert finalize_task["template"] == "finalize"
        assert finalize_task["depends"] == f"recognize-{backend}.Succeeded"
        assert finalize_task["arguments"]["parameters"][0]["value"] == backend
        publish_task = dag_tasks[f"publish-{backend}"]
        assert publish_task["template"] == "publish"
        assert publish_task["depends"] == f"finalize-{backend}.Succeeded"
        recognize_task = dag_tasks[f"recognize-{backend}"]
        assert recognize_task["depends"] == "prepare.Succeeded"


def test_recognition_dag_tasks_select_backends_conditionally() -> None:
    dag_tasks = load_dag_tasks()

    for backend in BACKEND_IMAGES:
        task = dag_tasks[f"recognize-{backend}"]
        assert task["when"] == (
            f"{{{{tasks.validate-backends.outputs.parameters.run_{backend.replace('-', '_')}}}}}"
            " == true"
        ), backend


def test_backend_selection_list_is_unchanged() -> None:
    _, parameters = load()

    assert parameters["backends"] == (
        '["parakeet","primeline","qwen","nemotron","voxtral","faster-whisper","canary"]'
    )


def test_no_backend_named_image_parameter_remains() -> None:
    _, parameters = load()
    raw = TEMPLATE_PATH.read_text(encoding="utf-8")

    for stale in (
        "application_image",
        "parakeet_image",
        "primeline_image",
        "qwen_image",
        "nemotron_image",
        "voxtral_image",
        "faster_whisper_image",
        "canary_image",
    ):
        assert stale not in parameters
        assert stale not in raw


def test_recognition_template_uses_the_shared_offline_environment() -> None:
    """All recognition runtimes receive the same offline environment block.

    TRANSFORMERS_OFFLINE is deliberately uniform: runtime images that never
    read Transformers (CTranslate2) ignore the harmless unused flag, which
    keeps the shared recognize template a single environment definition.
    """
    templates, _ = load()
    env = {
        variable["name"]: variable["value"]
        for variable in templates["recognize"]["container"]["env"]
    }
    assert env == SHARED_RECOGNITION_ENV


def test_validator_script_still_supports_all_backends() -> None:
    templates, _ = load()

    script = templates["validate-backends"]["container"]["args"][0]
    for backend in ASR_BACKENDS:
        assert f'"{backend}"' in script
    assert "backend.replace('-', '_')" in script


def test_recognition_tasks_use_the_canonical_recognize_command() -> None:
    templates, _ = load()

    args = templates["recognize"]["container"]["args"]
    assert args[0] == "recognize"
    assert "--prepared" in args and "--backend" in args
    assert args[args.index("--prepared") + 1] == "/work/prepared"
    assert args[args.index("--output") + 1] == "/work/asr"
    assert args[args.index("--backend") + 1] == "{{inputs.parameters.backend}}"
    raw = TEMPLATE_PATH.read_text(encoding="utf-8")
    for stale in ("recognize-prepared", "--asr-result", "--expected-backend"):
        assert stale not in raw


def test_finalize_task_uses_the_canonical_finalize_command() -> None:
    templates, _ = load()

    args = templates["finalize"]["container"]["args"]
    assert args[0] == "finalize"
    assert "--prepared" in args and "--asr" in args and "--backend" in args
    assert args[args.index("--asr") + 1] == "/work/asr"
    assert args[args.index("--backend") + 1] == "{{inputs.parameters.backend}}"
    raw = TEMPLATE_PATH.read_text(encoding="utf-8")
    for stale in ("finalize-prepared", "recognize-prepared", "--asr-result",
                  "--expected-backend"):
        assert stale not in raw


def test_prepare_task_uses_the_canonical_prepare_command() -> None:
    templates, _ = load()

    args = templates["prepare"]["container"]["args"]
    assert args[0] == "prepare"
    assert args[1] == "/input/recording"
    assert args[args.index("--output") + 1] == "/work/prepared"


def test_worker_invocations_parse_with_the_real_cli() -> None:
    """Every Argo worker invocation must be valid CLI arguments.

    Guards against Argo/CLI drift: a flag the parser does not accept makes
    every recognition or finalize task exit 2 before any work happens. The
    shared recognize template is resolved per backend to the exact effective
    command its DAG task produces.
    """
    from speech_transcriber.cli import build_parser

    templates, _ = load()
    parser = build_parser()

    def parse(argv: list[str], context: str) -> None:
        try:
            parser.parse_args(argv)
        except SystemExit as error:  # pragma: no cover - failure message
            raise AssertionError(f"invalid args in {context}: {argv}") from error

    parse(
        [resolve(value, "parakeet") for value in templates["prepare"]["container"]["args"]],
        "prepare",
    )

    shared_args = templates["recognize"]["container"]["args"]
    for backend in ASR_BACKENDS:
        resolved = [resolve(value, backend) for value in shared_args]
        parse(resolved, f"recognize-{backend}")

    finalize_args = templates["finalize"]["container"]["args"]
    resolved_finalize = [resolve(value, "parakeet") for value in finalize_args]
    parse(resolved_finalize, "finalize")

"""Runtime-oriented Argo image-parameter checks for the WorkflowTemplate."""

from pathlib import Path

import yaml

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


def load() -> tuple[dict[str, dict], dict[str, str]]:
    document = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
    templates = {template["name"]: template for template in document["spec"]["templates"]}
    parameters = {
        parameter["name"]: parameter["value"]
        for parameter in document["spec"]["arguments"]["parameters"]
    }
    return templates, parameters


def test_runtime_image_parameters_replaced_backend_image_parameters() -> None:
    _, parameters = load()

    assert set(parameters) == {
        "utility_image",
        "transformers_image",
        "nemo_image",
        "ctranslate2_image",
        "backends",
    }


def test_recognize_templates_map_backends_to_runtime_images() -> None:
    templates, _ = load()

    for backend, parameter in BACKEND_IMAGES.items():
        recognize = templates[f"recognize-{backend}"]
        assert recognize["container"]["image"] == f"{{{{workflow.parameters.{parameter}}}}}", (
            backend
        )


def test_prepare_and_finalize_use_the_transformers_image() -> None:
    templates, _ = load()

    assert templates["prepare"]["container"]["image"] == (
        "{{workflow.parameters.transformers_image}}"
    )
    assert templates["finalize"]["container"]["image"] == (
        "{{workflow.parameters.transformers_image}}"
    )


def test_recognition_tasks_keep_one_gpu_and_read_only_model_cache() -> None:
    templates, _ = load()

    for backend in BACKEND_IMAGES:
        recognize = templates[f"recognize-{backend}"]
        assert recognize["container"]["resources"]["limits"]["nvidia.com/gpu"] == "1", backend
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
    dag_tasks = {task["name"]: task for task in templates["transcription"]["dag"]["tasks"]}

    for backend in BACKEND_IMAGES:
        recognize_template = templates[f"recognize-{backend}"]
        assert recognize_template["outputs"]["artifacts"][0]["s3"]["key"] == (
            f"runs/{{{{workflow.uid}}}}/asr/{backend}/"
        )
        # The explicit per-backend DAG branch is preserved; only images changed.
        recognize_task = dag_tasks[f"recognize-{backend}"]
        assert recognize_task["template"] == f"recognize-{backend}"
        assert recognize_task["depends"] == "prepare.Succeeded"
        finalize_task = dag_tasks[f"finalize-{backend}"]
        assert finalize_task["template"] == "finalize"
        assert finalize_task["depends"] == f"recognize-{backend}.Succeeded"
        assert finalize_task["arguments"]["parameters"][0]["value"] == backend
        publish_task = dag_tasks[f"publish-{backend}"]
        assert publish_task["template"] == "publish"
        assert publish_task["depends"] == f"finalize-{backend}.Succeeded"


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


def test_offline_flags_and_gpu_are_preserved_on_recognition_tasks() -> None:
    templates, _ = load()
    offline_by_backend = {
        "parakeet": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "primeline": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "qwen": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "nemotron": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "voxtral": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "faster-whisper": {"HF_HUB_OFFLINE": "1"},
        "canary": {"HF_HUB_OFFLINE": "1"},
    }
    for backend, expected_env in offline_by_backend.items():
        env = {
            variable["name"]: variable["value"]
            for variable in templates[f"recognize-{backend}"]["container"]["env"]
        }
        for name, value in expected_env.items():
            assert env[name] == value, backend


def test_validator_script_still_supports_all_backends() -> None:
    templates, _ = load()

    script = templates["validate-backends"]["container"]["args"][0]
    for backend in ("parakeet", "primeline", "qwen", "nemotron", "voxtral", "faster-whisper",
                    "canary"):
        assert f'"{backend}"' in script
    assert "backend.replace('-', '_')" in script
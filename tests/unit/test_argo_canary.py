from pathlib import Path

import yaml


def test_canary_argo_branch_uses_dedicated_gpu_recognition_and_common_finalizer() -> None:
    template_path = Path(__file__).parents[2] / "deploy/argo/transcription-workflowtemplate.yaml"
    document = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    templates = {template["name"]: template for template in document["spec"]["templates"]}
    parameters = {
        parameter["name"]: parameter["value"]
        for parameter in document["spec"]["arguments"]["parameters"]
    }

    assert parameters["canary_image"] == "ghcr.io/balex42/transcription-pipeline-canary:sha-b7ce9fa"

    validator = templates["validate-backends"]
    output_paths = {
        parameter["name"]: parameter["valueFrom"]["path"]
        for parameter in validator["outputs"]["parameters"]
    }
    assert output_paths["run_canary"] == "/tmp/run_canary"
    assert output_paths["run_faster_whisper"] == "/tmp/run_faster_whisper"
    validator_script = validator["container"]["args"][0]
    assert '"canary"' in validator_script
    assert "backend.replace('-', '_')" in validator_script

    dag_tasks = {task["name"]: task for task in templates["transcription"]["dag"]["tasks"]}
    recognize = dag_tasks["recognize-canary"]
    assert recognize["template"] == "recognize-canary"
    assert recognize["depends"] == "prepare.Succeeded"
    assert recognize["when"] == "{{tasks.validate-backends.outputs.parameters.run_canary}} == true"
    assert dag_tasks["finalize-canary"]["template"] == "finalize"
    assert dag_tasks["finalize-canary"]["depends"] == "recognize-canary.Succeeded"
    assert dag_tasks["publish-canary"]["template"] == "publish"
    assert dag_tasks["publish-canary"]["depends"] == "finalize-canary.Succeeded"

    canary = templates["recognize-canary"]
    assert canary["container"]["image"] == "{{workflow.parameters.canary_image}}"
    assert canary["container"]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert canary["outputs"]["artifacts"][0]["s3"]["key"] == "runs/{{workflow.uid}}/asr/canary/"
    assert canary["volumes"] == [
        {"name": "model-cache", "persistentVolumeClaim": {"claimName": "speech-model-cache"}}
    ]
    assert canary["container"]["volumeMounts"][0] == {
        "name": "model-cache",
        "mountPath": "/models",
        "readOnly": True,
    }

    finalizer = templates["finalize"]
    assert finalizer["container"]["image"] == "{{workflow.parameters.application_image}}"
    assert "resources" not in finalizer["container"]
    assert "model-cache" not in str(finalizer)


def test_primeline_argo_branch_uses_nemo_image_gpu_recognition_and_common_finalizer() -> None:
    template_path = Path(__file__).parents[2] / "deploy/argo/transcription-workflowtemplate.yaml"
    document = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    templates = {template["name"]: template for template in document["spec"]["templates"]}
    parameters = {
        parameter["name"]: parameter["value"]
        for parameter in document["spec"]["arguments"]["parameters"]
    }

    assert parameters["primeline_image"] == (
        "ghcr.io/balex42/transcription-pipeline-canary:sha-b7ce9fa"
    )
    assert parameters["backends"] == (
        '["parakeet","primeline","qwen","nemotron","voxtral","faster-whisper","canary","granite"]'
    )

    validator = templates["validate-backends"]
    output_paths = {
        parameter["name"]: parameter["valueFrom"]["path"]
        for parameter in validator["outputs"]["parameters"]
    }
    assert output_paths["run_primeline"] == "/tmp/run_primeline"
    validator_script = validator["container"]["args"][0]
    assert '"primeline"' in validator_script

    dag_tasks = {task["name"]: task for task in templates["transcription"]["dag"]["tasks"]}
    recognize = dag_tasks["recognize-primeline"]
    assert recognize["template"] == "recognize-primeline"
    assert recognize["depends"] == "prepare.Succeeded"
    assert (
        recognize["when"]
        == "{{tasks.validate-backends.outputs.parameters.run_primeline}} == true"
    )
    assert dag_tasks["finalize-primeline"]["template"] == "finalize"
    assert dag_tasks["finalize-primeline"]["depends"] == "recognize-primeline.Succeeded"
    assert dag_tasks["publish-primeline"]["template"] == "publish"
    assert dag_tasks["publish-primeline"]["depends"] == "finalize-primeline.Succeeded"
    assert dag_tasks["finalize-primeline"]["arguments"]["parameters"][0]["value"] == "primeline"

    primeline = templates["recognize-primeline"]
    args = primeline["container"]["args"]
    assert "--asr" in args
    assert args[args.index("--asr") + 1] == "primeline"
    assert primeline["container"]["image"] == "{{workflow.parameters.primeline_image}}"
    assert primeline["container"]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    env = {
        variable["name"]: variable["value"] for variable in primeline["container"]["env"]
    }
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert primeline["outputs"]["artifacts"][0]["s3"]["key"] == (
        "runs/{{workflow.uid}}/asr/primeline/"
    )
    assert primeline["volumes"] == [
        {"name": "model-cache", "persistentVolumeClaim": {"claimName": "speech-model-cache"}}
    ]
    assert primeline["container"]["volumeMounts"][0] == {
        "name": "model-cache",
        "mountPath": "/models",
        "readOnly": True,
    }


def test_granite_argo_branch_uses_generic_image_gpu_recognition_and_common_finalizer() -> None:
    template_path = Path(__file__).parents[2] / "deploy/argo/transcription-workflowtemplate.yaml"
    document = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    templates = {template["name"]: template for template in document["spec"]["templates"]}
    parameters = {
        parameter["name"]: parameter["value"]
        for parameter in document["spec"]["arguments"]["parameters"]
    }

    # Granite uses the generic Transformers runtime image, not a dedicated one.
    assert parameters["granite_image"] == "ghcr.io/balex42/transcription-pipeline:sha-b7ce9fa"
    assert parameters["backends"] == (
        '["parakeet","primeline","qwen","nemotron","voxtral","faster-whisper","canary","granite"]'
    )

    validator = templates["validate-backends"]
    output_paths = {
        parameter["name"]: parameter["valueFrom"]["path"]
        for parameter in validator["outputs"]["parameters"]
    }
    assert output_paths["run_granite"] == "/tmp/run_granite"
    validator_script = validator["container"]["args"][0]
    assert '"granite"' in validator_script

    dag_tasks = {task["name"]: task for task in templates["transcription"]["dag"]["tasks"]}
    recognize = dag_tasks["recognize-granite"]
    assert recognize["template"] == "recognize-granite"
    assert recognize["depends"] == "prepare.Succeeded"
    assert (
        recognize["when"]
        == "{{tasks.validate-backends.outputs.parameters.run_granite}} == true"
    )
    assert dag_tasks["finalize-granite"]["template"] == "finalize"
    assert dag_tasks["finalize-granite"]["depends"] == "recognize-granite.Succeeded"
    assert dag_tasks["publish-granite"]["template"] == "publish"
    assert dag_tasks["publish-granite"]["depends"] == "finalize-granite.Succeeded"
    assert dag_tasks["finalize-granite"]["arguments"]["parameters"][0]["value"] == "granite"

    granite = templates["recognize-granite"]
    args = granite["container"]["args"]
    assert "--asr" in args
    assert args[args.index("--asr") + 1] == "granite"
    assert granite["container"]["image"] == "{{workflow.parameters.granite_image}}"
    assert granite["container"]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    env = {
        variable["name"]: variable["value"] for variable in granite["container"]["env"]
    }
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert granite["outputs"]["artifacts"][0]["s3"]["key"] == (
        "runs/{{workflow.uid}}/asr/granite/"
    )
    assert granite["volumes"] == [
        {"name": "model-cache", "persistentVolumeClaim": {"claimName": "speech-model-cache"}}
    ]
    assert granite["container"]["volumeMounts"][0] == {
        "name": "model-cache",
        "mountPath": "/models",
        "readOnly": True,
    }

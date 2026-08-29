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

    assert parameters["canary_image"] == "ghcr.io/balex42/transcription-pipeline-canary:sha-b52b410"

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

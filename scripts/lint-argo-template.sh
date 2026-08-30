#!/usr/bin/env bash
# Offline lint of the Argo WorkflowTemplate: schema and DAG-reference
# validation with no cluster and no credentials.
#
# The template's recording artifact has no storage location in the repository;
# it is supplied at submission time. Rendering the spec as a submit-time
# Workflow with a placeholder S3 artifact lets `argo lint --offline` resolve
# that argument and validate the template exactly as Argo does on submit.
#
# Usage: scripts/lint-argo-template.sh [path-to-argo-binary] [template.yaml]
set -euo pipefail

ARGO_BIN="${1:-argo}"
TEMPLATE="${2:-deploy/argo/transcription-workflowtemplate.yaml}"

RENDERED="$(mktemp /tmp/argo-lint-workflow-XXXXXX.yaml)"
trap 'rm -f "$RENDERED"' EXIT

python3 - "$TEMPLATE" > "$RENDERED" <<'PY'
import sys

import yaml

document = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
workflow = {
    "apiVersion": "argoproj.io/v1alpha1",
    "kind": "Workflow",
    "metadata": {"generateName": "lint-"},
    "spec": document["spec"],
}
artifact = workflow["spec"]["arguments"]["artifacts"][0]
if artifact.get("name") != "recording":
    raise SystemExit(f"expected the recording artifact first, got: {artifact.get('name')!r}")
artifact["s3"] = {
    "endpoint": "minio.example.invalid:9000",
    "bucket": "lint-bucket",
    "insecure": True,
    "accessKeySecret": {"name": "dummy", "key": "a"},
    "secretKeySecret": {"name": "dummy", "key": "s"},
    "addressingStyle": "path",
    "key": "lint/dummy",
}
yaml.safe_dump(workflow, sys.stdout, sort_keys=False, default_flow_style=False)
PY

# Unset server discovery so linting can never reach a cluster.
env -u ARGO_SERVER -u ARGO_TOKEN -u ARGO_SECURE -u ARGO_INSECURE_SKIP_VERIFY \
    "$ARGO_BIN" lint --offline "$RENDERED"
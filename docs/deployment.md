# Deploy private prediction inference

Build independently: `docker build --platform linux/amd64 -t ue6-ia:local .`.
Python 3.11 / TensorFlow 2.17.0 / TF-DF 1.10.0 follow the existing requirements.
The image contains application code, not student spreadsheets, datasets or trained models.
Do not train on the VPS. Training and dataset transfer are not part of deployment.

## Model and secret

Provide the complete, compatible local `models/tfdf_riesgo` artifact through an
explicitly authorized transfer later. Mount its parent as `/models:ro` and set
`UE6_MODELS_DIR=/models`. Include assets, variables, saved_model.pb and metadata.json;
metadata features must match this application's contract. Preserve permissions so
UID 10001 can read all files. Do not copy a model into a public container registry.
Keep artifacts outside Dokploy's disposable repository checkout and back them up
privately. Prefer a persistent named volume with a separately authorized provision
step, or an explicit host bind that is not recreated with a checkout.

Set `UE6_API_TOKEN_FILE` to a mounted secret path, or configure UE6_API_TOKEN through
Dokploy's protected environment. The API's PREDICTION_TOKEN must have the same value.
Neither secret is included in Git. The image binds 0.0.0.0:8001 inside its container;
never publish that port to the host or add a public domain for this service.

In Dokploy create an Application from THIS repository and its Dockerfile, running
Linux amd64. Attach a private network shared only with the API. Configure model and
secret mounts before deployment. Set the API's PREDICTION_URL to this service's
actual internal DNS name on port 8001. No sibling checkout is needed.

## Verification

The existing `/health` returns HTTP 200 even when `status` is `sin_modelo`.
The container healthcheck checks JSON status equals `ok` and does not log model data.
A configured token is still needed for prediction; health alone does not test it.
Confirm readiness, rejection of a missing/wrong token, and one synthetic prediction
before enabling the API sweep. A missing or incompatible artifact must remain unhealthy.

On 2026-10-19, the image built successfully and an isolated four-service Compose
stack reached healthy status. One token-backed API-to-IA synthetic prediction was
persisted in PostgreSQL. Full suites, missing/wrong-token and missing-model checks
were not rerun; production deployment has not been validated. This is a prototype;
container readiness does not establish model validity or calibration.

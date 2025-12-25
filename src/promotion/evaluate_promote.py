import os
import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "IrisClassifier"
METRIC_NAME = "accuracy"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", TRACKING_URI)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(REGISTRY_URI)

client = MlflowClient(
    tracking_uri=TRACKING_URI,
    registry_uri=REGISTRY_URI
)

def get_latest_model_version():
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError("No model versions found")

    return max(versions, key=lambda v: int(v.version))


def get_production_model():
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        if v.current_stage == "Production":
            return v
    return None


def get_run_metric(run_id: str, metric: str) -> float:
    run = client.get_run(run_id)
    return run.data.metrics.get(metric)


def promote_if_better():
    candidate = get_latest_model_version()
    candidate_metric = get_run_metric(candidate.run_id, METRIC_NAME)

    prod = get_production_model()

    if prod is None:
        print("No Production model found. Promoting candidate.")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=candidate.version,
            stage="Production",
            archive_existing_versions=True
        )
        return

    prod_metric = get_run_metric(prod.run_id, METRIC_NAME)

    print(f"Candidate accuracy: {candidate_metric}")
    print(f"Production accuracy: {prod_metric}")

    if candidate_metric > prod_metric:
        print("Candidate is better. Promoting to Production.")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=candidate.version,
            stage="Production",
            archive_existing_versions=True
        )
    else:
        print("Candidate is not better. Keeping current Production model.")


if __name__ == "__main__":
    promote_if_better()

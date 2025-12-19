# My MLOps Platform

An end-to-end MLOps platform built step-by-step:
- Experiment tracking (MLflow)
- Model training
- Model serving
- Monitoring & drift detection

This repository is structured to mirror real-world production MLOps systems.



-encountered issue during minio connection:
    1. created .env in root
    2. run Get-Content .env | ForEach-Object {
  if ($_ -match "(.+?)=(.+)") {
    [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
  }
}
 in powershell


 -configured minio:
    - for artifacts logs; after docker is up; create a bucket named 'mlflow-artifacts' in minio(port:9001). Then run the training script
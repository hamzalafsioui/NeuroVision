FROM apache/airflow:3.3.2-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-postgres \
    mlflow==3.14.0

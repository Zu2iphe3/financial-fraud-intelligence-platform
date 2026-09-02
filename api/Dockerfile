FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app

WORKDIR /opt/app

COPY api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt \
    && useradd --create-home --uid 10001 apiuser

# The model is not copied into image history. Compose mounts the ignored local
# artifact read-only after it has been promoted from the reviewed MLflow run.
COPY --chown=apiuser:apiuser src /opt/app/src
COPY --chown=apiuser:apiuser config /opt/app/config

USER apiuser

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

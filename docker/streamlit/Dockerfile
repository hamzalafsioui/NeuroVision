FROM python:3.11-slim

WORKDIR /app

COPY docker/streamlit/requirements.txt requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

COPY streamlit_app/ ./streamlit_app/

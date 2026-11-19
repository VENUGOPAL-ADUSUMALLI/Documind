FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py api_client.py ./

RUN useradd --create-home appuser
USER appuser

# Railway sets PORT; 8501 is Streamlit's usual port for running it locally.
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]

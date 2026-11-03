import os

import httpx

API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000")

# Uploading and answering call external AI APIs, so allow generous time.
TIMEOUT = 120


class ApiError(Exception):
    """A readable error message safe to show in the UI."""


def _request(method, path, **kwargs):
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.HTTPError:
        raise ApiError(f"Cannot reach the DocuMind API at {API_URL}. Is the backend running?")

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(str(detail))
    return response


def list_documents():
    return _request("GET", "/documents").json()


def upload_document(filename, data):
    files = {"file": (filename, data, "application/pdf")}
    return _request("POST", "/documents", files=files).json()


def delete_document(document_id):
    _request("DELETE", f"/documents/{document_id}")


def ask(question, document_id=None):
    payload = {"question": question, "document_id": document_id}
    return _request("POST", "/ask", json=payload).json()

import streamlit as st

import api_client
from api_client import ApiError

st.set_page_config(page_title="DocuMind", page_icon="📚", layout="wide")

STATUS_ICONS = {"done": "✅", "processing": "⏳", "pending": "⏳", "failed": "❌"}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def render_citation_warning(invalid_citations):
    """Tell the user when the model cited pages it was never shown (they were removed)."""
    if invalid_citations:
        st.warning(
            "Some citations could not be verified and were removed: "
            + "; ".join(invalid_citations)
        )


def render_sources(sources):
    with st.expander(f"📎 Sources · {len(sources)}"):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"**{number}. {source['filename']}** · page {source['page']} "
                f"· score {source['score']}"
            )
            st.text(source["excerpt"])


# --------------------------------------------------
# Sidebar: documents
# --------------------------------------------------

with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("Ask questions about your PDFs.")

    try:
        documents = api_client.list_documents()
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Add documents")
    files = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    if files and st.button("Upload", type="primary", use_container_width=True):
        with st.spinner("Reading, chunking and embedding..."):
            for file in files:
                try:
                    api_client.upload_document(file.name, file.getvalue())
                except ApiError as exc:
                    st.error(f"{file.name}: {exc}")
                    break
            else:
                st.session_state.uploader_key += 1
                st.rerun()

    st.subheader("Your documents")
    if not documents:
        st.caption("No documents yet.")

    for document in documents:
        icon = STATUS_ICONS.get(document["status"], "•")
        name_col, delete_col = st.columns([5, 1])
        name_col.markdown(f"{icon} **{document['filename']}**")
        name_col.caption(f"{document['page_count']} pages · {document['status']}")
        if delete_col.button("🗑", key=f"delete_{document['id']}", help="Delete"):
            try:
                api_client.delete_document(document["id"])
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
        if document["status"] == "failed" and document["error"]:
            st.caption(f"⚠️ {document['error']}")

    ready = [d for d in documents if d["status"] == "done"]
    options = {"All documents": None}
    options.update({d["filename"]: d["id"] for d in ready})
    scope = st.selectbox("Search in", list(options))

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Main: conversation
# --------------------------------------------------

st.header("Document assistant")

if not ready:
    st.info("Upload a PDF in the sidebar to get started.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_citation_warning(message.get("invalid_citations"))
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask anything about your documents...", max_chars=2000, disabled=not ready)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents..."):
                result = api_client.ask(question, options[scope])
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.markdown(result["answer"])
            render_citation_warning(result.get("invalid_citations"))
            if result["sources"]:
                render_sources(result["sources"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "invalid_citations": result.get("invalid_citations", []),
                }
            )

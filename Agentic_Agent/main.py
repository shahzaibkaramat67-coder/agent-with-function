# from pathlib import Path
import os

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from Agentic_Agent.piplimimg import entryPoint
from Tools.Rag.system_setup import createChromaDB
from fileLoder.fileLoder import file_loaders

load_dotenv()

os.environ["USER_AGENT"] = "my-streamlit-app"


def ensure_api_keys():
    if st.session_state.get("gemini_api_key"):
        key = st.session_state.gemini_api_key.strip()
        os.environ["GEMINI_NEW_API_KEY"] = key
    if st.session_state.get("tavily_api_key"):
        os.environ["TAVILY_API_KEY"] = st.session_state.tavily_api_key
    if st.session_state.get("embedding_model"):
        os.environ["EMBADDING_MODEL_SENTENCE"] = st.session_state.embedding_model


def run_query(question: str):
    try:
        result = entryPoint(question)
        return result
    except Exception as exc:
        return {
            "final_answer": f"An error occurred while processing your request: {exc}",
            "steps": [],
            "status": "failed",
        }


def main():
    st.set_page_config(
        page_title="Agentic AI Assistant",
        page_icon="🤖",
        layout="wide",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False

    st.sidebar.title("⚙️ Configuration")
    st.session_state.gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_NEW_API_KEY", ""),
        type="password",
    )
    st.session_state.tavily_api_key = st.sidebar.text_input(
        "Tavily API Key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
    )
    st.session_state.embedding_model = st.sidebar.text_input(
        "Embedding Model",
        value=os.getenv(
            "EMBADDING_MODEL_SENTENCE", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    ensure_api_keys()

    st.title("🤖 Agentic AI Assistant")
    st.caption("Deploy-ready document and web agent built with Streamlit.")

    uploaded_file = st.file_uploader(
        "Upload a document to build the RAG index",
        type=["pdf", "txt", "docx"],
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if uploaded_file is not None:
            st.write(f"Selected file: {uploaded_file.name}")
    with col2:
        if st.button("Build document index", use_container_width=True):
            if uploaded_file is None:
                st.warning("Please select a file before indexing.")
            else:
                try:
                    save_dir = Path("uploads")
                    save_dir.mkdir(exist_ok=True)
                    file_path = save_dir / uploaded_file.name
                    file_path.write_bytes(uploaded_file.getvalue())

                    documents = file_loaders(str(file_path))
                    if documents is None:
                        st.error(
                            "Unsupported file type. Please upload PDF, TXT, or DOCX."
                        )
                    else:
                        createChromaDB(documents)
                        st.session_state.index_ready = True
                        st.success(
                            "Document indexed successfully. You can now ask the agent questions."
                        )
                except Exception as exc:
                    st.error(f"Failed to index the document: {exc}")

    if not st.session_state.index_ready:
        st.info(
            "Upload and index a document to enable RAG search, or ask a general question using web search."
        )

    st.subheader("Ask a question")
    question = st.text_area("Your query", height=120, placeholder="Ask anything...")

    if st.button("Run agent", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Processing your request..."):
                result = run_query(question)

            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append(
                {"role": "assistant", "content": result["final_answer"]}
            )

            st.markdown("### Final answer")
            st.write(result["final_answer"])

            if result.get("steps"):
                with st.expander("Agent steps", expanded=False):
                    for step in result["steps"]:
                        st.markdown(f"#### Iteration {step.get('iteration', '')}")
                        st.json(step)

    if st.session_state.messages:
        st.subheader("Conversation")
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
                st.markdown("---")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Streamlit chat UI for the Customer Support RAG.

Run with:
    streamlit run chatbot.py
"""
import os
import glob

import streamlit as st
from dotenv import load_dotenv

from src.rag_system import CustomerSupportRAG

load_dotenv()

st.set_page_config(page_title="Customer Support Chatbot", page_icon="💬")


@st.cache_resource(show_spinner="Initializing RAG system and loading knowledge base...")
def get_rag_system() -> CustomerSupportRAG:
    api_key = os.getenv("API_KEY")
    if not api_key:
        st.error("API_KEY is not set. Add it to your .env file and restart.")
        st.stop()

    rag = CustomerSupportRAG(api_key=api_key)

    csv_path = os.path.join("data", "test_kb.csv")
    if os.path.exists(csv_path):
        rag.load_knowledge_base_from_csv(csv_path)

    for pdf in glob.glob(os.path.join("data", "*.pdf")):
        rag.load_knowledge_base_from_pdf(pdf)

    return rag


def render_sources(result: dict) -> None:
    sources = result.get("sources", [])
    confidence = result.get("confidence", "unknown")
    tokens = result.get("total_tokens")

    meta_bits = [f"Confidence: **{confidence}**", f"Sources: **{len(sources)}**"]
    if tokens is not None:
        meta_bits.append(f"Tokens: **{tokens}**")
    st.caption(" · ".join(meta_bits))

    if sources:
        with st.expander("View sources"):
            for i, src in enumerate(sources, 1):
                score = src.get("relevance_score")
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                st.markdown(
                    f"**{i}. [{src.get('category', 'n/a')}]** "
                    f"{src.get('question', '')}  \n"
                    f"Relevance: `{score_str}`"
                )


def main() -> None:
    st.title("💬 RAG Chatbot")
    st.caption("Ask questions about the knowledge base loaded from `data/`.")

    rag = get_rag_system()

    with st.sidebar:
        st.subheader("Session")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Load a PDF")
        pdf_path = st.text_input(
            "Absolute path to PDF",
            placeholder="/Users/you/Downloads/MedicalCard.pdf",
        )
        if st.button("Ingest PDF"):
            if not pdf_path or not os.path.isfile(pdf_path):
                st.error("File not found at that path.")
            else:
                with st.spinner(f"Ingesting {os.path.basename(pdf_path)}..."):
                    rag.load_knowledge_base_from_pdf(pdf_path)
                st.success(f"Loaded {os.path.basename(pdf_path)} into knowledge base.")

        st.divider()
        st.markdown("**Model**")
        st.code(rag.llm_model, language=None)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "result" in msg:
                render_sources(msg["result"])

    prompt = st.chat_input("Ask a question...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.generate_response(prompt)
        answer = result.get("response", "No response")
        st.markdown(answer)
        render_sources(result)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "result": result}
    )


if __name__ == "__main__":
    main()

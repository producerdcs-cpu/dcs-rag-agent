"""
DCS RAG Agent — Streamlit demo
Run: streamlit run src/app_streamlit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import run_agent

st.set_page_config(page_title="DCS RAG Agent", page_icon="🤖", layout="centered")

st.title("🤖 DCS RAG Agent")
st.caption("LangGraph · Tools · Memory · Multi-step · DcsProducer®")

with st.sidebar:
    st.header("Controles")
    st.markdown(
        """
        **Ferramentas disponíveis**
        - `search_knowledge_base` — busca RAG
        - `calculate` — cálculos simples

        **LLM**  
        Configure `LLM_PROVIDER` no `.env`
        """
    )

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Pergunte ao agente...")

if question:
    st.session_state.history.append(("user", question))
    with st.spinner("Agente pensando + usando tools..."):
        try:
            answer = run_agent(question)
        except Exception as e:
            answer = f"Erro: {e}"
    st.session_state.history.append(("assistant", answer))

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

st.markdown("---")
st.markdown(
    "<small>© DcsProducer® · <a href='https://github.com/producerdcs-cpu/dcs-rag-agent'>GitHub</a></small>",
    unsafe_allow_html=True,
)

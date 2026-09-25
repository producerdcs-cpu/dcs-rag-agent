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
st.caption("LangGraph · Tools · Memory (checkpointer) · Multi-step · LAB · DcsProducer®")

with st.sidebar:
    st.header("Controles")
    thread_id = st.text_input("Thread ID (memória)", value="default")
    show_traj = st.checkbox("Mostrar trajetória", value=False)
    st.markdown(
        """
        **Ferramentas**
        - `search_knowledge_base` — Chroma real (ou mock)
        - `calculate` — cálculos simples

        **LLM** — configure `LLM_PROVIDER` no `.env`
        """
    )

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Pergunte ao agente...")

if question:
    st.session_state.history.append(("user", question))
    with st.spinner("Agente pensando + tools + memória..."):
        try:
            if show_traj:
                out = run_agent(question, thread_id=thread_id, return_trajectory=True)
                answer = out["answer"]
                traj_info = f"\n\n_Trajetória: {out['n_messages']} msgs · tools={out['used_tools']}_"
                answer = answer + traj_info
            else:
                answer = run_agent(question, thread_id=thread_id)
        except Exception as e:
            answer = f"Erro: {e}"
    st.session_state.history.append(("assistant", answer))

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

st.markdown("---")
st.markdown(
    "<small>© DcsProducer® · LAB · "
    "<a href='https://github.com/producerdcs-cpu/dcs-rag-agent'>GitHub</a></small>",
    unsafe_allow_html=True,
)

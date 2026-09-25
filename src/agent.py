"""
DCS RAG Agent — Core agent with LangGraph
Retrieval real (Chroma) + Tools + Memory (checkpointer) + Multi-step + Trajectory eval

Sequência do starter (dcs-rag-starter). Classe: LAB.
Fonte de verdade: Lista_Projetos.xlsx + pdf-wordpress-editor + Hub canônico.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# Config (alinhado ao starter)
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dcs_rag_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: Optional[str]


# ---------------------------------------------------------------------------
# Real Chroma retrieval (fallback to mock if Chroma/deps missing or empty)
# ---------------------------------------------------------------------------
def _get_retriever():
    """Tenta carregar Chroma do starter. Retorna None se indisponível."""
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        if not Path(CHROMA_PERSIST_DIR).exists():
            return None

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vs = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        return vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    except Exception:
        return None


@tool
def search_knowledge_base(query: str) -> str:
    """Busca na base de conhecimento RAG (Chroma real do starter) e retorna trechos relevantes."""
    retriever = _get_retriever()
    if retriever is None:
        return (
            f"[Fallback mock — Chroma não encontrado em {CHROMA_PERSIST_DIR}]\n"
            "RAG combina recuperação de informação com geração de texto por LLMs. "
            "Pipeline típico: ingestão → chunking → embeddings → vector store → retrieval → geração. "
            "Execute ingestão no dcs-rag-starter para ativar retrieval real."
        )
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "Nenhum trecho relevante encontrado na base."
        parts = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "n/a")
            parts.append(f"[Chunk {i} | {src}]\n{d.page_content[:800]}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Erro no retrieval: {e}"


@tool
def calculate(expression: str) -> str:
    """Avalia uma expressão matemática simples (seguro para demo)."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Expressão contém caracteres não permitidos."
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Erro no cálculo: {e}"


TOOLS = [search_knowledge_base, calculate]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "mock").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            temperature=0.1,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
        )

    from langchain_core.language_models.fake import FakeListLLM
    return FakeListLLM(
        responses=[
            "Resposta do agente (modo mock). Configure LLM_PROVIDER=ollama|groq|openai e rode ingestão no starter para retrieval real."
        ]
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Você é um agente técnico GenAI da DcsProducer® (classe LAB).
Você tem acesso a ferramentas. Use-as quando necessário para responder com precisão.
Sempre fundamentar respostas no contexto recuperado quando disponível.
Se não souber, diga claramente.
"""


def agent_node(state: AgentState):
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS) if hasattr(llm, "bind_tools") else llm
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph + Checkpointer (memória de thread)
# ---------------------------------------------------------------------------
_memory = MemorySaver()  # in-memory; troque por SqliteSaver para persistência em disco


def build_agent_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer or _memory)


def run_agent(
    question: str,
    thread_id: str = "default",
    return_trajectory: bool = False,
) -> str | Dict[str, Any]:
    """
    Executa o agente com memória por thread_id.
    Se return_trajectory=True, retorna dict com resposta + passos (para avaliação).
    """
    app = build_agent_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke(
        {"messages": [HumanMessage(content=question)], "context": None},
        config=config,
    )
    messages = result["messages"]
    last = messages[-1]
    answer = last.content if hasattr(last, "content") else str(last)

    if not return_trajectory:
        return answer

    # Trajectory simples para avaliação
    steps = []
    for m in messages:
        role = m.__class__.__name__
        content = getattr(m, "content", str(m))[:500]
        tool_calls = getattr(m, "tool_calls", None)
        steps.append({
            "role": role,
            "content_preview": content,
            "has_tool_calls": bool(tool_calls),
            "tool_names": [tc.get("name") for tc in (tool_calls or [])] if tool_calls else [],
        })

    return {
        "answer": answer,
        "thread_id": thread_id,
        "n_messages": len(messages),
        "trajectory": steps,
        "used_tools": any(s["has_tool_calls"] for s in steps),
    }


# ---------------------------------------------------------------------------
# Avaliação de trajetória (leve, sem deps extras)
# ---------------------------------------------------------------------------
def evaluate_trajectory(traj: Dict[str, Any], expected_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Avalia uma trajetória do agente:
    - used_tools: se usou alguma tool
    - tool_correctness: se as tools esperadas foram chamadas
    - length: número de passos
    - final_answer_present: se há resposta final
    """
    steps = traj.get("trajectory", [])
    used = traj.get("used_tools", False)
    tool_names_called = []
    for s in steps:
        tool_names_called.extend(s.get("tool_names") or [])

    expected = expected_tools or []
    if expected:
        hit = sum(1 for t in expected if t in tool_names_called)
        tool_correctness = hit / len(expected)
    else:
        tool_correctness = 1.0 if used else 0.0

    return {
        "thread_id": traj.get("thread_id"),
        "used_tools": used,
        "tools_called": list(set(tool_names_called)),
        "tool_correctness": round(tool_correctness, 3),
        "n_steps": len(steps),
        "final_answer_present": bool(traj.get("answer")),
        "answer_preview": (traj.get("answer") or "")[:200],
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="DCS RAG Agent CLI")
    parser.add_argument("--ask", type=str, help="Pergunta para o agente")
    parser.add_argument("--thread", type=str, default="default", help="Thread ID para memória")
    parser.add_argument("--trajectory", action="store_true", help="Retornar trajetória completa")
    parser.add_argument("--eval", action="store_true", help="Avaliar trajetória (implica --trajectory)")
    parser.add_argument("--expected-tools", type=str, default="", help="Tools esperadas (csv)")
    args = parser.parse_args()

    if not args.ask:
        parser.error("--ask é obrigatório")

    want_traj = args.trajectory or args.eval
    out = run_agent(args.ask, thread_id=args.thread, return_trajectory=want_traj)

    if want_traj:
        print("\n=== Resposta ===")
        print(out["answer"])
        print("\n=== Trajetória (resumo) ===")
        for i, s in enumerate(out["trajectory"], 1):
            tools = f" tools={s['tool_names']}" if s["has_tool_calls"] else ""
            print(f"  {i}. {s['role']}{tools}")
        if args.eval:
            expected = [t.strip() for t in args.expected_tools.split(",") if t.strip()]
            metrics = evaluate_trajectory(out, expected_tools=expected or None)
            print("\n=== Avaliação de trajetória ===")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print("\n=== Resposta do Agente ===")
        print(out)

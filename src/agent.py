"""
DCS RAG Agent — Core agent with LangGraph
Retrieval + Tools + Memory + Multi-step reasoning

Aligned with GenAI Agent / RAG Engineer requirements.
"""

from __future__ import annotations

import os
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: Optional[str]


# ---------------------------------------------------------------------------
# Tools (exemplos — expanda conforme necessidade)
# ---------------------------------------------------------------------------
@tool
def search_knowledge_base(query: str) -> str:
    """Busca na base de conhecimento RAG (Chroma) e retorna trechos relevantes."""
    # Placeholder: em produção, chama o retriever do dcs-rag-starter
    return (
        f"[Mock retrieval para: '{query}']\n"
        "RAG combina recuperação de informação com geração de texto por LLMs. "
        "Pipeline típico: ingestão → chunking → embeddings → vector store → retrieval → geração."
    )


@tool
def calculate(expression: str) -> str:
    """Avalia uma expressão matemática simples (seguro para demo)."""
    try:
        # Apenas operações básicas para demo
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Expressão contém caracteres não permitidos."
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Erro no cálculo: {e}"


TOOLS = [search_knowledge_base, calculate]


# ---------------------------------------------------------------------------
# LLM factory (mesmo padrão do starter)
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

    # Mock
    from langchain_core.language_models.fake import FakeListLLM
    return FakeListLLM(
        responses=[
            "Resposta do agente (modo mock). Configure LLM_PROVIDER=ollama|groq|openai."
        ]
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Você é um agente técnico GenAI da DcsProducer®.
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
# Graph
# ---------------------------------------------------------------------------
def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent(question: str) -> str:
    app = build_agent_graph()
    result = app.invoke({"messages": [HumanMessage(content=question)], "context": None})
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DCS RAG Agent CLI")
    parser.add_argument("--ask", type=str, required=True, help="Pergunta para o agente")
    args = parser.parse_args()

    print("\n=== Resposta do Agente ===")
    print(run_agent(args.ask))

# DCS RAG Agent

**Classe: LAB** (portfólio GenAI). **Não é AEGIS.** Não substitui o Hub `dcsproducer-hub`.
Ver `GOVERNANCE.md` — opções: manter LAB ou privatizar.

Agente RAG com LangGraph — tools + memória + multi-step.
Sequência do starter privado `dcs-rag-starter`.

Fonte de verdade de projetos: planilha + skill `pdf-wordpress-editor` + Hub canônico.

```
Pergunta → Agent Node → (Tools?) → Retrieval / Cálculo → Resposta
```

## Quick start

```bash
git clone https://github.com/producerdcs-cpu/dcs-rag-agent.git
cd dcs-rag-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.agent --ask "O que é RAG?"
```

Não commitar `.env` com chaves reais.

MIT © 2026 DcsProducer®

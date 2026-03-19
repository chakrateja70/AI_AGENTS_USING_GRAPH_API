# AI Agents Using Graph API

Minimal examples of building LLM agents with **LangGraph** (graph-based orchestration) and calling LLMs via the **OpenAI Responses API** / **LangChain ChatOpenAI**. The repo currently includes:

- A simple 2-node LangGraph chat flow (`main.py`)
- A tool-using LangGraph agent loop with Tavily + arXiv tools (`phase2.py`)
- Experimental scripts for Groq and Google Gemini in `models/`

> Note: This repository is currently a collection of scripts/demos (not a packaged library).

---

## Repo structure

```
.
├── main.py
├── phase2.py
├── models
│   ├── groq.py
│   └── llm_gemini.py
├── agent_graph.png
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .python-version
```

---

## Requirements

- Python **3.11** (see `.python-version`)

Install dependencies either with `pip` or a modern tool that reads `pyproject.toml`:

### Option A: pip + requirements.txt

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: pip (pyproject.toml)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

---

## Environment variables

Copy `.env.example` to `.env` and fill in keys:

```bash
cp .env.example .env
```

Expected variables:

- `OPENAI_API_KEY` (required for `main.py` and `phase2.py`)
- `TAVILY_API_KEY` (required only for the Tavily tool in `phase2.py`)
- `GROQ_API_KEY` (present in `.env.example`; used by `models/groq.py` if you wire it up)

---

## Examples

### 1) Simple LangGraph chat flow (OpenAI Responses API)

`main.py` builds a tiny graph:

- `user_input_node` → appends user input to the message list
- `agent_node` → calls OpenAI and returns the assistant response

Run:

```bash
python main.py
```

The script currently invokes the graph with a hardcoded prompt:

- `"what is my previous question?"`

Update the `app.invoke({...})` call to change the prompt.

---

### 2) Tool-using agent loop (LangGraph + LangChain tools)

`phase2.py` demonstrates an agent that can:

- Call a toy `multiply(a, b)` tool
- Call Tavily search (`tavily_tool_run`) for fresh web results
- Call arXiv search (`arxiv_tool_run`) for research papers

It also saves a visualization of the agent graph to `agent_graph.png`.

Run:

```bash
python phase2.py
```

The script currently runs with a hardcoded question:

- `"what is latest news about iran and israel?"`

---

## Notes / known issues

- `models/llm_gemini.py` contains a hardcoded API key in the current codebase. Consider moving it to an environment variable and removing the key from git history.
- `models/groq.py` appears to be an experimental snippet and may need fixes (e.g., import path / correct Groq SDK usage) before it will run.

---

## License

Add a LICENSE file if you plan to open-source this project.
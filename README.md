# Codebase Q&A Agent

An AI-powered agent that indexes any codebase and answers questions about it from memory using RAG (Retrieval-Augmented Generation). Built on [Agentfield](https://agentfield.ai).

## How It Works

```
Index Phase (no LLM):    Scan → Extract Symbols → Chunk at Boundaries → BM25 Keywords → SQLite
Query Phase (LLM):       Question → Hybrid Retrieval → Context Assembly → Groq LLM → Answer
```

- **Indexing is instant** — pure Python, no API calls, no embeddings required
- **Queries use RAG** — BM25 + symbol matching + optional semantic search → only relevant code goes to the LLM
- **Supports 14 languages** — Python, JS, TS, Go, Java, C#, Rust, Ruby, PHP, C/C++, Swift

## Features

- Multi-project indexing with per-project SQLite databases
- Conversation sessions with follow-up context
- GitHub URL clone-and-index
- Project slug/ID system (no raw filesystem paths in API)
- Persistent vector embeddings (optional, requires sentence-transformers)
- Tree-sitter AST parsing with regex fallback
- File watcher for automatic re-indexing
- Code explorer endpoint for fetching source files
- Security hardened: binary detection, path traversal prevention, symlink guards

## Quick Start

```bash
# 1. Start the Agentfield server
af server

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY=your_key_here

# 4. Run the agent
af run codebase-qa-agent
```

## API Endpoints

### Indexing

**Index a local project:**
```bash
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_index_project \
  -H "Content-Type: application/json" \
  -d '{"input": {"project_path": "/home/you/myproject"}}'
```

**Clone and index from GitHub:**
```bash
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_clone_and_index \
  -H "Content-Type: application/json" \
  -d '{"input": {"github_url": "https://github.com/owner/repo"}}'
```

**Incremental update:**
```bash
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_update_index \
  -H "Content-Type: application/json" \
  -d '{"input": {"project_path": "/home/you/myproject"}}'
```

**Delete a project** (accepts path, slug, or project_id):
```bash
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_delete_project \
  -H "Content-Type: application/json" \
  -d '{"input": {"project_identifier": "myproject"}}'
```

### Querying

**Ask a question:**
```bash
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_answer_question \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "How does authentication work?", "session_id": "s1"}}'
```

**Find relevant files** (no LLM, instant):
```bash
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_find_relevant_files \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "database connection"}}'
```

**Get file source code:**
```bash
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_get_file_content \
  -H "Content-Type: application/json" \
  -d '{"input": {"file_path": "src/main.py"}}'
```

**List all indexed projects:**
```bash
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_list_projects \
  -H "Content-Type: application/json" \
  -d '{"input": {}}'
```

### File Watching

```bash
# Start watching for changes
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_watch_project \
  -H "Content-Type: application/json" \
  -d '{"input": {"project_path": "/home/you/myproject"}}'

# Stop watching
curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_unwatch_project \
  -H "Content-Type: application/json" \
  -d '{"input": {"project_path": "/home/you/myproject"}}'
```

## Project Structure

```
main.py                  # Agent config and startup
reasoners/
  indexer.py             # Index, update, clone, delete, watch endpoints
  qa.py                  # Answer, find files, get content, list projects
skills/
  scanner.py             # Directory walker with security guards
  extractor.py           # Symbol extraction (AST + regex), chunking, keywords
  storage.py             # SQLite persistence, sessions, project management
  embeddings.py          # Optional vector embeddings (sentence-transformers)
  ast_parser.py          # Tree-sitter AST parsing (optional)
  watcher.py             # File watcher for auto re-indexing (optional)
  git_ops.py             # GitHub clone operations
tests/                   # Unit tests (48 tests)
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Tech Stack

- **Framework:** Agentfield
- **LLM:** Groq (llama-3.3-70b-versatile) — free tier, sub-second inference
- **Storage:** SQLite with WAL mode, per-project databases
- **Retrieval:** BM25 IDF + symbol boosting + optional semantic similarity
- **Languages:** Python 3.12+

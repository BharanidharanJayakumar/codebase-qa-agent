import os
from pathlib import Path
from dotenv import load_dotenv
from agentfield import Agent, AIConfig, MemoryConfig

# Always load .env from this file's directory, not the CWD
load_dotenv(Path(__file__).resolve().parent / ".env")

from reasoners.indexer import indexer_router
from reasoners.qa import qa_router

app = Agent(
    node_id="codebase-qa-agent",
    agentfield_server=os.getenv("AGENTFIELD_CONTROL_PLANE_URL", "http://localhost:8080"),
    version="1.0.0",
    dev_mode=True,

    # Groq — free tier, extremely fast inference (~1-3s responses)
    # Get your free API key at: https://console.groq.com
    # Set it with: export GROQ_API_KEY=your_key_here
    ai_config=AIConfig(
        model="groq/llama-3.3-70b-versatile",
    ),
    # Persistent memory so the index survives across executions
    # Default is "session" which clears data when each request finishes
    memory_config=MemoryConfig(auto_inject=[], memory_retention="persistent", cache_results=False),
)

# Register both routers with the agent
# This is like mounting blueprints in Flask or routers in FastAPI
app.include_router(indexer_router)
app.include_router(qa_router)

if __name__ == "__main__":
    app.serve(auto_port=True, dev=True, reload=False)

import os
from agentfield import Agent, AIConfig

# We import both routers we'll build — indexing logic and Q&A logic
# kept in separate files so each has one clear responsibility
from reasoners.indexer import indexer_router
from reasoners.qa import qa_router

app = Agent(
    node_id="codebase-qa-agent",
    agentfield_server=os.getenv("AGENTFIELD_CONTROL_PLANE_URL", "http://localhost:8080"),
    version="1.0.0",
    dev_mode=True,

    # Ollama runs locally — no API key needed, no cost
    # LiteLLM (used under the hood by Agentfield) understands "ollama/model-name"
    # api_base tells it where Ollama is running on your machine
    ai_config=AIConfig(
        model="ollama/llama3.2",
        api_base="http://localhost:11434",
    ),
)

# Register both routers with the agent
# This is like mounting blueprints in Flask or routers in FastAPI
app.include_router(indexer_router)
app.include_router(qa_router)

if __name__ == "__main__":
    app.serve(auto_port=True, dev=True, reload=False)

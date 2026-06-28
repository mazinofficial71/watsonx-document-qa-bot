import os

# --- WatsonX Settings ---
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
PROJECT_ID  = os.getenv("WATSONX_PROJECT_ID", "skills-network")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")

# --- Model IDs ---
LLM_MODEL_ID       = "ibm/granite-4-h-small"
EMBEDDING_MODEL_ID = "ibm/granite-embedding-135m-en"

# --- LLM Parameters ---
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.1

# --- Text Splitter ---
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100

# --- Embedding ---
EMBED_TRUNCATE_TOKENS = 512

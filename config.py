"""Application configuration driven by environment variables.

All settings have sensible defaults for local development.
Copy ``.env.example`` to ``.env`` and adjust values for your environment.
"""

import os

# ---------------------------------------------------------------------------
# Python gRPC server (C# connects to us on this address)
# ---------------------------------------------------------------------------

GRPC_SERVER_HOST: str = os.getenv("GRPC_SERVER_HOST", "0.0.0.0")
GRPC_SERVER_PORT: int = int(os.getenv("GRPC_SERVER_PORT", "50051"))

# Thread pool size for the gRPC server.  Each concurrent RPC occupies one
# thread, so this caps concurrent request handling.
GRPC_MAX_WORKERS: int = int(os.getenv("GRPC_MAX_WORKERS", "10"))

# ---------------------------------------------------------------------------
# C# story / state server (we connect to it as a gRPC client)
# ---------------------------------------------------------------------------

CSHARP_SERVER_ADDRESS: str = os.getenv("CSHARP_SERVER_ADDRESS", "localhost:50052")

# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

NUM_RECOMMENDATIONS: int = 6       # total slots returned per request
NUM_CONTENT_BASED: int = 2         # content-based filtering slots
NUM_COLLABORATIVE: int = 2         # collaborative filtering slots
NUM_TOPICAL: int = 1               # topical (tag-based) slots
NUM_WILDCARD: int = 1              # wildcard / discovery slots

# ---------------------------------------------------------------------------
# Story catalogue cache
# ---------------------------------------------------------------------------

# How often (seconds) to refresh the story catalogue from the C# server.
CATALOGUE_REFRESH_INTERVAL_SECONDS: int = int(
    os.getenv("CATALOGUE_REFRESH_INTERVAL_SECONDS", "300")
)

# ---------------------------------------------------------------------------
# User state persistence
# ---------------------------------------------------------------------------

# How often (seconds) to persist all user state to the C# server.
STATE_PERSIST_INTERVAL_SECONDS: int = int(
    os.getenv("STATE_PERSIST_INTERVAL_SECONDS", "60")
)

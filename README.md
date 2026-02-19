# Story Recommender — Python gRPC Service

A Python recommender system prototype that integrates with a C# app server via gRPC. It tracks user interactions with stories, maintains per-user preference profiles in memory, and returns 6 personalised recommendations within 500 ms.

## Architecture

```
C# App Server ──gRPC──► Python Recommender (this repo)
      ▲                         │
      └────────gRPC─────────────┘
         (story catalogue + state persistence)
```

**Python acts as a gRPC server** — the C# client sends user events and recommendation requests to Python.

**Python also acts as a gRPC client** — it calls back to C# to fetch the story catalogue and persist/load user state as event logs.

### Recommendation slots (6 total, returned in random order)

| # | Strategy | Algorithm |
|---|---|---|
| 2 | Content-based | Cosine similarity: user theme+tag weight vector vs story indicator vectors |
| 2 | Collaborative filtering | User-user cosine similarity; aggregate stories from top-20 similar users |
| 1 | Topical | User's highest-weight tag → best matching unviewed stories |
| 1 | Wildcard | Random, preferring stories in unexplored themes |

### User preference model

Every interaction updates a per-user theme/tag weight vector:

| Event | Weight delta per theme/tag |
|---|---|
| Viewed | +1.0 |
| Completed | +2.0 (additive with viewed) |
| Scored 1–5 | `(score − 3) × 0.5` |
| Mood 1–5 | stored only, no weight impact |

State is persisted to the C# server as a raw event log and replayed on startup.

## Project Structure

```
proto/                    gRPC service definitions (.proto)
generated/                protoc-generated Python bindings (not committed)
recommender/
  models.py               Story, UserEvent, UserProfile dataclasses
  catalogue.py            StoryCatalogue — fetches & caches stories from C#
  user_state.py           UserStateStore — event ingestion, weights, persistence
  engine.py               RecommendationEngine — orchestrates 6 slots
  service.py              RecommenderServicer — gRPC servicer
  strategies/             Four strategy implementations
main.py                   Entry point
config.py                 Environment-variable configuration
tests/                    pytest test suite (157 tests)
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate gRPC Python bindings from the proto file
make proto

# 3. Configure environment (copy and edit)
cp .env.example .env
```

## Running

```bash
# Start the recommender server (requires a running C# server)
python main.py

# Or with explicit config
CSHARP_SERVER_ADDRESS=localhost:50052 GRPC_SERVER_PORT=50051 python main.py
```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage report
pytest --cov=recommender tests/
```

## Configuration

All settings are driven by environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `GRPC_SERVER_HOST` | `0.0.0.0` | Python gRPC listen address |
| `GRPC_SERVER_PORT` | `50051` | Python gRPC listen port |
| `CSHARP_SERVER_ADDRESS` | `localhost:50052` | C# server address |
| `GRPC_MAX_WORKERS` | `10` | gRPC thread pool size |
| `CATALOGUE_REFRESH_INTERVAL_SECONDS` | `300` | How often to refresh the story catalogue |
| `STATE_PERSIST_INTERVAL_SECONDS` | `60` | How often to persist user state to C# |

## gRPC Interface

### RecommenderService (Python as server — C# calls these)

```protobuf
rpc UserViewedStory(...)       returns (Empty);   // fire-and-forget
rpc UserCompletedStory(...)    returns (Empty);   // fire-and-forget
rpc UserAnsweredQuestion(...)  returns (Empty);   // fire-and-forget (score 1–5)
rpc UserProvidedMood(...)      returns (Empty);   // fire-and-forget (mood 1–5)
rpc GetRecommendations(...)    returns (GetRecommendationsResponse);  // ≤500ms
```

### StoryService (Python as client — Python calls C#)

```protobuf
rpc GetStoryCatalogue(...)  returns (GetStoryCatalogueResponse);
rpc SaveUserState(...)      returns (Empty);
rpc LoadUserState(...)      returns (LoadUserStateResponse);
```

Full message definitions are in [`proto/recommender.proto`](proto/recommender.proto).

## Regenerating gRPC Bindings

```bash
make proto
```

This runs `grpc_tools.protoc` and outputs `generated/recommender_pb2.py` and `generated/recommender_pb2_grpc.py`. These files are not committed to the repository.

## Dependencies

- `grpcio` / `grpcio-tools` — gRPC transport and code generation
- `protobuf` — Protocol Buffers runtime
- `numpy` — cosine similarity computations (no heavy ML framework needed at this scale)
- `pytest` / `pytest-cov` — testing

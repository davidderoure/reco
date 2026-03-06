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

**Python also acts as a gRPC client** — it calls back to C# to fetch the story catalogue and persist/load user state.

### Recommendation slots (6 total)

| # | Strategy | Algorithm |
|---|---|---|
| 2 | Content-based | Cosine similarity: user theme+tag weight vector vs story indicator vectors |
| 2 | Collaborative filtering | User-user cosine similarity; aggregate stories from top-20 similar users |
| 1 | Topical | User's highest-weight tag → best matching unviewed stories |
| 1 | Wildcard | Random, preferring stories in unexplored themes |

#### Slot stability

Stories from the previous recommendation set that the user has not yet viewed or completed stay in the same list position on the next request. Only slots freed by user actions are replaced with fresh picks. This means the UI stays consistent between refreshes — stories don't jump around unless the user has acted on them.

#### Progressive coverage

When filling an open slot the engine first looks for stories it has never recommended to that user. This guarantees that a user who consistently acts on recommendations will eventually be offered every story in the catalogue. Previously-recommended stories are only repeated once all novel candidates are exhausted.

### User preference model

Every interaction updates a per-user theme/tag weight vector:

| Event | Weight delta per theme/tag |
|---|---|
| Viewed | +1.0 |
| Completed | +2.0 (additive with viewed) |
| Scored 1–5 | `(score − 3) × 0.5` |
| Read ≥ 50% | +1.0 (same as viewed; applied once, idempotent) |
| Read < 50% | no weight impact; progress recorded only |
| Mood 1–5 | stored only, no weight impact |

The read-progress threshold (50%) is defined by `READ_VIEWED_THRESHOLD_PERCENT` in `user_state.py`.

### State persistence

User state is persisted to the C# server as a compact **user model** — the already-computed derived state — rather than a raw event log. This keeps the payload size bounded by the size of the story catalogue vocabulary regardless of how many interactions a user has had, making startup and periodic saves scalable to large numbers of users.

The persisted model includes: theme/tag weights, viewed/completed/scored story sets, recent mood scores, the last recommendation set (for slot stability), and the set of all stories ever recommended (for progressive coverage).

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
main.py                   Entry point — real recommender service
mock_server.py            Mock C# backend + browser test UI (see below)
config.py                 Environment-variable configuration
tests/                    pytest test suite (179 tests)
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

### With the mock server (no C# required)

`mock_server.py` stands in for the C# backend and provides a browser UI for
interactive testing. Start both processes in separate terminals:

**Terminal 1 — mock backend + web UI:**
```bash
python mock_server.py
```

**Terminal 2 — recommender:**
```bash
python main.py
```

Then open **http://localhost:8080** in a browser.

| Port | Process | Role |
|---|---|---|
| `50052` | `mock_server.py` | StoryService gRPC (acts as C# backend) |
| `50051` | `main.py` | RecommenderService gRPC |
| `8080` | `mock_server.py` | Browser test UI |

#### Browser UI panels

| Panel | Description |
|---|---|
| **Story Catalogue** | 27 sample Oxford museum stories — click View, Complete, Score, Mood, or Read% to fire events |
| **Recommendations** | 6 recommended stories returned by the engine after clicking "Get Recommendations" |
| **Preference Weights** | Live bar chart of theme/tag weights, updated after every interaction |
| **Event Log** | Timestamped record of every event fired for the selected user |

Use the user selector (alice, bob, charlie, diana, test_user, or a custom ID) to
switch between users and observe how different interaction histories produce
different recommendations, and how the collaborative strategy responds once
multiple users have built up histories.

If you get `OSError: [Errno 48] Address already in use`, a previous process is
still holding a port. Free it with:
```bash
kill $(lsof -ti :8080 -ti :50052) 2>/dev/null
```

### With a real C# server

```bash
# Start the recommender (points at your C# server)
CSHARP_SERVER_ADDRESS=<host>:50052 python main.py
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

All requests carry `user_id` (string) and `timestamp` (UTC). There is no concept of a user session; state is saved periodically by a background thread.

```protobuf
rpc UserViewedStory(...)       returns (Empty);   // fire-and-forget
rpc UserCompletedStory(...)    returns (Empty);   // fire-and-forget
rpc UserAnsweredQuestion(...)  returns (Empty);   // fire-and-forget (score 1–5)
rpc UserProvidedMood(...)      returns (Empty);   // fire-and-forget (mood_score 1–5)
rpc UserReadStory(...)         returns (Empty);   // fire-and-forget (read_percent 0–100)
rpc GetRecommendations(...)    returns (GetRecommendationsResponse);  // ≤500ms
```

### StoryService (Python as client — Python calls C#)

```protobuf
rpc GetStoryCatalogue(...)  returns (GetStoryCatalogueResponse);
rpc SaveUserModel(...)      returns (Empty);
rpc LoadUserModel(...)      returns (LoadUserModelResponse);
```

`StoryMessage` carries `story_id`, `title`, `themes` (exactly one per story), `tags` (free-text), and `authors` (display names of the story's author(s)).

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

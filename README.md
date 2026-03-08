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

All 6 slots are recalculated on every call — the result is never returned from cache. If a story from the previous set reappears in the fresh calculation (and the user has not acted on it), it is placed at its original slot index. This keeps the display consistent when the catalogue is nearly exhausted and some stories must be repeated, while still allowing consecutive calls without any interaction to return different stories ("try again").

#### Progressive coverage

When filling an open slot the engine first looks for stories it has never recommended to that user. This guarantees that a user who consistently acts on recommendations will eventually be offered every story in the catalogue. Previously-recommended stories are only repeated once all novel candidates are exhausted.

#### Mood-responsive allocation

The default slot counts above shift based on the user's recent average mood (last 5 entries):

| Recent mood avg | Content | Collaborative | Topical | Wildcard | Rationale |
|---|---|---|---|---|---|
| ≤ 2.5 (low) | 3 | 2 | 1 | 0 | Familiar comfort content; no surprises |
| 2.5–3.5 (neutral / no data) | 2 | 2 | 1 | 1 | Default |
| ≥ 3.5 (high) | 1 | 2 | 1 | 2 | Broader exploration |

Collaborative and Topical slots are held constant because collaborative recommendations surface what similar-mood users enjoy, and the topical slot anchors a known deep interest regardless of current mood state.

### User preference model

Every interaction updates a per-user theme/tag weight vector:

| Event | Weight delta per theme/tag |
|---|---|
| Viewed | +1.0 |
| Completed | +2.0 (additive with viewed) |
| Scored 1–5 | `(score − 3) × 0.5` |
| Read ≥ 50% | +1.0 (same as viewed; applied once, idempotent) |
| Read < 50% | no weight impact; progress recorded only |
| Mood 1–5 | attribution feedback: `±(mood_delta / 4) × 0.5` applied to themes/tags engaged since the previous mood event — improvement boosts them, decline dampens (floor 0) |
| Bookmarked | recorded as an analytic event only; no current weight impact |

The read-progress threshold (50%) is defined by `READ_VIEWED_THRESHOLD_PERCENT` in `user_state.py`.

Mood attribution factor (`MOOD_ATTRIBUTION_FACTOR = 0.5`) means a maximum mood swing of +4 (e.g. 1 → 5) adds at most half a "viewed" event of extra weight per theme/tag. The transient accumulator that tracks engaged content between mood events is not persisted; it resets on service restart.

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
load_users.py             Synthetic load generator — 100 test users (see below)
config.py                 Environment-variable configuration
tests/                    pytest test suite (202 tests)
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
| **Story Catalogue** | 27 sample Oxford museum stories — click View, Complete, Score, Mood, Read%, or Bookmark to fire events |
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

### Synthetic load generator (`load_users.py`)

`load_users.py` populates the recommender with realistic interaction histories
for up to 100 synthetic users (`load_user_001` … `load_user_N`), making the
collaborative-filtering strategy meaningful right away.

Each user is assigned a randomly-generated but **consistent** preference profile:
theme weights drawn from a right-skewed Beta distribution (so each user has a
handful of strong interests and mostly ignores the rest).  An overall engagement
factor controls how many stories the user interacts with.

| Event | Probability / value |
|---|---|
| Viewed | `engagement × (0.05 + 0.95 × theme_pref)` |
| Read% | 70 % of views; amount ∝ preference (triangular distribution) |
| Completed | `pref² × engagement × 0.8` — only strongly preferred stories |
| Scored 1–5 | 80 % of completions; score ~ N(1 + pref × 4, 0.8) |
| Mood 1–5 | 1 per reading session (3–7 stories); score correlates with session's avg preference, so the recommender's mood-attribution mechanism sees realistic signal |

Stories are sorted chronologically and grouped into sessions before sending, so each mood event arrives at the recommender *after* the story events it should be attributed to. Interactions are spread across a 60-day window so the timestamps are realistic.

**Run with mock server already started (Terminal 1 + 2 above):**
```bash
# Terminal 3 — generate events for 100 users
python load_users.py

# Fewer users for a quick test
python load_users.py --users 20

# Fully reproducible run (same seed → same interaction histories)
python load_users.py --seed 42

# Also fetch recommendations for each user at the end
python load_users.py --recs

# Remote recommender
python load_users.py --addr host:50051
```

After running, switch between `load_user_001` … `load_user_100` in the browser
UI's user selector to inspect individual Preference Weight profiles and see how
the collaborative-filtering strategy groups similar users.

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
rpc UserBookmarkedStory(...)   returns (Empty);   // fire-and-forget (analytic only, no rec effect)
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

"""
mock_server.py — Self-contained mock C# backend + browser test UI.

Ports
-----
50052  gRPC  StoryService  (this server acts as the C# backend)
 8080  HTTP  Web UI + JSON API (browser-facing)

The HTTP handlers call RecommenderService on localhost:50051 as a gRPC client.

Startup order
-------------
1. python mock_server.py   — StoryService gRPC on 50052, web UI on 8080
2. python main.py          — recommender connects to 50052, serves on 50051
3. Open http://localhost:8080

No extra dependencies — uses only what is already in requirements.txt
(grpcio, protobuf) plus Python stdlib.
"""

from __future__ import annotations

import json
import logging
import socketserver
import threading
import time
from concurrent import futures
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import grpc
from google.protobuf import empty_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from generated import recommender_pb2, recommender_pb2_grpc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOCK_GRPC_PORT: int = 50052
HTTP_PORT: int = 8080
RECOMMENDER_GRPC_ADDR: str = "localhost:50051"
GRPC_MAX_WORKERS: int = 10

logger = logging.getLogger("mock_server")

# ---------------------------------------------------------------------------
# Static story catalogue — 15 stories, 6 themes, 18 tags
# ---------------------------------------------------------------------------

SAMPLE_STORIES: list[dict[str, Any]] = [
    # Adventure (3)
    {
        "story_id": "adv_001",
        "title": "The Kraken's Bargain",
        "themes": ["adventure"],
        "tags": ["pirates", "ocean", "treasure"],
    },
    {
        "story_id": "adv_002",
        "title": "Summit of No Return",
        "themes": ["adventure"],
        "tags": ["mountains", "survival", "wilderness"],
    },
    {
        "story_id": "adv_003",
        "title": "River of Forgotten Kings",
        "themes": ["adventure", "history"],
        "tags": ["jungle", "archaeology", "ancient"],
    },
    # Mystery (3)
    {
        "story_id": "mys_001",
        "title": "The Clockmaker's Cipher",
        "themes": ["mystery"],
        "tags": ["detective", "clues", "city"],
    },
    {
        "story_id": "mys_002",
        "title": "Whispers in the Attic",
        "themes": ["mystery"],
        "tags": ["haunted", "clues", "family"],
    },
    {
        "story_id": "mys_003",
        "title": "The Missing Cartographer",
        "themes": ["mystery", "adventure"],
        "tags": ["maps", "clues", "wilderness"],
    },
    # Horror (2)
    {
        "story_id": "hor_001",
        "title": "Below the Waterline",
        "themes": ["horror"],
        "tags": ["ocean", "monsters", "darkness"],
    },
    {
        "story_id": "hor_002",
        "title": "Children of the Long Night",
        "themes": ["horror"],
        "tags": ["darkness", "supernatural", "isolation"],
    },
    # Calm (2)
    {
        "story_id": "clm_001",
        "title": "The Lantern Keeper",
        "themes": ["calm"],
        "tags": ["lighthouse", "ocean", "peace"],
    },
    {
        "story_id": "clm_002",
        "title": "Meadow Before the Rain",
        "themes": ["calm"],
        "tags": ["nature", "flowers", "peace"],
    },
    # Romance (2)
    {
        "story_id": "rom_001",
        "title": "Letters Across the Channel",
        "themes": ["romance"],
        "tags": ["wartime", "letters", "longing"],
    },
    {
        "story_id": "rom_002",
        "title": "A Dance at Midsummer",
        "themes": ["romance", "calm"],
        "tags": ["festival", "music", "longing"],
    },
    # History (3)
    {
        "story_id": "his_001",
        "title": "The Spy of Alexandria",
        "themes": ["history", "mystery"],
        "tags": ["ancient", "espionage", "city"],
    },
    {
        "story_id": "his_002",
        "title": "Iron Roads West",
        "themes": ["history", "adventure"],
        "tags": ["frontier", "trains", "survival"],
    },
    {
        "story_id": "his_003",
        "title": "Last Day of the Shogunate",
        "themes": ["history"],
        "tags": ["samurai", "ancient", "conflict"],
    },
]

_STORY_BY_ID: dict[str, dict[str, Any]] = {s["story_id"]: s for s in SAMPLE_STORIES}

_DEFAULT_USERS: list[str] = ["alice", "bob", "charlie", "diana", "test_user"]

# ---------------------------------------------------------------------------
# In-memory user state store
# ---------------------------------------------------------------------------


class InMemoryUserStateStore:
    """Thread-safe in-memory store for user event logs.

    Mirrors the UserStateMessage proto structure. The recommender calls
    SaveUserState periodically to persist its in-memory state here, and
    LoadUserState at startup to replay events.

    Event dict schema::

        {
            "event_type": str,          # "viewed"|"completed"|"scored"|"mood"
            "story_id":   str,          # "" for mood events
            "score":      int,          # 0 when not applicable
            "timestamp_seconds": int,
        }
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: dict[str, list[dict[str, Any]]] = {}

    def record_event(
        self,
        user_id: str,
        event_type: str,
        story_id: str = "",
        score: int = 0,
    ) -> None:
        """Append one event for *user_id*. Called by the HTTP /api/event handler."""
        entry = {
            "event_type": event_type,
            "story_id": story_id,
            "score": score,
            "timestamp_seconds": int(datetime.now(timezone.utc).timestamp()),
        }
        with self._lock:
            self._events.setdefault(user_id, []).append(entry)

    def get_events(self, user_id: str) -> list[dict[str, Any]]:
        """Return a snapshot of events for *user_id*."""
        with self._lock:
            return list(self._events.get(user_id, []))

    def get_all_events(self) -> dict[str, list[dict[str, Any]]]:
        """Return a full snapshot of all user events."""
        with self._lock:
            return {uid: list(evts) for uid, evts in self._events.items()}

    def replace_from_proto_states(self, user_states: Any) -> None:
        """Overwrite internal state from a proto SaveUserStateRequest batch.

        The recommender is the source of truth for processed state; when it
        calls SaveUserState we accept its version wholesale.
        """
        with self._lock:
            for state in user_states:
                events = []
                for e in state.events:
                    events.append(
                        {
                            "event_type": e.event_type,
                            "story_id": e.story_id,
                            "score": e.score,
                            "timestamp_seconds": e.timestamp.seconds,
                        }
                    )
                self._events[state.user_id] = events


# Module-level singleton shared between gRPC servicer and HTTP handlers
_user_state_store = InMemoryUserStateStore()

# ---------------------------------------------------------------------------
# gRPC StoryService implementation
# ---------------------------------------------------------------------------


def _make_timestamp(seconds: int) -> Timestamp:
    ts = Timestamp()
    ts.seconds = seconds
    ts.nanos = 0
    return ts


class MockStoryServiceServicer(recommender_pb2_grpc.StoryServiceServicer):
    """Implements StoryService — the role that the real C# server plays.

    * GetStoryCatalogue — returns SAMPLE_STORIES on every call.
    * SaveUserState     — accepts and stores the recommender's event logs.
    * LoadUserState     — returns stored event logs to the recommender at startup.
    """

    def GetStoryCatalogue(
        self,
        request: recommender_pb2.GetStoryCatalogueRequest,
        context: grpc.ServicerContext,
    ) -> recommender_pb2.GetStoryCatalogueResponse:
        """Return the full static catalogue."""
        stories = [
            recommender_pb2.StoryMessage(
                story_id=s["story_id"],
                title=s["title"],
                themes=s["themes"],
                tags=s["tags"],
            )
            for s in SAMPLE_STORIES
        ]
        logger.info("GetStoryCatalogue → %d stories", len(stories))
        return recommender_pb2.GetStoryCatalogueResponse(stories=stories)

    def SaveUserState(
        self,
        request: recommender_pb2.SaveUserStateRequest,
        context: grpc.ServicerContext,
    ) -> empty_pb2.Empty:
        """Accept the recommender's full event log batch."""
        _user_state_store.replace_from_proto_states(request.user_states)
        logger.info(
            "SaveUserState: received state for %d users",
            len(request.user_states),
        )
        return empty_pb2.Empty()

    def LoadUserState(
        self,
        request: recommender_pb2.LoadUserStateRequest,
        context: grpc.ServicerContext,
    ) -> recommender_pb2.LoadUserStateResponse:
        """Return stored event logs to the recommender.

        An empty ``user_ids`` list means "return all users".
        """
        all_events = _user_state_store.get_all_events()
        wanted = set(request.user_ids) if request.user_ids else set(all_events.keys())

        user_states = []
        for uid in wanted:
            event_msgs = [
                recommender_pb2.UserEventMessage(
                    event_type=e["event_type"],
                    story_id=e["story_id"],
                    score=e["score"],
                    timestamp=_make_timestamp(e["timestamp_seconds"]),
                )
                for e in all_events.get(uid, [])
            ]
            user_states.append(
                recommender_pb2.UserStateMessage(user_id=uid, events=event_msgs)
            )

        logger.info("LoadUserState → %d users", len(user_states))
        return recommender_pb2.LoadUserStateResponse(user_states=user_states)


# ---------------------------------------------------------------------------
# gRPC server
# ---------------------------------------------------------------------------


def _build_grpc_server() -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=GRPC_MAX_WORKERS))
    recommender_pb2_grpc.add_StoryServiceServicer_to_server(
        MockStoryServiceServicer(), server
    )
    server.add_insecure_port(f"0.0.0.0:{MOCK_GRPC_PORT}")
    return server


def _run_grpc_server(server: grpc.Server) -> None:
    server.start()
    logger.info("StoryService gRPC server listening on port %d", MOCK_GRPC_PORT)
    server.wait_for_termination()


# ---------------------------------------------------------------------------
# Recommender gRPC client helpers (called from HTTP handlers)
# ---------------------------------------------------------------------------

# Reuse a single channel for all outbound calls to the recommender.
# Protected by a lock because channels are thread-safe but we want a single
# instance.
_recommender_channel: grpc.Channel | None = None
_channel_lock = threading.Lock()


def _get_recommender_stub() -> recommender_pb2_grpc.RecommenderServiceStub:
    """Return a stub backed by a lazily-created, reused gRPC channel."""
    global _recommender_channel
    with _channel_lock:
        if _recommender_channel is None:
            _recommender_channel = grpc.insecure_channel(RECOMMENDER_GRPC_ADDR)
    return recommender_pb2_grpc.RecommenderServiceStub(_recommender_channel)


def _now_ts() -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(datetime.now(timezone.utc))
    return ts


def grpc_get_recommendations(user_id: str) -> tuple[list[str], str | None]:
    """Call GetRecommendations → (story_ids, error_message | None)."""
    try:
        resp = _get_recommender_stub().GetRecommendations(
            recommender_pb2.GetRecommendationsRequest(user_id=user_id),
            timeout=3.0,
        )
        return list(resp.story_ids), None
    except grpc.RpcError as exc:
        msg = f"gRPC error: {exc.code()}: {exc.details()}"
        logger.error("GetRecommendations failed: %s", msg)
        return [], msg


def grpc_send_event(
    event_type: str, user_id: str, story_id: str = "", score: int = 0
) -> str | None:
    """Fire one event at the recommender. Returns error message or None."""
    ts = _now_ts()
    try:
        stub = _get_recommender_stub()
        if event_type == "viewed":
            stub.UserViewedStory(
                recommender_pb2.UserViewedStoryRequest(
                    user_id=user_id, story_id=story_id, timestamp=ts
                ),
                timeout=3.0,
            )
        elif event_type == "completed":
            stub.UserCompletedStory(
                recommender_pb2.UserCompletedStoryRequest(
                    user_id=user_id, story_id=story_id, timestamp=ts
                ),
                timeout=3.0,
            )
        elif event_type == "scored":
            stub.UserAnsweredQuestion(
                recommender_pb2.UserAnsweredQuestionRequest(
                    user_id=user_id, story_id=story_id, score=score, timestamp=ts
                ),
                timeout=3.0,
            )
        elif event_type == "mood":
            stub.UserProvidedMood(
                recommender_pb2.UserProvidedMoodRequest(
                    user_id=user_id, mood_score=score, timestamp=ts
                ),
                timeout=3.0,
            )
        return None
    except grpc.RpcError as exc:
        msg = f"gRPC error: {exc.code()}: {exc.details()}"
        logger.warning("Event %s failed: %s", event_type, msg)
        return msg


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Each HTTP request is handled in its own thread.

    Required because each handler blocks on a gRPC call to the recommender.
    """

    daemon_threads = True


def _send_json(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(handler: BaseHTTPRequestHandler, msg: str, status: int = 400) -> None:
    _send_json(handler, {"error": msg}, status=status)


class MockServerHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing the JSON API and serving the embedded web UI.

    Routes
    ------
    GET  /                          Serve embedded HTML page
    GET  /api/stories               Full catalogue JSON
    GET  /api/recommendations       Call recommender gRPC, return enriched JSON
    POST /api/event                 Fire one recommender event
    GET  /api/state                 Return user event log
    """

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Suppress per-request stdout noise; errors still reach the logger
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_ui()
        elif path == "/api/stories":
            _send_json(self, SAMPLE_STORIES)
        elif path == "/api/recommendations":
            self._handle_recommendations(params)
        elif path == "/api/state":
            self._handle_state(params)
        else:
            _send_error(self, "Not found", status=404)

    def do_POST(self) -> None:
        if urlparse(self.path).path == "/api/event":
            self._handle_event()
        else:
            _send_error(self, "Not found", status=404)

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    def _serve_ui(self) -> None:
        body = WEB_UI_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_recommendations(self, params: dict) -> None:
        """GET /api/recommendations?user_id=X"""
        uid = (params.get("user_id") or [""])[0].strip()
        if not uid:
            _send_error(self, "user_id query parameter required")
            return
        story_ids, err = grpc_get_recommendations(uid)
        if err:
            _send_error(self, err, status=502)
            return
        enriched = [
            _STORY_BY_ID.get(
                sid,
                {"story_id": sid, "title": sid, "themes": [], "tags": []},
            )
            for sid in story_ids
        ]
        _send_json(self, {"user_id": uid, "recommendations": enriched})

    def _handle_event(self) -> None:
        """POST /api/event  body: {type, user_id, story_id?, score?}"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception as exc:
            _send_error(self, f"Invalid JSON: {exc}")
            return

        event_type = body.get("type", "")
        user_id = str(body.get("user_id", "")).strip()
        story_id = str(body.get("story_id", "")).strip()
        score = int(body.get("score", 0))

        if not user_id:
            _send_error(self, "user_id required")
            return

        # Validate
        if event_type in ("viewed", "completed") and not story_id:
            _send_error(self, f"story_id required for {event_type}")
            return
        if event_type == "scored" and (not story_id or not 1 <= score <= 5):
            _send_error(self, "story_id and score 1-5 required for scored")
            return
        if event_type == "mood" and not 1 <= score <= 5:
            _send_error(self, "score 1-5 required for mood")
            return
        if event_type not in ("viewed", "completed", "scored", "mood"):
            _send_error(self, f"Unknown event type: {event_type!r}")
            return

        # Record locally first (so the state panel shows events immediately,
        # even before the recommender's periodic SaveUserState call)
        _user_state_store.record_event(
            user_id, event_type, story_id=story_id, score=score
        )

        # Forward to recommender
        err = grpc_send_event(event_type, user_id, story_id=story_id, score=score)
        if err:
            _send_json(self, {"ok": False, "warning": err})
        else:
            _send_json(self, {"ok": True})

    def _handle_state(self, params: dict) -> None:
        """GET /api/state?user_id=X"""
        uid = (params.get("user_id") or [""])[0].strip()
        if not uid:
            _send_error(self, "user_id query parameter required")
            return
        events = _user_state_store.get_events(uid)
        _send_json(self, {"user_id": uid, "events": events})


# ---------------------------------------------------------------------------
# Embedded web UI
# ---------------------------------------------------------------------------

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recommender Test UI</title>
<style>
*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 14px;
  background: #f1f5f9;
  color: #1e293b;
}
header {
  background: #1e293b;
  color: #f8fafc;
  padding: 0.75rem 1rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}
header h1 { margin: 0; font-size: 1rem; font-weight: 700; letter-spacing: 0.02em; }
header select, header input[type=text] {
  padding: 0.3rem 0.5rem;
  border-radius: 6px;
  border: none;
  font-size: 0.875rem;
}
.main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  padding: 0.75rem;
}
.full-width { grid-column: 1 / -1; }
.panel {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 1px 3px rgba(0,0,0,.1);
  padding: 0.75rem;
  overflow: auto;
  max-height: 460px;
}
.panel h2 {
  margin: 0 0 0.5rem 0;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #64748b;
}
.story-card {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.5rem 0.65rem;
  margin-bottom: 0.5rem;
  background: #f8fafc;
}
.story-card strong { display: block; margin-bottom: 0.2rem; }
.story-card .story-id { font-size: 0.7rem; color: #94a3b8; margin-bottom: 0.3rem; }
.pills { display: flex; flex-wrap: wrap; gap: 0.25rem; margin: 0.2rem 0; }
.theme-pill {
  font-size: 0.68rem;
  font-weight: 600;
  color: #fff;
  padding: 0.1rem 0.45rem;
  border-radius: 999px;
}
.tag-pill {
  font-size: 0.68rem;
  color: #475569;
  background: #e2e8f0;
  padding: 0.1rem 0.4rem;
  border-radius: 999px;
}
.card-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
  margin-top: 0.4rem;
}
.btn {
  font-size: 0.75rem;
  padding: 0.25rem 0.55rem;
  border: 1px solid #cbd5e1;
  border-radius: 5px;
  background: #fff;
  cursor: pointer;
  transition: background 0.1s;
}
.btn:hover { background: #f1f5f9; }
.btn.primary {
  background: #3b82f6;
  color: #fff;
  border-color: #3b82f6;
}
.btn.primary:hover { background: #2563eb; }
.btn.danger { border-color: #fca5a5; color: #dc2626; }
.btn.danger:hover { background: #fef2f2; }
.slot-label {
  font-size: 0.68rem;
  color: #94a3b8;
  margin-bottom: 0.15rem;
}
.status-bar {
  font-size: 0.8rem;
  padding: 0.2rem 0.5rem;
  border-radius: 5px;
  min-width: 180px;
}
.status-ok  { color: #16a34a; }
.status-err { color: #dc2626; }
.weight-bar-wrap { margin-bottom: 0.35rem; }
.weight-label {
  display: flex; justify-content: space-between;
  font-size: 0.75rem; color: #475569; margin-bottom: 0.1rem;
}
.weight-bar-bg { background: #e2e8f0; border-radius: 3px; height: 7px; }
.weight-bar-fill { background: #3b82f6; border-radius: 3px; height: 7px; transition: width 0.3s; }
.weight-negative .weight-bar-fill { background: #f87171; }
.weight-section-title {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #94a3b8;
  margin: 0.5rem 0 0.25rem;
}
#event-log-list {
  list-style: none;
  margin: 0;
  padding: 0;
  font-family: monospace;
  font-size: 0.75rem;
}
#event-log-list li {
  padding: 0.2rem 0;
  border-bottom: 1px solid #f1f5f9;
  color: #475569;
}
#event-log-list li:first-child { color: #1e293b; font-weight: 600; }
.empty-state { color: #94a3b8; font-style: italic; font-size: 0.8rem; }
#recommender-status {
  font-size: 0.75rem;
  padding: 0.4rem 0.75rem;
  border-radius: 6px;
  background: #fef9c3;
  color: #854d0e;
  border: 1px solid #fde68a;
}
#recommender-status.ok { background: #dcfce7; color: #166534; border-color: #bbf7d0; }
</style>
</head>
<body>
<header>
  <h1>&#127752; Recommender Test UI</h1>
  <label style="color:#94a3b8;font-size:.8rem">User:
    <select id="user-select">
      <option value="alice">alice</option>
      <option value="bob">bob</option>
      <option value="charlie">charlie</option>
      <option value="diana">diana</option>
      <option value="test_user">test_user</option>
      <option value="__custom__">Custom&hellip;</option>
    </select>
  </label>
  <input id="custom-user" type="text" placeholder="Enter user ID&hellip;"
         style="display:none;width:130px">
  <button class="btn primary" onclick="getRecommendations()">&#10024; Get Recommendations</button>
  <button class="btn" onclick="loadState()">&#8635; Refresh State</button>
  <span id="status-bar" class="status-bar"></span>
  <span id="recommender-status">&#9679; Checking recommender&hellip;</span>
</header>

<div class="main-grid">

  <!-- Left: catalogue -->
  <div class="panel">
    <h2>Story Catalogue (<span id="cat-count">0</span>)</h2>
    <div id="catalogue-list"><p class="empty-state">Loading&hellip;</p></div>
  </div>

  <!-- Right: recommendations -->
  <div class="panel">
    <h2>Recommendations</h2>
    <div id="recs-list"><p class="empty-state">Click &ldquo;Get Recommendations&rdquo; to start.</p></div>
  </div>

  <!-- Full-width: preference weights -->
  <div class="panel full-width">
    <h2>Preference Weights &mdash; <span id="weights-user"></span></h2>
    <div id="weights-content"><p class="empty-state">No interactions yet.</p></div>
  </div>

  <!-- Full-width: event log -->
  <div class="panel full-width">
    <h2>Event Log &mdash; <span id="log-user"></span></h2>
    <ul id="event-log-list"><li class="empty-state">No events recorded yet.</li></ul>
  </div>

</div>

<script>
// ---------------------------------------------------------------------------
// Theme colour map
// ---------------------------------------------------------------------------
const THEME_COLORS = {
  adventure: '#f59e0b',
  mystery:   '#8b5cf6',
  horror:    '#ef4444',
  calm:      '#10b981',
  romance:   '#ec4899',
  history:   '#6b7280',
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let catalogue = [];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function currentUserId() {
  const sel = document.getElementById('user-select');
  if (sel.value === '__custom__') {
    return (document.getElementById('custom-user').value || '').trim();
  }
  return sel.value;
}

function setStatus(msg, isError = false) {
  const el = document.getElementById('status-bar');
  el.textContent = msg;
  el.className = 'status-bar ' + (isError ? 'status-err' : 'status-ok');
}

function themePill(theme) {
  const color = THEME_COLORS[theme] || '#999';
  return `<span class="theme-pill" style="background:${color}">${theme}</span>`;
}

function tagPill(tag) {
  return `<span class="tag-pill">${tag}</span>`;
}

function storyCardHTML(story, showButtons) {
  const themes = (story.themes || []).map(themePill).join('');
  const tags   = (story.tags   || []).map(tagPill).join('');
  const sid    = story.story_id;
  let buttons  = '';
  if (showButtons) {
    buttons = `<div class="card-buttons">
      <button class="btn" onclick="sendEvent('viewed','${sid}')">&#128214; View</button>
      <button class="btn" onclick="sendEvent('completed','${sid}')">&#9989; Complete</button>
      <button class="btn" onclick="promptScore('${sid}')">&#11088; Score</button>
      <button class="btn" onclick="promptMood()">&#128149; Mood</button>
    </div>`;
  }
  return `<div class="story-card">
    <strong>${story.title}</strong>
    <div class="story-id">${sid}</div>
    <div class="pills">${themes}</div>
    <div class="pills">${tags}</div>
    ${buttons}
  </div>`;
}

// ---------------------------------------------------------------------------
// Recommender health check
// ---------------------------------------------------------------------------
async function checkRecommenderHealth() {
  const el = document.getElementById('recommender-status');
  try {
    // A GetRecommendations call for a dummy user is the simplest health probe
    const resp = await fetch('/api/recommendations?user_id=__health__');
    const data = await resp.json();
    if (data.error && data.error.includes('gRPC')) {
      el.textContent = '\\u25CF Recommender unreachable';
      el.className = '';
    } else {
      el.textContent = '\\u25CF Recommender connected';
      el.className = 'ok';
    }
  } catch {
    el.textContent = '\\u25CF Recommender unreachable';
    el.className = '';
  }
}

// ---------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------
async function loadCatalogue() {
  const resp = await fetch('/api/stories');
  catalogue = await resp.json();
  document.getElementById('cat-count').textContent = catalogue.length;
  document.getElementById('catalogue-list').innerHTML =
    catalogue.map(s => storyCardHTML(s, true)).join('');
}

// ---------------------------------------------------------------------------
// Recommendations
// ---------------------------------------------------------------------------
async function getRecommendations() {
  const userId = currentUserId();
  if (!userId) { setStatus('Enter a user ID first.', true); return; }
  setStatus('Fetching recommendations\\u2026');
  try {
    const resp = await fetch('/api/recommendations?user_id=' + encodeURIComponent(userId));
    const data = await resp.json();
    if (data.error) { setStatus(data.error, true); return; }

    const container = document.getElementById('recs-list');
    if (!data.recommendations.length) {
      container.innerHTML = '<p class="empty-state">No recommendations returned (is the recommender running?)</p>';
      setStatus('No recommendations received.', true);
      return;
    }
    container.innerHTML = data.recommendations.map((s, i) =>
      `<div class="slot-label">slot ${i + 1}</div>` + storyCardHTML(s, true)
    ).join('');
    setStatus(`${data.recommendations.length} recommendations loaded.`);
    appendLog(`GetRecommendations → ${data.recommendations.map(s => s.story_id).join(', ')}`);
    await loadState();
  } catch (e) {
    setStatus('Error: ' + e.message, true);
  }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
async function sendEvent(type, storyId, score) {
  const userId = currentUserId();
  if (!userId) { setStatus('Enter a user ID first.', true); return; }

  const body = { type, user_id: userId, story_id: storyId || '', score: score || 0 };
  try {
    const resp = await fetch('/api/event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (data.error) { setStatus(data.error, true); return; }
    if (data.warning) setStatus('Sent (recommender warning: ' + data.warning + ')', true);
    else setStatus('Event sent: ' + type);

    const label = score ? `${type}(${score})` : type;
    const storyLabel = storyId ? ` on ${storyId}` : '';
    appendLog(`${label}${storyLabel}`);
    await loadState();
  } catch (e) {
    setStatus('Error: ' + e.message, true);
  }
}

function promptScore(storyId) {
  const s = prompt('Rate this story (1 = poor \\u2192 5 = excellent):');
  const score = parseInt(s, 10);
  if (score >= 1 && score <= 5) sendEvent('scored', storyId, score);
  else if (s !== null) alert('Please enter a number between 1 and 5.');
}

function promptMood() {
  const s = prompt('How are you feeling right now? (1 = very low \\u2192 5 = great):');
  const score = parseInt(s, 10);
  if (score >= 1 && score <= 5) sendEvent('mood', '', score);
  else if (s !== null) alert('Please enter a number between 1 and 5.');
}

// ---------------------------------------------------------------------------
// State + weights
// ---------------------------------------------------------------------------
async function loadState() {
  const userId = currentUserId();
  if (!userId) return;
  document.getElementById('weights-user').textContent = userId;
  document.getElementById('log-user').textContent = userId;

  try {
    const resp = await fetch('/api/state?user_id=' + encodeURIComponent(userId));
    const data = await resp.json();
    renderEventLogPanel(data.events || []);
    renderWeights(data.events || []);
  } catch (e) {
    console.error('loadState error', e);
  }
}

function renderEventLogPanel(events) {
  const ul = document.getElementById('event-log-list');
  if (!events.length) {
    ul.innerHTML = '<li class="empty-state">No events recorded yet.</li>';
    return;
  }
  ul.innerHTML = events.slice().reverse().map(e => {
    const ts  = new Date(e.timestamp_seconds * 1000).toLocaleTimeString();
    const score = e.score ? ` score=${e.score}` : '';
    const story = e.story_id ? ` \u2192 ${e.story_id}` : '';
    return `<li>[${ts}] <strong>${e.event_type}</strong>${story}${score}</li>`;
  }).join('');
}

function renderWeights(events) {
  // Reconstruct weights client-side using the same rules as UserStateStore:
  //   viewed    +1.0 per theme/tag
  //   completed +2.0 per theme/tag
  //   scored    (score-3)*0.5 per theme/tag
  const storyMap = Object.fromEntries(catalogue.map(s => [s.story_id, s]));
  const themeW = {};
  const tagW   = {};

  for (const e of events) {
    if (!e.story_id) continue;
    const story = storyMap[e.story_id];
    if (!story) continue;

    let delta = 0;
    if (e.event_type === 'viewed')    delta = 1.0;
    if (e.event_type === 'completed') delta = 2.0;
    if (e.event_type === 'scored')    delta = (e.score - 3) * 0.5;
    if (delta === 0) continue;

    for (const t of story.themes) themeW[t] = (themeW[t] || 0) + delta;
    for (const t of story.tags)   tagW[t]   = (tagW[t]   || 0) + delta;
  }

  const container = document.getElementById('weights-content');
  if (!Object.keys(themeW).length && !Object.keys(tagW).length) {
    container.innerHTML = '<p class="empty-state">No interactions yet &mdash; interact with stories to see weights build up.</p>';
    return;
  }

  const maxAbs = val => Math.max(...Object.values(val).map(Math.abs), 0.001);

  function weightBarsHTML(weights, maxVal, useThemeColors) {
    return Object.entries(weights)
      .sort(([, a], [, b]) => b - a)
      .map(([key, val]) => {
        const pct = Math.abs(val) / maxVal * 100;
        const neg = val < 0;
        const color = useThemeColors
          ? (THEME_COLORS[key] || '#3b82f6')
          : '#3b82f6';
        const fill = neg ? '#f87171' : color;
        return `<div class="weight-bar-wrap ${neg ? 'weight-negative' : ''}">
          <div class="weight-label">
            <span>${key}</span><span>${val > 0 ? '+' : ''}${val.toFixed(1)}</span>
          </div>
          <div class="weight-bar-bg">
            <div class="weight-bar-fill" style="width:${pct.toFixed(1)}%;background:${fill}"></div>
          </div>
        </div>`;
      }).join('');
  }

  const themeMax = maxAbs(themeW);
  const tagMax   = maxAbs(tagW);

  container.innerHTML =
    '<div class="weight-section-title">Themes</div>' +
    weightBarsHTML(themeW, themeMax, true) +
    '<div class="weight-section-title">Tags</div>' +
    weightBarsHTML(tagW, tagMax, false);
}

// ---------------------------------------------------------------------------
// Event log (header strip — for quick action feedback)
// ---------------------------------------------------------------------------
function appendLog(msg) {
  // Just update the status bar; full event log is in the panel
  console.log('[event]', msg);
}

// ---------------------------------------------------------------------------
// User selection
// ---------------------------------------------------------------------------
document.getElementById('user-select').addEventListener('change', function () {
  const isCustom = this.value === '__custom__';
  document.getElementById('custom-user').style.display = isCustom ? 'inline' : 'none';
  if (!isCustom) loadState();
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', async () => {
  await loadCatalogue();
  await loadState();
  await checkRecommenderHealth();
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the StoryService gRPC server and the HTTP web UI server.

    Startup order:
    1. gRPC StoryService on port 50052 (daemon thread).
    2. HTTP web UI on port 8080 (main thread — blocks until Ctrl-C).

    Then start ``python main.py`` in a separate terminal, followed by
    opening http://localhost:8080 in a browser.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # gRPC server — daemon thread
    grpc_server = _build_grpc_server()
    grpc_thread = threading.Thread(
        target=_run_grpc_server,
        args=(grpc_server,),
        name="grpc-storyservice",
        daemon=True,
    )
    grpc_thread.start()
    time.sleep(0.3)  # let the gRPC port bind before main.py tries to connect

    # HTTP server — main thread
    http_server = _ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), MockServerHTTPHandler)
    logger.info("Web UI available at  http://localhost:%d", HTTP_PORT)
    logger.info(
        "Expecting recommender at %s  (start with: python main.py)",
        RECOMMENDER_GRPC_ADDR,
    )
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        grpc_server.stop(grace=2)


if __name__ == "__main__":
    main()

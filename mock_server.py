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
# Static story catalogue — 27 stories, 10 themes
# ---------------------------------------------------------------------------

SAMPLE_STORIES: list[dict[str, Any]] = [
    # ── craft (3) ──────────────────────────────────────────────────────────
    # Stories about making, skill, and the hands that shaped objects.
    {
        "story_id": "cft_001",
        "title": "The Goldsmith's Commission",
        "authors": ["Eleanor Vane"],
        "themes": ["craft"],
        "tags": [
            "anglo-saxon", "goldsmith", "alfred-the-great", "9th-century",
            "enamel", "royalty", "ashmolean",
        ],
    },
    {
        "story_id": "cft_002",
        "title": "Five Thousand Years of Blue",
        "authors": ["Tariq Nassar"],
        "themes": ["craft"],
        "tags": [
            "mesopotamia", "cylinder-seal", "lapis-lazuli", "ur",
            "3rd-millennium-bce", "artisan", "ashmolean",
        ],
    },
    {
        "story_id": "cft_003",
        "title": "The Girl Who Counted Stitches",
        "authors": ["Jane Morrow"],
        "themes": ["craft"],
        "tags": [
            "needlework", "childhood", "georgian-england", "sampler",
            "literacy", "domestic-arts", "18th-century",
        ],
    },
    # ── discovery (3) ──────────────────────────────────────────────────────
    # Stories about finding, knowing, and the cost of curiosity.
    {
        "story_id": "dsc_001",
        "title": "The Last Bird",
        "authors": ["Robert Sefton"],
        "themes": ["discovery"],
        "tags": [
            "dodo", "extinction", "mauritius", "17th-century",
            "natural-history", "sailors", "oxford-natural-history-museum",
        ],
    },
    {
        "story_id": "dsc_002",
        "title": "What the Ichthyosaur Told Mary Anning",
        "authors": ["Celia Trench", "Adaeze Nwachukwu"],
        "themes": ["discovery"],
        "tags": [
            "palaeontology", "dorset", "fossils", "working-class",
            "women-in-science", "victorian", "oxford-natural-history-museum",
        ],
    },
    {
        "story_id": "dsc_003",
        "title": "The Curiosity Collector",
        "authors": ["Sarah Blount"],
        "themes": ["discovery"],
        "tags": [
            "tradescant", "virginia", "collecting", "17th-century",
            "algonquian", "natural-history", "ashmolean",
        ],
    },
    # ── belief (3) ─────────────────────────────────────────────────────────
    # Stories about faith, ritual, and what objects carry across death.
    {
        "story_id": "blf_001",
        "title": "To Hold an Ancestor",
        "authors": ["Mere Tane", "Catherine Fox"],
        "themes": ["belief"],
        "tags": [
            "maori", "new-zealand", "greenstone", "pounamu",
            "ancestor", "heirloom", "pitt-rivers",
        ],
    },
    {
        "story_id": "blf_002",
        "title": "The Shabti Speaks",
        "authors": ["Amira El-Said"],
        "themes": ["belief"],
        "tags": [
            "ancient-egypt", "shabti", "afterlife", "new-kingdom",
            "craftsman", "thebes", "ashmolean",
        ],
    },
    {
        "story_id": "blf_003",
        "title": "Amulet Road",
        "authors": ["Leila Farrokhzad", "Magnus Lindqvist"],
        "themes": ["belief"],
        "tags": [
            "silk-road", "amulets", "central-asia", "islamic-art",
            "buddhism", "apotropaic", "ashmolean",
        ],
    },
    # ── loss (3) ───────────────────────────────────────────────────────────
    # Stories about grief, extinction, and things that cannot be recovered.
    {
        "story_id": "los_001",
        "title": "The Cloak That Crossed an Ocean",
        "authors": ["Linda Sturgeon", "James Running Bear"],
        "themes": ["loss"],
        "tags": [
            "algonquian", "powhatan", "virginia", "colonialism",
            "17th-century", "shell-beads", "ashmolean",
        ],
    },
    {
        "story_id": "los_002",
        "title": "A Face in Wax",
        "authors": ["Nadia Khalil"],
        "themes": ["loss"],
        "tags": [
            "roman-egypt", "fayum", "mummy-portrait", "encaustic",
            "grief", "2nd-century", "ashmolean",
        ],
    },
    {
        "story_id": "los_003",
        "title": "Letters Never Sent",
        "authors": ["Margaret Forsyth"],
        "themes": ["loss"],
        "tags": [
            "world-war-one", "trench-art", "france", "mourning",
            "working-class-soldiers", "personal-objects", "pitt-rivers",
        ],
    },
    # ── conflict (3) ───────────────────────────────────────────────────────
    # Stories about violence, resistance, and survival.
    {
        "story_id": "cnf_001",
        "title": "The Siege of Kerma",
        "authors": ["Amara Diallo"],
        "themes": ["conflict"],
        "tags": [
            "nubia", "ancient-egypt", "kerma", "warfare",
            "nile-valley", "military-technology", "ashmolean",
        ],
    },
    {
        "story_id": "cnf_002",
        "title": "Headhunting Shields and the Sarawak Raj",
        "authors": ["Ling Mei Tan", "Francis Abbot"],
        "themes": ["conflict"],
        "tags": [
            "borneo", "iban", "colonial-violence", "shield",
            "southeast-asia", "resistance", "pitt-rivers",
        ],
    },
    {
        "story_id": "cnf_003",
        "title": "The Partisan's Rifle",
        "authors": ["Milena Kovac"],
        "themes": ["conflict"],
        "tags": [
            "world-war-two", "balkans", "resistance", "firearms",
            "women-combatants", "occupation", "pitt-rivers",
        ],
    },
    # ── power (3) ──────────────────────────────────────────────────────────
    # Stories about authority, those who held it, and those who bore its weight.
    {
        "story_id": "pwr_001",
        "title": "The King's Memory",
        "authors": ["Chukwuemeka Obi"],
        "themes": ["power"],
        "tags": [
            "benin-kingdom", "west-africa", "brass-casting", "oba",
            "ancestor-veneration", "colonial-looting", "ashmolean",
        ],
    },
    {
        "story_id": "pwr_002",
        "title": "The Charter and the Fist",
        "authors": ["Tom Brannigan"],
        "themes": ["power"],
        "tags": [
            "chartism", "victorian", "reform", "working-class",
            "political-protest", "democracy", "ashmolean",
        ],
    },
    {
        "story_id": "pwr_003",
        "title": "The Weight of Hammurabi's Word",
        "authors": ["Layla Hassan"],
        "themes": ["power"],
        "tags": [
            "babylonian", "cuneiform", "hammurabi", "law",
            "scribe", "ancient-iraq", "ashmolean",
        ],
    },
    # ── science (3) ────────────────────────────────────────────────────────
    # Stories about observation, measurement, and the thrill of understanding.
    {
        "story_id": "sci_001",
        "title": "Reading the Stars at Córdoba",
        "authors": ["Fatima Al-Rashid"],
        "themes": ["science"],
        "tags": [
            "astrolabe", "islamic-golden-age", "al-andalus", "medieval",
            "women-in-science", "astronomy", "history-of-science-museum",
        ],
    },
    {
        "story_id": "sci_002",
        "title": "Smallpox, a Lancet, and Lady Mary",
        "authors": ["Constance Drew", "Babatunde Olatunji"],
        "themes": ["science"],
        "tags": [
            "vaccination", "ottoman-empire", "18th-century", "inoculation",
            "public-health", "women-and-medicine", "history-of-science-museum",
        ],
    },
    {
        "story_id": "sci_003",
        "title": "Dead Men's Bones",
        "authors": ["David Holt"],
        "themes": ["science"],
        "tags": [
            "william-buckland", "palaeontology", "yorkshire", "hyena",
            "19th-century", "geology", "oxford-natural-history-museum",
        ],
    },
    # ── trade (2) ──────────────────────────────────────────────────────────
    # Stories about exchange, the objects that travelled the world, and at what price.
    {
        "story_id": "trd_001",
        "title": "The Long Way Round",
        "authors": ["Li Wei", "Anna de Vries"],
        "themes": ["trade"],
        "tags": [
            "chinese-porcelain", "dutch-east-india-company", "jingdezhen",
            "17th-century", "global-trade", "canton", "ashmolean",
        ],
    },
    {
        "story_id": "trd_002",
        "title": "The Cowrie and the Crown",
        "authors": ["Adwoa Mensah", "Jean-Pierre Moreau"],
        "themes": ["trade"],
        "tags": [
            "cowrie-shells", "west-africa", "transatlantic-slave-trade",
            "currency", "maldives", "18th-century", "pitt-rivers",
        ],
    },
    # ── migration (3) ──────────────────────────────────────────────────────
    # Stories about leaving, arriving, and what people carried with them.
    {
        "story_id": "mgr_001",
        "title": "The Hand-Axe and the First Crossing",
        "authors": ["Nadia Petrova", "Kwame Boateng"],
        "themes": ["migration"],
        "tags": [
            "palaeolithic", "hand-axe", "human-dispersal", "africa",
            "prehistoric", "lithic-technology", "ashmolean",
        ],
    },
    {
        "story_id": "mgr_002",
        "title": "The Weaver's Tongue",
        "authors": ["Pierre Lemaire"],
        "themes": ["migration"],
        "tags": [
            "huguenots", "religious-persecution", "silk-weaving",
            "spitalfields", "refugee", "17th-century", "ashmolean",
        ],
    },
    {
        "story_id": "mgr_003",
        "title": "Partition Cloth",
        "authors": ["Sunita Kaur", "Peter Whitfield"],
        "themes": ["migration"],
        "tags": [
            "india-pakistan-partition", "phulkari", "punjab", "displacement",
            "textile", "1947", "pitt-rivers",
        ],
    },
    # ── kinship (3) ────────────────────────────────────────────────────────
    # Stories about love, family, and the bonds that objects carry.
    {
        "story_id": "kin_001",
        "title": "Every Bead a Message",
        "authors": ["Nomsa Dube"],
        "themes": ["kinship"],
        "tags": [
            "zulu", "beadwork", "courtship", "coded-language",
            "south-africa", "19th-century", "pitt-rivers",
        ],
    },
    {
        "story_id": "kin_002",
        "title": "A Ring in the Mud",
        "authors": ["Simon Harker"],
        "themes": ["kinship"],
        "tags": [
            "roman", "ring", "thames", "oxford",
            "soldier", "2nd-century", "ashmolean",
        ],
    },
    {
        "story_id": "kin_003",
        "title": "The Hopi Kachina and the Absent Father",
        "authors": ["Delores Honanie", "Marcus Webb"],
        "themes": ["kinship"],
        "tags": [
            "hopi", "kachina-doll", "southwest-usa", "fatherhood",
            "indigenous-spirituality", "ceremony", "pitt-rivers",
        ],
    },
]

_STORY_BY_ID: dict[str, dict[str, Any]] = {s["story_id"]: s for s in SAMPLE_STORIES}

_DEFAULT_USERS: list[str] = ["alice", "bob", "charlie", "diana", "test_user"]

# ---------------------------------------------------------------------------
# In-memory user state store
# ---------------------------------------------------------------------------


class InMemoryUserStateStore:
    """Thread-safe in-memory store for user state.

    Maintains two separate data sets:

    * **Local event log** (``_events``): raw events recorded by the HTTP
      handler for the web UI's event log panel. Not used by the recommender.
    * **User models** (``_models``): compiled user models saved by the
      recommender via ``SaveUserModel`` and returned via ``LoadUserModel``.

    Event dict schema::

        {
            "event_type": str,          # "scored"|"mood"|"read_progress"|"bookmark"
            "story_id":   str,          # "" for mood events
            "score":      int,          # 0 when not applicable
            "timestamp_seconds": int,
        }
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._models: dict[str, Any] = {}  # user_id → UserModelMessage proto

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

    def get_model_weights(self, user_id: str) -> dict[str, Any] | None:
        """Return serialised weights from the latest saved model, or None.

        The recommender persists compiled models via SaveUserModel on a
        periodic basis (default: every 60 s).  These weights reflect every
        event the recommender has processed, including events sent directly
        over gRPC (e.g. from load_users.py) that never touched the HTTP
        event log.
        """
        with self._lock:
            model = self._models.get(user_id)
            if model is None:
                return None
            return {
                "theme_weights": dict(model.theme_weights),
                "tag_weights": dict(model.tag_weights),
                "viewed_count": len(model.viewed_story_ids),
                "completed_count": len(model.completed_story_ids),
            }

    def all_user_ids(self) -> list[str]:
        """Return a sorted list of all known user IDs.

        Includes users whose events were recorded via the HTTP API and users
        whose models were received via SaveUserModel (e.g. load_users.py).
        """
        with self._lock:
            return sorted(self._events.keys() | self._models.keys())

    def save_models(self, user_models: Any) -> None:
        """Store compiled user models received from the recommender's SaveUserModel call."""
        with self._lock:
            for model in user_models:
                self._models[model.user_id] = model

    def load_models(self, user_ids: list[str]) -> list[Any]:
        """Return stored user models. If *user_ids* is empty, return all."""
        with self._lock:
            wanted = set(user_ids) if user_ids else set(self._models.keys())
            return [self._models[uid] for uid in wanted if uid in self._models]


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
    * SaveUserModel     — accepts and stores the recommender's compiled user models.
    * LoadUserModel     — returns stored models to the recommender at startup.
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
                authors=s.get("authors", []),
            )
            for s in SAMPLE_STORIES
        ]
        logger.info("GetStoryCatalogue → %d stories", len(stories))
        return recommender_pb2.GetStoryCatalogueResponse(stories=stories)

    def SaveUserModel(
        self,
        request: recommender_pb2.SaveUserModelRequest,
        context: grpc.ServicerContext,
    ) -> empty_pb2.Empty:
        """Accept the recommender's compiled user model batch."""
        _user_state_store.save_models(request.user_models)
        logger.info(
            "SaveUserModel: received model for %d users",
            len(request.user_models),
        )
        return empty_pb2.Empty()

    def LoadUserModel(
        self,
        request: recommender_pb2.LoadUserModelRequest,
        context: grpc.ServicerContext,
    ) -> recommender_pb2.LoadUserModelResponse:
        """Return stored user models to the recommender.

        An empty ``user_ids`` list means "return all users".
        """
        models = _user_state_store.load_models(list(request.user_ids))
        logger.info("LoadUserModel → %d users", len(models))
        return recommender_pb2.LoadUserModelResponse(user_models=models)


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
            recommender_pb2.GetRecommendationsRequest(user_id=user_id, timestamp=_now_ts()),
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
        if event_type == "scored":
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
        elif event_type == "read_progress":
            stub.UserReadStory(
                recommender_pb2.UserReadStoryRequest(
                    user_id=user_id, story_id=story_id, read_percent=score, timestamp=ts
                ),
                timeout=3.0,
            )
        elif event_type == "bookmark":
            stub.UserBookmarkedStory(
                recommender_pb2.UserBookmarkedStoryRequest(
                    user_id=user_id, story_id=story_id, timestamp=ts
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
    handler.send_header("Cache-Control", "no-store")
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
    GET  /api/state                 Return user event log + persisted model weights
    GET  /api/users                 Return sorted list of all known user IDs
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
        elif path == "/api/users":
            _send_json(self, _user_state_store.all_user_ids())
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
                {"story_id": sid, "title": sid, "themes": [], "tags": [], "authors": []},
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
        if event_type == "scored" and (not story_id or not 1 <= score <= 10):
            _send_error(self, "story_id and score 1-10 required for scored")
            return
        if event_type == "mood" and not 1 <= score <= 10:
            _send_error(self, "score 1-10 required for mood")
            return
        if event_type == "read_progress" and (not story_id or not 0 <= score <= 100):
            _send_error(self, "story_id and score 0-100 required for read_progress")
            return
        if event_type == "bookmark" and not story_id:
            _send_error(self, "story_id required for bookmark")
            return
        if event_type not in (
            "scored", "mood", "read_progress", "bookmark"
        ):
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
        """GET /api/state?user_id=X

        Returns the local HTTP event log plus the most recently persisted
        model weights from the recommender (if available).  The model is
        updated by the recommender's periodic SaveUserModel call (default
        every 60 s) and reflects all events, including those sent directly
        over gRPC (e.g. from load_users.py).
        """
        uid = (params.get("user_id") or [""])[0].strip()
        if not uid:
            _send_error(self, "user_id query parameter required")
            return
        events = _user_state_store.get_events(uid)
        model = _user_state_store.get_model_weights(uid)
        _send_json(self, {"user_id": uid, "events": events, "model": model})


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
.story-card strong { display: block; margin-bottom: 0.15rem; }
.story-card .story-authors { font-size: 0.72rem; color: #64748b; font-style: italic; margin-bottom: 0.15rem; }
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
  <button class="btn" onclick="refreshUserList(); loadState()">&#8635; Refresh State</button>
  <button class="btn" onclick="promptMood()">&#128149; Mood</button>
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
  craft:     '#d97706',   // amber
  discovery: '#2563eb',   // blue
  belief:    '#7c3aed',   // violet
  loss:      '#475569',   // slate
  conflict:  '#dc2626',   // red
  power:     '#92400e',   // brown
  science:   '#0891b2',   // cyan
  trade:     '#059669',   // emerald
  migration: '#db2777',   // pink
  kinship:   '#65a30d',   // lime
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let catalogue = [];

// Serialise all server operations so getRecommendations always sees the
// results of any pending View/Complete/Mood events.  onclick handlers for
// story buttons are not awaited by the browser, so without this a user who
// rapidly clicks View → Complete → Get Recommendations could race the
// recommendation request ahead of the events.
let _pendingOps = Promise.resolve();

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
  const themes  = (story.themes  || []).map(themePill).join('');
  const tags    = (story.tags    || []).map(tagPill).join('');
  const authors = (story.authors || []).join(', ');
  const sid     = story.story_id;
  let buttons   = '';
  if (showButtons) {
    buttons = `<div class="card-buttons">
      <button class="btn" onclick="promptScore('${sid}')">&#11088; Score</button>
      <button class="btn" onclick="promptMood()">&#128149; Mood</button>
      <button class="btn" onclick="promptReadProgress('${sid}')">&#128336; Read%</button>
      <button class="btn" onclick="sendEvent('bookmark','${sid}')">&#128278; Bookmark</button>
    </div>`;
  }
  return `<div class="story-card">
    <strong>${story.title}</strong>
    ${authors ? `<div class="story-authors">${authors}</div>` : ''}
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
// Serialised server operations
// All calls to the server are chained through _pendingOps so that
// getRecommendations always waits for any in-flight View/Complete/Mood events
// to complete before requesting fresh recommendations.
// ---------------------------------------------------------------------------
async function getRecommendations() {
  // Wait for any pending event operations to settle first.
  await _pendingOps;

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
async function _sendEventImpl(type, storyId, score) {
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

function sendEvent(type, storyId, score) {
  // Chain this event onto the pending-ops queue so that concurrent button
  // clicks are processed in order and getRecommendations() can await them.
  _pendingOps = _pendingOps.then(() => _sendEventImpl(type, storyId, score));
}

function promptScore(storyId) {
  const s = prompt('Rate this story (1 = poor \\u2192 10 = excellent):');
  const score = parseInt(s, 10);
  if (score >= 1 && score <= 10) sendEvent('scored', storyId, score);
  else if (s !== null) alert('Please enter a number between 1 and 10.');
}

function promptMood() {
  const s = prompt('How are you feeling right now? (1 = very low \\u2192 10 = great):');
  const score = parseInt(s, 10);
  if (score >= 1 && score <= 10) sendEvent('mood', '', score);
  else if (s !== null) alert('Please enter a number between 1 and 10.');
}

function promptReadProgress(storyId) {
  const s = prompt('How far through this story are you? (0\\u2013100%):');
  const pct = parseInt(s, 10);
  if (pct >= 0 && pct <= 100) sendEvent('read_progress', storyId, pct);
  else if (s !== null) alert('Please enter a whole number between 0 and 100.');
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
    renderEventLogPanel(data.events || [], !!data.model);
    // Prefer authoritative weights from the persisted model (covers all events,
    // including those sent directly over gRPC by load_users.py).  Fall back to
    // reconstructing from the local HTTP event log for newly-active users whose
    // model hasn't been persisted yet.
    if (data.model && Object.keys(data.model.theme_weights || {}).length) {
      renderWeights(data.model.theme_weights, data.model.tag_weights);
    } else {
      const {themeW, tagW} = weightsFromEvents(data.events || []);
      renderWeights(themeW, tagW);
    }
  } catch (e) {
    console.error('loadState error', e);
  }
}

function renderEventLogPanel(events, hasModel) {
  const ul = document.getElementById('event-log-list');
  if (!events.length) {
    const note = hasModel
      ? 'No browser events recorded &mdash; interactions were sent directly over gRPC (e.g. by load_users.py). Preference weights are shown from the persisted model.'
      : 'No events recorded yet.';
    ul.innerHTML = `<li class="empty-state">${note}</li>`;
    return;
  }
  ul.innerHTML = events.slice().reverse().map(e => {
    const ts  = new Date(e.timestamp_seconds * 1000).toLocaleTimeString();
    const score = e.score ? ` score=${e.score}` : '';
    const story = e.story_id ? ` \u2192 ${e.story_id}` : '';
    return `<li>[${ts}] <strong>${e.event_type}</strong>${story}${score}</li>`;
  }).join('');
}

function weightsFromEvents(events) {
  // Reconstruct weights from the local HTTP event log using the same rules
  // as UserStateStore.  Used as a fallback when no persisted model exists yet.
  const storyMap = Object.fromEntries(catalogue.map(s => [s.story_id, s]));
  const themeW = {};
  const tagW   = {};
  for (const e of events) {
    if (!e.story_id) continue;
    const story = storyMap[e.story_id];
    if (!story) continue;
    let delta = 0;
    if (e.event_type === 'read_progress' && e.score >= 50) delta = 1.0;
    if (e.event_type === 'scored') delta = (e.score - 5) * 0.25;
    if (delta === 0) continue;
    for (const t of story.themes) themeW[t] = (themeW[t] || 0) + delta;
    for (const t of story.tags)   tagW[t]   = (tagW[t]   || 0) + delta;
  }
  return {themeW, tagW};
}

function renderWeights(themeW, tagW) {
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
// Known-user discovery
// ---------------------------------------------------------------------------
async function refreshUserList() {
  // Fetch all user IDs the mock server knows about (from saved models and
  // local event logs) and add any that aren't already in the dropdown.
  try {
    const resp = await fetch('/api/users');
    const ids = await resp.json();
    const sel = document.getElementById('user-select');
    const existing = new Set([...sel.options].map(o => o.value));
    const customOpt = sel.querySelector('option[value="__custom__"]');
    for (const uid of ids) {
      if (existing.has(uid)) continue;
      const opt = document.createElement('option');
      opt.value = uid;
      opt.textContent = uid;
      sel.insertBefore(opt, customOpt);
      existing.add(uid);
    }
  } catch (e) {
    console.error('refreshUserList error', e);
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', async () => {
  await loadCatalogue();
  await refreshUserList();
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

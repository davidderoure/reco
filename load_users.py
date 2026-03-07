#!/usr/bin/env python3
"""
load_users.py — Synthetic load generator for the recommender service.

Creates N test users (load_user_001 … load_user_N), each with a randomly-
generated but *consistent* preference profile over the 10 story themes.
The profile is seeded per-user, so re-running with the same --seed always
produces identical interaction histories.

For each user the script fires a realistic mix of events:
  • viewed          — always, when the user decides to interact with a story
  • read_progress   — 70 % of views, amount read correlated with preference
  • completed       — proportional to preference², so only truly liked stories
  • scored (1–5)    — 80 % of completions; score correlated with preference
  • mood (1–5)      — 0–5 random mood events per user, not tied to any story

Usage
-----
  python load_users.py                       # 100 users, localhost:50051
  python load_users.py --users 20            # fewer users
  python load_users.py --addr host:50051     # remote service
  python load_users.py --seed 42             # fully reproducible run
  python load_users.py --recs                # fetch recommendations per user

Requires the recommender service (python main.py) to be running.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from generated import recommender_pb2, recommender_pb2_grpc

# ---------------------------------------------------------------------------
# Story catalogue (mirrors SAMPLE_STORIES in mock_server.py)
# ---------------------------------------------------------------------------

STORIES: list[dict[str, str]] = [
    # craft
    {"story_id": "cft_001", "theme": "craft"},
    {"story_id": "cft_002", "theme": "craft"},
    {"story_id": "cft_003", "theme": "craft"},
    # discovery
    {"story_id": "dsc_001", "theme": "discovery"},
    {"story_id": "dsc_002", "theme": "discovery"},
    {"story_id": "dsc_003", "theme": "discovery"},
    # belief
    {"story_id": "blf_001", "theme": "belief"},
    {"story_id": "blf_002", "theme": "belief"},
    {"story_id": "blf_003", "theme": "belief"},
    # loss
    {"story_id": "los_001", "theme": "loss"},
    {"story_id": "los_002", "theme": "loss"},
    {"story_id": "los_003", "theme": "loss"},
    # conflict
    {"story_id": "cnf_001", "theme": "conflict"},
    {"story_id": "cnf_002", "theme": "conflict"},
    {"story_id": "cnf_003", "theme": "conflict"},
    # power
    {"story_id": "pwr_001", "theme": "power"},
    {"story_id": "pwr_002", "theme": "power"},
    {"story_id": "pwr_003", "theme": "power"},
    # science
    {"story_id": "sci_001", "theme": "science"},
    {"story_id": "sci_002", "theme": "science"},
    {"story_id": "sci_003", "theme": "science"},
    # trade
    {"story_id": "trd_001", "theme": "trade"},
    {"story_id": "trd_002", "theme": "trade"},
    # migration
    {"story_id": "mgr_001", "theme": "migration"},
    {"story_id": "mgr_002", "theme": "migration"},
    {"story_id": "mgr_003", "theme": "migration"},
    # kinship
    {"story_id": "kin_001", "theme": "kinship"},
    {"story_id": "kin_002", "theme": "kinship"},
    {"story_id": "kin_003", "theme": "kinship"},
]

THEMES: list[str] = sorted({s["theme"] for s in STORIES})

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("load_users")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIMEOUT = 5.0  # seconds — applied to every RPC call


def _ts(dt: datetime) -> Timestamp:
    """Convert a datetime to a protobuf Timestamp."""
    t = Timestamp()
    t.FromDatetime(dt.astimezone(timezone.utc))
    return t


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Per-user preference model
# ---------------------------------------------------------------------------

def _theme_preferences(rng: random.Random) -> dict[str, float]:
    """Return a preference weight in [0, 1] per theme for one user.

    Uses a Beta(0.4, 1.5) draw per theme — this distribution is strongly
    right-skewed (most draws near 0), so users typically have a handful of
    strong interests and ignore most themes.  The weights are then normalised
    so the user's top theme always scores 1.0, making the profile shape
    clear when rendered in the browser's Preference Weights panel.
    """
    raw = {theme: rng.betavariate(0.4, 1.5) for theme in THEMES}
    max_w = max(raw.values()) or 1.0
    return {t: w / max_w for t, w in raw.items()}


# ---------------------------------------------------------------------------
# Event generation for one user
# ---------------------------------------------------------------------------

def _simulate_user(
    user_id: str,
    stub: recommender_pb2_grpc.RecommenderServiceStub,
    rng: random.Random,
    *,
    get_recs: bool = False,
) -> int:
    """Generate and send all events for one user.  Returns the RPC count."""
    prefs = _theme_preferences(rng)

    # Overall engagement factor: how active this user is (0.05 … 1.0).
    # Beta(1.5, 1.5) is symmetric and centred around 0.5.
    engagement = rng.betavariate(1.5, 1.5)

    # Spread interactions across a 60-day window ending now.
    now = datetime.now(timezone.utc)
    window_s = 60 * 24 * 3600
    start = now - timedelta(seconds=window_s)

    rpc_count = 0

    # Shuffle story order so the interaction sequence differs per user.
    shuffled = STORIES[:]
    rng.shuffle(shuffled)

    for story in shuffled:
        pref = prefs[story["theme"]]

        # View probability: higher engagement and stronger preference both
        # increase the chance of viewing.  Floor at 5 % so even ignored
        # themes get occasional exploration.
        view_prob = engagement * (0.05 + 0.95 * pref)
        if rng.random() > view_prob:
            continue

        # Pick a random moment in the 60-day window for this story's events.
        event_dt = start + timedelta(seconds=rng.uniform(0, window_s))
        sid = story["story_id"]

        # ── viewed ───────────────────────────────────────────────────────────
        stub.UserViewedStory(
            recommender_pb2.UserViewedStoryRequest(
                user_id=user_id, story_id=sid, timestamp=_ts(event_dt)
            ),
            timeout=_TIMEOUT,
        )
        rpc_count += 1

        # ── read_progress (70 % of views) ────────────────────────────────────
        # Amount read follows a triangular distribution whose mode is at
        # pref × 100, so preferred stories tend to be read further.
        if rng.random() < 0.70:
            read_pct = int(rng.triangular(0, 100, pref * 100))
            stub.UserReadStory(
                recommender_pb2.UserReadStoryRequest(
                    user_id=user_id,
                    story_id=sid,
                    read_percent=read_pct,
                    timestamp=_ts(event_dt + timedelta(minutes=rng.uniform(1, 20))),
                ),
                timeout=_TIMEOUT,
            )
            rpc_count += 1

        # ── completed (probability ∝ pref² × engagement) ─────────────────────
        # Squaring the preference makes completion selective: a story needs to
        # be genuinely liked (pref ≳ 0.7) before completion becomes likely.
        complete_prob = (pref ** 2) * engagement * 0.8
        if rng.random() < complete_prob:
            stub.UserCompletedStory(
                recommender_pb2.UserCompletedStoryRequest(
                    user_id=user_id,
                    story_id=sid,
                    timestamp=_ts(event_dt + timedelta(minutes=rng.uniform(5, 30))),
                ),
                timeout=_TIMEOUT,
            )
            rpc_count += 1

            # ── scored (80 % of completions) ─────────────────────────────────
            # Score is normally distributed around (1 + pref × 4), clipped to
            # [1, 5], so preferred stories earn 4–5 and disliked ones earn 1–2.
            if rng.random() < 0.80:
                raw = rng.gauss(1.0 + pref * 4.0, 0.8)
                score = int(_clamp(round(raw), 1, 5))
                stub.UserAnsweredQuestion(
                    recommender_pb2.UserAnsweredQuestionRequest(
                        user_id=user_id,
                        story_id=sid,
                        score=score,
                        timestamp=_ts(event_dt + timedelta(minutes=rng.uniform(10, 35))),
                    ),
                    timeout=_TIMEOUT,
                )
                rpc_count += 1

    # ── mood events (0–5 per user, scattered across the window) ──────────────
    for _ in range(rng.randint(0, 5)):
        mood_dt = start + timedelta(seconds=rng.uniform(0, window_s))
        stub.UserProvidedMood(
            recommender_pb2.UserProvidedMoodRequest(
                user_id=user_id,
                mood_score=rng.randint(1, 5),
                timestamp=_ts(mood_dt),
            ),
            timeout=_TIMEOUT,
        )
        rpc_count += 1

    # ── optional recommendation fetch ────────────────────────────────────────
    if get_recs:
        resp = stub.GetRecommendations(
            recommender_pb2.GetRecommendationsRequest(
                user_id=user_id,
                timestamp=_ts(datetime.now(timezone.utc)),
            ),
            timeout=_TIMEOUT,
        )
        rpc_count += 1
        # Log the top themes of the returned stories for a quick sanity check.
        theme_by_id = {s["story_id"]: s["theme"] for s in STORIES}
        themes_str = ", ".join(
            f"{sid}({theme_by_id.get(sid, '?')})" for sid in resp.story_ids
        )
        log.info("  recs  %-20s  %s", user_id, themes_str)

    return rpc_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--users", type=int, default=100, metavar="N",
        help="Number of synthetic users to create (default: 100)",
    )
    p.add_argument(
        "--addr", default="localhost:50051", metavar="HOST:PORT",
        help="Recommender gRPC address (default: localhost:50051)",
    )
    p.add_argument(
        "--seed", type=int, default=None, metavar="INT",
        help="Global RNG seed for a fully reproducible run (default: random)",
    )
    p.add_argument(
        "--recs", action="store_true",
        help="Call GetRecommendations for each user after generating events",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # The global seed controls per-user seed generation.  Logging it means a
    # run can always be reproduced even when --seed was not supplied.
    global_rng = random.Random(args.seed)
    effective_seed = args.seed if args.seed is not None else global_rng.randint(0, 2**32 - 1)
    if args.seed is None:
        global_rng = random.Random(effective_seed)
    log.info("Global seed: %d", effective_seed)

    # Pre-generate per-user seeds so that the seed for load_user_001 is always
    # the same regardless of how many users are requested.
    user_seeds = [global_rng.randint(0, 2**32 - 1) for _ in range(args.users)]

    log.info("Connecting to recommender at %s …", args.addr)
    channel = grpc.insecure_channel(args.addr)
    stub = recommender_pb2_grpc.RecommenderServiceStub(channel)

    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except grpc.FutureTimeoutError:
        log.error("Cannot reach recommender at %s — is it running?", args.addr)
        sys.exit(1)

    log.info("Generating events for %d users …", args.users)
    t0 = time.monotonic()
    total_rpcs = 0

    for i in range(args.users):
        user_id = f"load_user_{i + 1:03d}"
        user_rng = random.Random(user_seeds[i])
        try:
            n = _simulate_user(user_id, stub, user_rng, get_recs=args.recs)
            total_rpcs += n
            log.info("[%3d/%d] %-22s  %3d RPCs sent", i + 1, args.users, user_id, n)
        except grpc.RpcError as exc:
            log.error("gRPC error for %s: %s — skipping", user_id, exc)

    elapsed = time.monotonic() - t0
    log.info(
        "Done.  %d users · %d total RPCs · %.1f s · %.0f RPCs/s",
        args.users, total_rpcs, elapsed, total_rpcs / max(elapsed, 0.001),
    )


if __name__ == "__main__":
    main()

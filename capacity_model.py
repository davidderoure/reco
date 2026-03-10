#!/usr/bin/env python3
"""
capacity_model.py — Persistence capacity model for the Story Recommender.

Models the volume and frequency of UserModel state saves to and loads from
the C# backend.  Edit the PARAMETERS section at the top, then run:

    python capacity_model.py

Two modes:
  Analytical  — formula-based estimates (always available)
  Empirical   — serialises a real UserModelMessage and compares sizes
                (requires: make proto)
"""

from __future__ import annotations

import math
import random

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit these to match your deployment target
# ═══════════════════════════════════════════════════════════════════════════════

# ── Story catalogue ───────────────────────────────────────────────────────────
N_STORIES: int = 27           # total stories in catalogue (27 in the mock)
N_THEMES: int = 10            # distinct theme labels      (10 in mock)
N_TAGS: int = 40              # distinct tags across all stories (estimate)
AVG_STORY_ID_BYTES: int = 8   # bytes in a typical story_id  (e.g. "story_01")
AVG_THEME_BYTES: int = 12     # bytes in a typical theme name (e.g. "architecture")
AVG_TAG_BYTES: int = 10       # bytes in a typical tag name   (e.g. "industrial")

# ── Deployment scales ─────────────────────────────────────────────────────────
# Add or remove entries freely.
SCALES: dict[str, dict] = {
    "small":  dict(n_users_total=1_000,    peak_concurrent=100),
    "medium": dict(n_users_total=50_000,   peak_concurrent=5_000),
    "large":  dict(n_users_total=500_000,  peak_concurrent=50_000),
}

# ── User state maturity bands ─────────────────────────────────────────────────
# frac_* is the fraction of N_STORIES present in each set/map.
# n_themes_weighted / n_tags_weighted are non-zero entries in the weight maps.
# n_mood_entries is capped at MOOD_HISTORY_LIMIT = 50 in user_state.py.
MATURITY_BANDS: dict[str, dict] = {
    "fresh": dict(
        frac_viewed=0.05, frac_completed=0.01, frac_scored=0.01,
        frac_recommended=0.10, n_themes_weighted=3,
        n_tags_weighted=5, n_mood_entries=5,
    ),
    "typical": dict(
        frac_viewed=0.40, frac_completed=0.15, frac_scored=0.10,
        frac_recommended=0.60, n_themes_weighted=N_THEMES,
        n_tags_weighted=20, n_mood_entries=25,
    ),
    "mature": dict(
        frac_viewed=1.00, frac_completed=0.50, frac_scored=0.40,
        frac_recommended=1.00, n_themes_weighted=N_THEMES,
        n_tags_weighted=N_TAGS, n_mood_entries=50,
    ),
}

# ── Persistence timing ────────────────────────────────────────────────────────
FLUSH_INTERVAL_S: int = 60    # STATE_PERSIST_INTERVAL_SECONDS (config default)

# Fraction of in-memory users that have unsaved changes each flush cycle.
# Current implementation: always 1.0 (no dirty-flag — all users saved every time).
# Set to a lower value to model a hypothetical dirty-flag optimisation.
DIRTY_FRACTION: float = 1.0

# ── Network / RPC assumptions ─────────────────────────────────────────────────
RPC_LATENCY_MS: int = 10      # round-trip latency to C# backend (ms)
NETWORK_BW_MBPS: int = 100    # available bandwidth for gRPC traffic (Mbps)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════
#
# Proto3 wire-format approximations:
#   repeated string entry     = 1 (tag) + 1 (len varint) + string_bytes
#   map<string,float> entry   = 2 (entry overhead) + 1+1+key_bytes (key field)
#                               + 1 (value tag) + 4 (float32)
#   map<string,int32> entry   = 2 + 1+1+key_bytes + 1 + 2 (small int32 varint)
#   repeated MoodEntry entry  ≈ 2 (msg overhead) + 10 (Timestamp) + 2 (score)
#   user_id string field      = 2 + len
# A +10% factor covers field-number overhead and varint length bytes missed
# by the simplified formula.  The empirical section validates this accuracy.

def per_user_bytes(
    frac_viewed: float,
    frac_completed: float,
    frac_scored: float,
    frac_recommended: float,
    n_themes_weighted: int,
    n_tags_weighted: int,
    n_mood_entries: int,
) -> int:
    """Return estimated protobuf serialised size for one UserModelMessage."""
    n = N_STORIES
    s = AVG_STORY_ID_BYTES

    user_id = 2 + 10                                       # ~10-char user ID
    viewed = math.ceil(frac_viewed * n)        * (2 + s)
    completed = math.ceil(frac_completed * n)  * (2 + s)
    story_scores = math.ceil(frac_scored * n)  * (2 + 2 + s + 2)
    mood = n_mood_entries                       * 14        # 2+10+2
    theme_weights = n_themes_weighted           * (2 + 2 + AVG_THEME_BYTES + 5)
    tag_weights = n_tags_weighted               * (2 + 2 + AVG_TAG_BYTES + 5)
    last_recs = 6                               * (2 + s)
    all_recs = math.ceil(frac_recommended * n) * (2 + s)

    raw = (user_id + viewed + completed + story_scores + mood
           + theme_weights + tag_weights + last_recs + all_recs)
    return int(raw * 1.10)   # +10% framing overhead


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_bytes(b: float) -> str:
    """Human-readable byte count."""
    if b < 1_024:
        return f"{b:.0f} B"
    if b < 1_024 ** 2:
        return f"{b / 1_024:.1f} KB"
    if b < 1_024 ** 3:
        return f"{b / 1_024 ** 2:.1f} MB"
    return f"{b / 1_024 ** 3:.2f} GB"


def fmt_duration(ms: float) -> str:
    """Human-readable duration from milliseconds."""
    if ms < 1_000:
        return f"{ms:.0f} ms"
    return f"{ms / 1_000:.1f} s"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Per-user state size by maturity band
# ═══════════════════════════════════════════════════════════════════════════════

def print_size_table() -> None:
    """Print estimated per-user protobuf size for each maturity band."""
    print(f"\n── Table 1: Per-user state size by maturity band {'─' * 29}")
    print(f"  Catalogue: {N_STORIES} stories | {N_THEMES} themes | {N_TAGS} tags"
          f"  (story_id ≈ {AVG_STORY_ID_BYTES}B, theme ≈ {AVG_THEME_BYTES}B, tag ≈ {AVG_TAG_BYTES}B)\n")

    hdr = (f"  {'Maturity':<10}  {'viewed':>7}  {'completed':>9}  {'scored':>6}"
           f"  {'mood':>4}  {'all_recs':>8}  {'Size':>8}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for name, band in MATURITY_BANDS.items():
        size = per_user_bytes(**band)
        viewed = math.ceil(band["frac_viewed"] * N_STORIES)
        completed = math.ceil(band["frac_completed"] * N_STORIES)
        scored = math.ceil(band["frac_scored"] * N_STORIES)
        all_recs = math.ceil(band["frac_recommended"] * N_STORIES)
        print(f"  {name:<10}  {viewed:>7}  {completed:>9}  {scored:>6}"
              f"  {band['n_mood_entries']:>4}  {all_recs:>8}  {fmt_bytes(size):>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2: Save / load volumes by deployment scale
# ═══════════════════════════════════════════════════════════════════════════════

def print_scale_table() -> None:
    """Print save payload and startup load time for each deployment scale."""
    print(f"\n── Table 2: Save / load volumes by deployment scale {'─' * 26}")
    print(f"  Flush interval: {FLUSH_INTERVAL_S}s  |  Dirty fraction: {DIRTY_FRACTION:.0%}"
          f"  |  Network: {NETWORK_BW_MBPS} Mbps  |  RPC latency: {RPC_LATENCY_MS} ms\n")

    typical = per_user_bytes(**MATURITY_BANDS["typical"])

    hdr = (f"  {'Scale':<8}  {'Users':>8}  {'State (total)':>13}"
           f"  {'Save payload':>12}  {'Save RPC/min':>12}  {'Startup load':>12}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for scale_name, scale in SCALES.items():
        n_total = scale["n_users_total"]
        total_state = n_total * typical

        # One batch SaveUserModel RPC every FLUSH_INTERVAL_S seconds,
        # containing all in-memory users (no dirty-flag optimisation).
        save_payload = n_total * DIRTY_FRACTION * typical
        saves_per_min = 60 / FLUSH_INTERVAL_S

        # Startup: one batch LoadUserModel RPC for all users.
        load_mb = total_state / (1_024 ** 2)
        transfer_ms = (load_mb * 8 / NETWORK_BW_MBPS) * 1_000
        load_time_ms = RPC_LATENCY_MS + transfer_ms

        print(f"  {scale_name:<8}  {n_total:>8,}  {fmt_bytes(total_state):>13}"
              f"  {fmt_bytes(save_payload):>12}  {saves_per_min:>12.1f}"
              f"  {fmt_duration(load_time_ms):>12}")

    print(f"\n  Note: save payload = all in-memory users × per-user bytes (no dirty-flag).")
    print(f"  Per-user size: {fmt_bytes(typical)} (typical maturity band).")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3: Save bandwidth vs. flush interval (matrix)
# ═══════════════════════════════════════════════════════════════════════════════

def print_bandwidth_matrix() -> None:
    """Print save bandwidth for each scale × flush interval combination."""
    print(f"\n── Table 3: Save bandwidth (bytes/sec) — scale × flush interval {'─' * 13}")

    typical = per_user_bytes(**MATURITY_BANDS["typical"])
    intervals = [30, 60, 120, 300]
    col_w = 12

    hdr = f"  {'Flush →':>{col_w}}" + "".join(f"  {f'{i}s':>{col_w}}" for i in intervals)
    sep = "  " + "─" * (col_w + len(intervals) * (col_w + 2))
    print(f"\n  Per-user size: {fmt_bytes(typical)} (typical maturity)\n")
    print(sep)
    print(hdr)
    print(sep)

    for scale_name, scale in SCALES.items():
        n_total = scale["n_users_total"]
        row = f"  {scale_name:>{col_w}}"
        for interval in intervals:
            bw = n_total * typical * DIRTY_FRACTION / interval
            row += f"  {(fmt_bytes(bw) + '/s'):>{col_w}}"
        print(row)

    print(sep)
    print(f"\n  Dirty fraction: {DIRTY_FRACTION:.0%}"
          f"  (1.0 = current, no dirty-flag; lower = hypothetical optimisation)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 4: Dirty-flag optimisation comparison
# ═══════════════════════════════════════════════════════════════════════════════

def print_dirty_flag_comparison() -> None:
    """Show how much bandwidth a dirty-flag optimisation would save."""
    print(f"\n── Table 4: Dirty-flag optimisation impact (medium scale) {'─' * 21}")

    typical = per_user_bytes(**MATURITY_BANDS["typical"])
    scale = SCALES["medium"]
    n_total = scale["n_users_total"]
    dirty_fractions = [1.0, 0.50, 0.10, 0.02]

    print(f"\n  Scale: medium ({n_total:,} users)  |  Flush every {FLUSH_INTERVAL_S}s\n")
    hdr = f"  {'Dirty fraction':>16}  {'Dirty users':>12}  {'Save payload':>12}  {'Bandwidth':>12}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for frac in dirty_fractions:
        dirty_users = int(n_total * frac)
        save_payload = dirty_users * typical
        bw = save_payload / FLUSH_INTERVAL_S
        label = "1.0 (current)" if frac == 1.0 else f"{frac:.0%} (hypothetical)"
        print(f"  {label:>16}  {dirty_users:>12,}  {fmt_bytes(save_payload):>12}"
              f"  {(fmt_bytes(bw) + '/s'):>12}")

    print(f"\n  A dirty-flag would make save cost proportional to active users in the"
          f"\n  flush window, rather than total in-memory users.  At 10% dirty fraction,"
          f"\n  bandwidth drops 10×.  Implementation cost: one boolean per UserProfile.")


# ═══════════════════════════════════════════════════════════════════════════════
# EMPIRICAL MEASUREMENT (requires: make proto)
# ═══════════════════════════════════════════════════════════════════════════════

def measure_proto_size() -> None:
    """Serialise a real UserModelMessage per maturity band and compare to formula."""
    print(f"\n── Empirical measurement: actual proto wire sizes {'─' * 29}")

    try:
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))
        from recommender_pb2 import UserModelMessage  # type: ignore[import]
        from google.protobuf.timestamp_pb2 import Timestamp
    except (ImportError, ModuleNotFoundError):
        print("  Skipped — run 'make proto' first to generate proto bindings.\n")
        return

    from datetime import datetime, timezone

    rng = random.Random(42)
    base_ts = Timestamp()
    base_ts.FromDatetime(datetime.now(timezone.utc))

    story_ids = [f"story_{i:03d}" for i in range(N_STORIES)]
    themes = [f"theme_{i:02d}" for i in range(N_THEMES)]
    tags = [f"tag_{i:03d}" for i in range(N_TAGS)]

    hdr = f"  {'Maturity':<10}  {'Analytical':>12}  {'Empirical':>10}  {'Delta':>7}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))

    for name, band in MATURITY_BANDS.items():
        n_viewed = math.ceil(band["frac_viewed"] * N_STORIES)
        n_completed = math.ceil(band["frac_completed"] * N_STORIES)
        n_scored = math.ceil(band["frac_scored"] * N_STORIES)
        n_recs = math.ceil(band["frac_recommended"] * N_STORIES)
        n_mood = band["n_mood_entries"]
        n_tw = band["n_themes_weighted"]
        n_tagw = band["n_tags_weighted"]

        msg = UserModelMessage()
        msg.user_id = "load_user_042"
        msg.viewed_story_ids[:] = story_ids[:n_viewed]
        msg.completed_story_ids[:] = story_ids[:n_completed]
        for sid in story_ids[:n_scored]:
            msg.story_scores[sid] = rng.randint(1, 10)
        for _ in range(n_mood):
            entry = msg.mood_scores.add()
            entry.score = rng.randint(1, 10)
            entry.timestamp.CopyFrom(base_ts)
        for th in themes[:n_tw]:
            msg.theme_weights[th] = rng.uniform(0.1, 5.0)
        for tg in tags[:n_tagw]:
            msg.tag_weights[tg] = rng.uniform(0.1, 5.0)
        msg.last_recommendations[:] = story_ids[:6]
        msg.recommended_story_ids[:] = story_ids[:n_recs]

        empirical = len(msg.SerializeToString())
        analytical = per_user_bytes(**band)
        delta = (empirical - analytical) / analytical * 100

        print(f"  {name:<10}  {fmt_bytes(analytical):>12}  {fmt_bytes(empirical):>10}"
              f"  {delta:>+6.1f}%")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("  Story Recommender — Persistence Capacity Model")
    print("=" * 78)
    print_size_table()
    print_scale_table()
    print_bandwidth_matrix()
    print_dirty_flag_comparison()
    measure_proto_size()
    print()

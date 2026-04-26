# gRPC Interface Comparison: V1 (main) vs V2 (tag-based-grpc)

## Architecture Overview

**Three-tier architecture:**
```
Mobile App ←→ C# Server ←→ Python Service
(UI/UX)       (orchestration,  (recommendation
               data storage)    computation)
```

**App and C# are self-contained** - the system works without Python (just no recommendations)

**Bidirectional gRPC between C# and Python:**
- **C# → Python**: Forward events, request recommendations
- **Python → C#**: Get catalog, save/load state

**Data ownership:**
- **C# server**: Source of truth (CMS stories, event log, saved user state)
- **Python service**: Computation engine (real-time recommendations, in-memory models)

**Flow:**
1. Python loads user state from C# at startup (`LoadUserModel`)
2. Python fetches story catalog from C# CMS (`GetStoryCatalogue`)
3. App sends events to C# server (HTTP/WebSocket)
4. C# logs events, then forwards to Python (`UserReadStory`, etc.)
5. C# requests recommendations from Python when needed
6. Python updates models in memory and returns recommendations to C#
7. C# sends recommendations back to app
8. Python saves state back to C# every 60s (`SaveUserModel`)
9. Daily exports: C# reads its own saved state (Python not involved)

---

## Executive Summary

**Good news:** The core structure is similar! The changes are primarily:
1. **Conceptual shift**: Mood → Connectedness
2. **Story metadata**: Themes+Tags → Tags only  
3. **Recommendation transparency**: Added method/slot tracking
4. **State management**: Split operational vs analytical

**Migration Impact:**
- C# team needs to update event handlers and proto definitions
- Python recommendation logic is completely new (but that's the point!)
- State persistence format has changed

---

## Side-by-Side: Event Messages

### User Interaction Events

| V1 (main) | V2 (tag-based-grpc) | Change Type |
|-----------|---------------------|-------------|
| `UserAnsweredQuestionRequest`<br>- score: 1-10<br>- question_number: 1-4 | `AnalyticsEvent` (type: "question_response")<br>- question_number: 1-4<br>- response: 1-5 | **Scale changed** 1-10 → 1-5<br>**Same 4 questions** |
| `UserProvidedMoodRequest`<br>- mood_score: 1-10<br>- optional story_id | **REMOVED** | **Deleted**<br>Replaced by Q1 (connectedness) |
| `UserReadStoryRequest`<br>- read_percent: 0-100<br>≥50% = viewed, 100% = completed | `AnalyticsEvent` (type: "story_progress")<br>- completion_percentage: 0-100<br>Same thresholds | **Renamed**, same logic |
| `UserBookmarkedStoryRequest` | `AnalyticsEvent` (type: "bookmark") | **Renamed** from "favorite" |
| N/A | `AnalyticsEvent` (type: "view")<br>Explicit story open event | **New**<br>Now tracks opens explicitly |
| N/A | `AnalyticsEvent` (type: "search")<br>Tag browsing events | **New**<br>Tracks tag exploration |

### Recommendation Request/Response

| V1 (main) | V2 (tag-based-grpc) | Change |
|-----------|---------------------|--------|
| `GetRecommendationsRequest`<br>- user_id<br>- timestamp | `RecommendationRequest`<br>- user_id<br>- boost_tags (optional)<br>- prefer_new_stories (optional)<br>- count (default: 6) | **Added context params** |
| `GetRecommendationsResponse`<br>- story_ids (6 IDs) | `RecommendationResponse`<br>- recommendations[] with:<br>&nbsp;&nbsp;• story_id<br>&nbsp;&nbsp;• title<br>&nbsp;&nbsp;• tags<br>&nbsp;&nbsp;• score<br>&nbsp;&nbsp;• **method** (new!)<br>&nbsp;&nbsp;• **slot_position** (new!)<br>&nbsp;&nbsp;• avg_connectedness | **Added provenance**<br>Now tracks WHY recommended |

---

## Story Structure

### V1 (main) - `StoryMessage`
```protobuf
message StoryMessage {
  string story_id = 1;
  string title = 2;
  repeated string themes = 3;    // Broad categories (exactly 1 per story in practice)
  repeated string tags = 4;      // Fine-grained tags
  repeated string authors = 5;   // Display names
}
```

### V2 (tag-based-grpc) - `Story`
```protobuf
message Story {
  string story_id = 1;
  string title = 2;
  repeated string tags = 3;      // Multiple tags only (no themes)
}
```

**Change:** `themes` field **removed**. Tags now do all the work (1+ per story).

---

## State Persistence

### V1 (main) - `UserModelMessage`

```protobuf
message UserModelMessage {
  string user_id = 1;
  repeated string viewed_story_ids = 2;
  repeated string completed_story_ids = 3;
  map<string, int32> story_scores = 4;         // 1-10 ratings
  repeated MoodEntry mood_scores = 5;           // Recent mood history
  map<string, float> theme_weights = 6;         // Accumulated weights
  map<string, float> tag_weights = 7;           // Accumulated weights
  repeated string last_recommendations = 8;
  repeated string recommended_story_ids = 9;
}
```

**Use:** `SaveUserModel()` called every 60 seconds

### V2 (tag-based-grpc) - Two Modes

**Operational (every few minutes):**
```json
{
  "users": {
    "user_123": {
      "viewed_stories": {...},
      "story_progress": {...},
      "tag_interactions": {...},
      "story_connectedness": {...},
      "story_ignore_count": {...}
    }
  },
  "stories": {...},
  "config": {...}
}
```

**Analytical (daily):**
```json
{
  // Everything from operational, PLUS:
  "users": {
    "user_123": {
      "recommendations_shown": [        // NEW - provenance
        {
          "story_id": "story1",
          "method": "content",
          "slot_position": 0,
          "selected": true
        }
      ],
      "question_responses": [...],      // NEW - all Q&A
      "story_sequences": [...]           // NEW - transitions
    }
  },
  "events": [...],                       // NEW - full event log
  "story_transitions": [...]             // NEW - global patterns
}
```

**Changes:**
1. **Split into two exports** (operational + analytical)
2. **Removed** `mood_scores`, `theme_weights`
3. **Added** recommendation provenance, sequences, full event log

---

## Conceptual Changes

### V1 Philosophy (main)
- **Primary signal:** Mood (1-10 scale)
- **Story organization:** Single theme + tags
- **Recommendation:** 6 slots (2-2-1-1) with mood-responsive allocation
- **State:** Single UserModel saved every 60s

### V2 Philosophy (tag-based-grpc)
- **Primary signal:** Connectedness (Q1, 1-5 scale)
- **Story organization:** Multiple tags only
- **Recommendation:** Configurable slots with provenance tracking
- **State:** Dual-mode (operational + analytical)

---

## Migration Path Options

### Option A: Clean Break (Recommended)

**Pros:**
- Cleaner codebase
- Better architecture going forward
- All changes discussed at once

**Implementation:**
1. C# team implements new proto definitions
2. Update event handlers (mood → connectedness Q1)
3. Change scale from 1-10 to 1-5
4. Add method/slot tracking to recommendation display
5. Update SaveUserModel to new format

**Timeline:** 2-3 weeks

### Option B: Backward Compatibility Layer

Add deprecated fields to ease transition:

```protobuf
message AnalyticsEvent {
  // V2 (new)
  string event_type = 2;
  int32 question_number = 6;
  int32 response = 7;  // 1-5
  
  // V1 (deprecated)
  int32 score = 10 [deprecated = true];  // 1-10, mapped to response
  int32 mood_score = 11 [deprecated = true];  // Maps to Q1
}

message Story {
  repeated string tags = 3;
  string theme = 4 [deprecated = true];  // First tag used as theme
}
```

**Pros:** Phased migration  
**Cons:** Maintaining two versions

### Option C: V2 Backend, V1 Interface Adapter

Keep V1 proto, translate in Python service layer:

```python
def UserAnsweredQuestion_v1(request):
    # Translate V1 score (1-10) to V2 response (1-5)
    v2_response = round((request.score - 1) * 4 / 9) + 1
    
    v2_event = AnalyticsEvent(
        event_type="question_response",
        question_number=request.question_number,
        response=v2_response
    )
    recommender.add_event(v2_event)
```

**Pros:** No C# changes  
**Cons:** Loses precision, technical debt

---

## Recommendation for Tomorrow's Meeting

**Present Option A** (clean break) with this framing:

### 1. Show What Stayed the Same ✅
- 4 questions (still there!)
- Read progress tracking (just renamed)
- Bookmarks (just renamed from favorites)
- 6 recommendations (same count)
- State persistence concept (enhanced, not replaced)

### 2. Explain What Changed & Why 🎯

| Change | Old Way | New Way | Why Better |
|--------|---------|---------|------------|
| **Primary signal** | Mood (1-10) | Connectedness (1-5, Q1) | More specific to story relevance |
| **Scale** | 1-10 | 1-5 | Better UX, clearer choices |
| **Story metadata** | Theme + tags | Tags only | More flexible, multi-faceted |
| **Rec transparency** | Just IDs | IDs + method + slot | Understand recommendations |
| **State saves** | One mode | Operational + analytical | Fault tolerance + testing |

### 3. Migration Impact 📋

**C# Team (2 weeks):**
- [ ] Update proto definitions
- [ ] Change event handlers (6 events → 5 events)
- [ ] Update scale (1-10 → 1-5)
- [ ] Add method badges to recommendation UI
- [ ] Update SaveUserModel/LoadUserModel

**Python Team (Already done!):**
- [x] New recommendation engine
- [x] Tag-based architecture
- [x] Dual-mode state management
- [x] Comprehensive documentation

---

## Key Talking Points for Meeting

1. **"The interface changes because the model changed"**
   - Moving from mood to connectedness is the core improvement
   - Tags > themes gives better content organization

2. **"Most events are the same, just renamed"**
   - Read progress = same thing
   - Bookmark = same thing (was favorite)
   - Questions = same structure (1-10 → 1-5)

3. **"We added transparency, not complexity"**
   - Recommendations now include *why* they were suggested
   - Helps debug and improves user trust

4. **"State persistence is more robust"**
   - Operational: Fast, frequent (fault tolerance)
   - Analytical: Comprehensive (testing/evaluation)
   - v1's SaveUserModel → v2's operational mode

5. **"We can help with migration"**
   - Mock server available for C# testing
   - Clear proto definitions in README
   - 2-week timeline is realistic

---

## Files to Review Together

1. **Old:** `proto/recommender.proto` (main branch)
2. **New:** README.md section "gRPC Interface Definition"
3. **New:** README.md section "State Management"

This document should help structure the discussion!

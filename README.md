# Story Recommender System - Technical Documentation

## Overview

This is a personalized story recommendation system designed for a mobile app that helps users discover stories from Oxford University museum artifacts. The system learns from user behavior and provides recommendations through multiple strategies: content-based filtering, collaborative filtering, topical boosting, and exploration.

**Key Design Principles:**
- User autonomy: Users can self-select away from triggering content
- Connectedness as primary signal: Measures how much users relate to stories
- Sequence awareness: Learns which stories work well in succession
- Configurable recommendation slots: Different methods can be adjusted
- Tag-based organization: Stories organized by multiple tags (no single theme)

## System Architecture

```
┌─────────────────┐
│   Mobile App    │
│    (C#/MAUI)    │
└────────┬────────┘
         │ gRPC
         ▼
┌─────────────────┐
│  Python Service │
│  (Recommender)  │
│                 │
│  • Events       │
│  • State        │
│  • Algorithms   │
└─────────────────┘
```

The mobile app sends analytics events via gRPC and receives recommendations. The Python service maintains user profiles and story data, updating recommendations based on user interactions.

## Sequence Diagram

```
User          App                    Recommender Service
 │             │                              │
 │  Opens app  │                              │
 ├────────────>│                              │
 │             │  GetRecommendations()        │
 │             ├─────────────────────────────>│
 │             │                              │
 │             │  6 stories with methods      │
 │             │<─────────────────────────────┤
 │             │                              │
 │ Selects     │                              │
 │ story #3    │                              │
 ├────────────>│                              │
 │             │  UserViewedStory()           │
 │             ├─────────────────────────────>│
 │             │                              │
 │ Reads       │                              │
 │ story       │  (scroll tracking)           │
 │ (scrolls    │                              │
 │  to 87%)    │                              │
 │             │                              │
 │ Finishes    │                              │
 ├────────────>│                              │
 │             │  UserReadStory()             │
 │             │  read_percent=100            │
 │             ├─────────────────────────────>│
 │             │                              │
 │ Shows       │                              │
 │ questions   │                              │
 │             │                              │
 │ Q1: 5/5     │                              │
 │ (connect)   │                              │
 ├────────────>│                              │
 │             │  UserAnsweredQuestion()      │
 │             │  question_num=1, response=5  │
 │             ├─────────────────────────────>│
 │             │                              │
 │ Q2: 4/5     │                              │
 ├────────────>│                              │
 │             │  UserAnsweredQuestion()      │
 │             │  question_num=2, response=4  │
 │             ├─────────────────────────────>│
 │             │                              │
 │ Bookmarks   │                              │
 ├────────────>│                              │
 │             │  UserBookmarkedStory()       │
 │             ├─────────────────────────────>│
 │             │                              │
 │ Requests    │                              │
 │ more recs   │                              │
 ├────────────>│                              │
 │             │  GetRecommendations()        │
 │             ├─────────────────────────────>│
 │             │  (uses connectedness=5       │
 │             │   to find similar stories)   │
 │             │                              │
 │             │  6 new stories               │
 │             │<─────────────────────────────┤
```

## gRPC Interface Definition

**recommender.proto:**

```protobuf
syntax = "proto3";

package recommender;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

// Main service interface
service StoryRecommender {
  // Fire-and-forget event notifications
  rpc UserAnsweredQuestion(UserAnsweredQuestionRequest) returns (google.protobuf.Empty);
  rpc UserReadStory(UserReadStoryRequest) returns (google.protobuf.Empty);
  rpc UserBookmarkedStory(UserBookmarkedStoryRequest) returns (google.protobuf.Empty);
  rpc UserViewedStory(UserViewedStoryRequest) returns (google.protobuf.Empty);
  rpc UserSearchedTag(UserSearchedTagRequest) returns (google.protobuf.Empty);
  
  // Request/response: returns recommendations
  rpc GetRecommendations(GetRecommendationsRequest) returns (GetRecommendationsResponse);
  
  // Load story metadata (called at startup or when stories are added)
  rpc LoadStories(StoryBatch) returns (LoadResponse);
}

// ===== EVENT MESSAGES (Fire-and-forget) =====

message UserAnsweredQuestionRequest {
  string user_id = 1;
  string story_id = 2;
  int32 response = 3;  // 1-5 (changed from V1's "score" 1-10)
  google.protobuf.Timestamp timestamp = 4;
  int32 question_number = 5;  // 1-4
}

message UserReadStoryRequest {
  string user_id = 1;
  string story_id = 2;
  int32 read_percent = 3;  // 0-100 (≥50% = viewed, 100% = completed)
  google.protobuf.Timestamp timestamp = 4;
}

message UserBookmarkedStoryRequest {
  string user_id = 1;
  string story_id = 2;
  google.protobuf.Timestamp timestamp = 3;
}

message UserViewedStoryRequest {
  string user_id = 1;
  string story_id = 2;
  google.protobuf.Timestamp timestamp = 3;
}

message UserSearchedTagRequest {
  string user_id = 1;
  string tag = 2;
  google.protobuf.Timestamp timestamp = 3;
}

// ===== RECOMMENDATION MESSAGES =====

message GetRecommendationsRequest {
  string user_id = 1;
  google.protobuf.Timestamp timestamp = 2;
  
  // Optional context for topical boosting
  repeated string boost_tags = 3;
  bool prefer_new_stories = 4;
  
  // Number of recommendations (default: 6)
  int32 count = 5;
}

message GetRecommendationsResponse {
  repeated Recommendation recommendations = 1;
}

message Recommendation {
  string story_id = 1;
  string title = 2;
  repeated string tags = 3;
  float score = 4;
  string method = 5;  // "content", "collaborative", "topical", "wildcard"
  int32 slot_position = 6;  // 0-5 (or 0-N)
  float avg_connectedness = 7;  // Average connectedness score (1-5), optional
}

// ===== STORY LOADING =====

message StoryBatch {
  repeated Story stories = 1;
}

message Story {
  string story_id = 1;
  string title = 2;
  repeated string tags = 3;
}

message LoadResponse {
  bool success = 1;
  int32 stories_loaded = 2;
  repeated string available_tags = 3;
}
```

## Event Types

### 1. UserViewedStory
Triggered when user opens a story.

```protobuf
message UserViewedStoryRequest {
  string user_id = 1;
  string story_id = 2;
  google.protobuf.Timestamp timestamp = 3;
}
```

**Example:**
```json
{
  "user_id": "user_123",
  "story_id": "story1",
  "timestamp": "2024-01-01T10:30:00Z"
}
```

### 2. UserReadStory
Triggered when user leaves a story (automatically tracked by scroll position or explicit completion).

```protobuf
message UserReadStoryRequest {
  string user_id = 1;
  string story_id = 2;
  int32 read_percent = 3;  // 0-100
  google.protobuf.Timestamp timestamp = 4;
}
```

**Example:**
```json
{
  "user_id": "user_123",
  "story_id": "story1",
  "read_percent": 100,
  "timestamp": "2024-01-01T10:35:00Z"
}
```

**Completion Thresholds:**
- 100% = Fully completed (strong positive signal)
- 50-99% = Partial read (moderate positive signal)
- 0-49% = Abandoned (weak/neutral signal)

### 3. UserAnsweredQuestion
Triggered when user answers a post-reading question.

```protobuf
message UserAnsweredQuestionRequest {
  string user_id = 1;
  string story_id = 2;
  int32 response = 3;  // 1-5
  google.protobuf.Timestamp timestamp = 4;
  int32 question_number = 5;  // 1-4
}
```

**Example:**
```json
{
  "user_id": "user_123",
  "story_id": "story1",
  "response": 5,
  "timestamp": "2024-01-01T10:36:00Z",
  "question_number": 1
}
```

**Questions:**
1. **Connectedness** (REQUIRED): "How connected did you feel to this story?" (1-5)
   - Primary signal for recommendations
   - 4-5 = high connectedness (find similar stories)
   - 1-2 = low connectedness (avoid similar stories)

2. **Emotional Impact** (OPTIONAL): "How emotionally impactful was this story?" (1-5)

3. **Would Recommend** (OPTIONAL): "Would you recommend this story to others?" (1-5)

4. **Thought-Provoking** (OPTIONAL): "How thought-provoking was this story?" (1-5)

**Note:** Question 1 may not be answered if user closes app. Handle missing responses gracefully.

### 4. UserBookmarkedStory
Triggered when user bookmarks a story (save for later).

```protobuf
message UserBookmarkedStoryRequest {
  string user_id = 1;
  string story_id = 2;
  google.protobuf.Timestamp timestamp = 3;
}
```

**Example:**
```json
{
  "user_id": "user_123",
  "story_id": "story1",
  "timestamp": "2024-01-01T10:37:00Z"
}
```

**Signal Strength:** Moderate preference (weaker than high connectedness, stronger than just viewing)

### 5. UserSearchedTag
Triggered when user browses by tag.

```protobuf
message UserSearchedTagRequest {
  string user_id = 1;
  string tag = 2;
  google.protobuf.Timestamp timestamp = 3;
}
```

**Example:**
```json
{
  "user_id": "user_123",
  "tag": "ancient",
  "timestamp": "2024-01-01T10:38:00Z"
}
```

## Recommendation Methods

The system uses configurable recommendation slots (default: 6 total):

| Method | Count | Description |
|--------|-------|-------------|
| **content** | 2 | Stories with similar tags to high-connectedness stories |
| **collaborative** | 2 | Stories that similar users connected with |
| **topical** | 1 | Boosted by context (new stories, specific tags) |
| **wildcard** | 1 | Exploration/serendipity (currently random from unseen) |

**Additional methods available:**
- **sequence**: Good follow-ups to last completed story (can be enabled)

### Method Configuration

The recommendation mix is configurable via the `recommendation_config` parameter:

```python
recommendation_config = {
    'content': 2,
    'collaborative': 2,
    'topical': 1,
    'wildcard': 1
}
```

This can be adjusted based on testing and user feedback.

## Ignore Decay Mechanism

**Problem:** Users may self-select away from triggering content by not selecting recommended stories.

**Solution:** Track how many times each story is recommended but not selected (ignore count). After a threshold, apply exponential decay to that story's recommendation score.

**Parameters (configurable):**
- `ignore_threshold`: Start decay after N ignores (default: 3)
- `ignore_decay_rate`: Reduce score by X% per additional ignore (default: 20%)

**Example:**
- Story recommended 5 times, selected 0 times
- Ignore count = 5
- Excess ignores = 5 - 3 = 2
- Decay factor = (1 - 0.2)^2 = 0.64
- Recommendation score multiplied by 0.64

**Reset:** Ignore count resets to 0 when user selects the story.

## Story Structure

Stories no longer have a single "theme" field. Instead, each story has one or more **tags**.

```python
Story(
    story_id="story1",
    title="The Alfred Jewel",
    tags=["ancient", "mysterious", "royal", "craftsmanship"]
)
```

**Tags are:**
- Dynamically loaded from story metadata
- Used for content-based filtering
- Browseable in the UI
- Expected to be mostly fixed but can change as content is added

## State Management

The system provides **two modes** of state persistence for different purposes:

### 1. Operational Checkpoints (Fault Tolerance)

**Purpose:** Keep service running after restarts, enable collaborative filtering  
**Frequency:** Every few minutes  
**Contents:** User profiles + story catalog (lightweight)  
**Format:** Optimized for fast loading

**HTTP Endpoint:**
```python
GET /checkpoint
```

Returns operational state (excludes analytical data like recommendation provenance):

```json
{
  "users": { ... },     // User profiles with preferences
  "stories": { ... },   // Story catalog
  "config": { ... },    // System configuration
  "checkpoint_metadata": {
    "checkpoint_timestamp": "2024-01-15T10:30:00",
    "total_users": 150,
    "total_stories": 25
  }
}
```

**Save to File:**
```python
POST /save_checkpoint
```

Returns: `{"success": true, "filepath": "checkpoints/checkpoint_20240115_103000.json"}`

**Usage in Production:**
```python
# In your service (every 5 minutes)
import schedule

def save_checkpoint():
    response = requests.post('http://localhost:5000/save_checkpoint')
    print(f"Checkpoint saved: {response.json()['filepath']}")

schedule.every(5).minutes.do(save_checkpoint)
```

### 2. Analytical Exports (Testing & Evaluation)

**Purpose:** Understand recommendation decisions, test logic, evaluate performance  
**Frequency:** Daily or on-demand  
**Contents:** Everything including recommendation provenance, sequences, events  
**Format:** Pretty-printed JSON for readability

**Full Analytical Export:**
```python
GET /export_state
```

Returns complete system state with all analytical data:

```json
{
  "stories": { ... },
  "users": {
    "user_123": {
      "viewed_stories": { ... },
      "story_connectedness": { ... },
      "recommendations_shown": [  // Provenance for analysis
        {
          "story_id": "story1",
          "method": "content",
          "slot_position": 0,
          "timestamp": "2024-01-15T10:30:00",
          "selected": true
        }
      ],
      "question_responses": [ ... ],
      "story_sequences": [ ... ]
    }
  },
  "events": [ ... ],
  "story_transitions": [ ... ],
  "export_metadata": {
    "export_timestamp": "2024-01-15T10:30:00",
    "export_mode": "full"
  }
}
```

**Daily Export (Date-Filtered):**
```python
GET /export_daily/2024-01-15
```

Returns analytical state with events filtered to specific date:

```json
{
  "stories": { ... },     // Full catalog
  "users": { ... },       // Full profiles (cumulative)
  "events": [ ... ],      // ONLY events from 2024-01-15
  "export_metadata": {
    "export_timestamp": "2024-01-16T00:00:00",
    "export_mode": "full",
    "start_date": "2024-01-15T00:00:00",
    "end_date": "2024-01-16T00:00:00"
  }
}
```

**Save Analytical Export:**
```python
POST /save_analytical_export
Content-Type: application/json

{
  "start_date": "2024-01-15T00:00:00",  // Optional
  "end_date": "2024-01-16T00:00:00"     // Optional
}
```

Returns: `{"success": true, "filepath": "exports/export_20240115.json"}`

**Usage for Daily Analysis:**
```python
# Run daily at midnight
from datetime import datetime, timedelta

def export_yesterday():
    yesterday = datetime.now() - timedelta(days=1)
    response = requests.post('http://localhost:5000/save_analytical_export', json={
        'start_date': yesterday.replace(hour=0, minute=0).isoformat(),
        'end_date': yesterday.replace(hour=23, minute=59).isoformat()
    })
    print(f"Daily export saved: {response.json()['filepath']}")
```

### Comparison Table

| Feature | Operational Checkpoint | Analytical Export |
|---------|----------------------|-------------------|
| **Frequency** | Every few minutes | Daily or on-demand |
| **Size** | Small (~100KB for 100 users) | Large (~10MB for 100 users) |
| **Events** | Not included | Full event history |
| **Recommendation Provenance** | Not included | Full details (method, slot, selected) |
| **Question Responses** | Not included | All responses |
| **Sequences** | Not included | Full transition graphs |
| **Purpose** | Service continuity | Understanding & testing |
| **Format** | Compact JSON | Pretty-printed JSON |
| **Loading Speed** | Fast (<1 sec) | Slower (10+ sec) |

### Key State Components

**Recommendation Provenance** (Analytical only):
```json
{
  "story_id": "story1",
  "method": "content",
  "slot_position": 0,
  "timestamp": "2024-01-15T10:30:00",
  "selected": true
}
```

**Question Responses** (Analytical only):
```json
[
  ["story1", 1, 5, "2024-01-15T10:30:00"],  // [story_id, q_num, response, timestamp]
  ["story1", 2, 4, "2024-01-15T10:31:00"]
]
```

**Ignore Counts** (Both modes):
```json
{
  "story2": 3,
  "story5": 1
}
```

### Python Usage

**Operational checkpoint:**
```python
# Save checkpoint
filepath = recommender.save_operational_checkpoint()

# Or specify path
filepath = recommender.save_operational_checkpoint("checkpoints/latest.json")
```

**Analytical export:**
```python
# Full export
filepath = recommender.export_analytical_state()

# Daily export with date filter
from datetime import datetime, timedelta
today = datetime.now().replace(hour=0, minute=0, second=0)
tomorrow = today + timedelta(days=1)

filepath = recommender.export_analytical_state(
    filepath=f"exports/daily_{today.strftime('%Y%m%d')}.json",
    start_date=today,
    end_date=tomorrow
)
```

## Testing the Mock Recommender

### 1. Create Mock Service (C#)

Implement the gRPC interface with simple logic:

```csharp
public class MockRecommenderService : StoryRecommender.StoryRecommenderBase
{
    public override Task<RecommendationResponse> GetRecommendations(
        RecommendationRequest request, ServerCallContext context)
    {
        // Return random 6 stories with mock methods
        var response = new RecommendationResponse();
        
        for (int i = 0; i < 6; i++)
        {
            response.Recommendations.Add(new Recommendation
            {
                StoryId = $"story{i+1}",
                Title = $"Mock Story {i+1}",
                Tags = { "ancient", "mysterious" },
                Score = Random.Shared.NextSingle(),
                Method = GetMethodForSlot(i),
                SlotPosition = i
            });
        }
        
        return Task.FromResult(response);
    }
    
    private string GetMethodForSlot(int slot)
    {
        return slot switch
        {
            0 or 1 => "content",
            2 or 3 => "collaborative",
            4 => "topical",
            5 => "wildcard",
            _ => "content"
        };
    }
    
    public override Task<EventResponse> RecordEvent(
        AnalyticsEvent request, ServerCallContext context)
    {
        // Just log the event
        Console.WriteLine($"Event: {request.EventType} for {request.StoryId}");
        
        return Task.FromResult(new EventResponse
        {
            Success = true,
            Message = "Event recorded"
        });
    }
}
```

### 2. Test Scenarios

**Scenario 1: New User**
```
1. GetRecommendations(user_id="new_user")
   → Should return 6 stories (mostly wildcard/topical since no history)

2. RecordEvent(VIEW, story_id="story1")

3. RecordEvent(STORY_PROGRESS, story_id="story1", completion_pct=100)

4. RecordEvent(QUESTION_RESPONSE, story_id="story1", q_num=1, response=5)
   → High connectedness

5. GetRecommendations(user_id="new_user")
   → Should return stories with tags similar to story1
```

**Scenario 2: Repeated Ignores**
```
1. GetRecommendations(user_id="user123")
   → Returns story2 in slot 0

2. RecordEvent(VIEW, story_id="story5")  // Selected different story

3. GetRecommendations(user_id="user123")
   → Returns story2 again (ignore_count=1)

4. RecordEvent(VIEW, story_id="story3")  // Ignored again

5. GetRecommendations(user_id="user123")
   → Returns story2 again (ignore_count=2)

6-7. Repeat 2 more times
   → ignore_count=4, decay starts (score × 0.8)

8. GetRecommendations(user_id="user123")
   → story2 less likely to appear
```

**Scenario 3: Tag Browsing**
```
1. RecordEvent(SEARCH, search_tag="ancient")
   → Records interest in "ancient" tag

2. GetRecommendations(user_id="user123")
   → Topical slot may boost "ancient" stories
```

## Implementation Notes

### Time Decay

The system uses exponential decay with configurable half-lives:

- **Event half-life**: 30 days (general preferences)
- **Connectedness half-life**: 14 days (connectedness signals decay faster)

**Formula:** `decay_factor = 0.5 ^ (days_ago / half_life_days)`

### Sequence Tracking

Stories completed within 24 hours (configurable) are considered a sequence. The system learns:
- Which specific story pairs lead to high connectedness
- Which tag transitions work well
- Personal vs. global sequence patterns

### Tag Similarity

Stories are compared using Jaccard similarity on their tag sets:

```
similarity = |tags1 ∩ tags2| / |tags1 ∪ tags2|
```

## Configuration Parameters

All configurable in `StoryRecommender.__init__()`:

```python
StoryRecommender(
    event_half_life_days=30.0,           # General event decay
    connectedness_half_life_days=14.0,   # Connectedness decay (faster)
    transition_window_minutes=1440.0,    # 24 hours for sequences
    recommendation_config={               # Slot allocation
        'content': 2,
        'collaborative': 2,
        'topical': 1,
        'wildcard': 1
    }
)

# Set ignore decay parameters
recommender.ignore_threshold = 3         # Start decay after 3 ignores
recommender.ignore_decay_rate = 0.2      # 20% decay per additional ignore
```

## Future Enhancements

**Under Discussion:**
- Serendipity logic for wildcard slot (instead of pure random)
- Mood tracking (for evaluation, not recommendations)
- Recommendation position consistency (repeat stories in same slot)
- Path-based recommendations (3+ story sequences)
- Tag diversity in recommendation sets

## Files

```
reco-v2/
├── recommender.py          # Core recommendation logic
├── app.py                  # Flask demo server
├── templates/              # HTML templates for demo UI
│   ├── base.html
│   ├── index.html
│   ├── browse_tag.html
│   ├── recommendations.html
│   ├── story.html
│   ├── story_questions.html
│   └── insights.html
├── recommender.proto       # gRPC interface definition (create from above)
└── README.md               # This file
```

## Running the Demo

```bash
# Install dependencies
pip install flask numpy

# Run the demo server
python app.py

# Visit in browser
http://localhost:5000
```

The demo provides:
- Tag-based browsing
- Personalized recommendations with method labels
- Reading progress tracking
- Post-reading questions
- Bookmark functionality
- Insights dashboard showing method performance
- State export endpoints

## Questions for Team Discussion

1. **Recommendation slot allocation**: Is 2-2-1-1 the right mix, or should we adjust based on testing?

2. **Ignore decay parameters**: Are threshold=3 and rate=20% appropriate for user safety?

3. **Wildcard strategy**: Should we implement serendipity logic, or keep pure random exploration?

4. **State export frequency**: Daily export sufficient, or need more frequent backups?

5. **Question 1 (connectedness)**: What if user doesn't answer? Current: treat as neutral, don't penalize.

6. **Tag stability**: How often will new tags be added? Affects caching strategy.

7. **Sequence window**: Is 24 hours the right cutoff for considering stories as a sequence?

---

**Document Version:** 1.0  
**Last Updated:** 2024-04-26  
**Status:** Ready for Mock Implementation

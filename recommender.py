import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import json

class Story:
    """Represents a story with metadata"""
    def __init__(self, story_id: str, title: str, tags: List[str]):
        self.id = story_id
        self.title = title
        self.tags = tags  # List of tags instead of single theme
        
        # Connectedness tracking
        self.connectedness_scores = []  # List of (score, timestamp) tuples
        self.avg_connectedness = None
        
        # Sequence effects - what works well AFTER this story
        self.best_next_stories = {}  # story_id -> avg_connectedness
        self.best_next_tags = {}   # tag -> avg_connectedness
        
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'tags': self.tags,
            'connectedness_scores': [(score, ts.isoformat()) for score, ts in self.connectedness_scores],
            'avg_connectedness': self.avg_connectedness,
            'best_next_stories': self.best_next_stories,
            'best_next_tags': self.best_next_tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Story':
        story = cls(data['id'], data['title'], data['tags'])
        story.connectedness_scores = [
            (score, datetime.fromisoformat(ts)) 
            for score, ts in data.get('connectedness_scores', [])
        ]
        story.avg_connectedness = data.get('avg_connectedness')
        story.best_next_stories = data.get('best_next_stories', {})
        story.best_next_tags = data.get('best_next_tags', {})
        return story


class StoryTransition:
    """Represents a transition from one story to another"""
    def __init__(self, from_story_id: str, to_story_id: str, 
                 user_id: str, timestamp: datetime,
                 connectedness_before: Optional[float] = None,
                 connectedness_after: Optional[float] = None,
                 time_between_minutes: float = 0.0):
        self.from_story_id = from_story_id
        self.to_story_id = to_story_id
        self.user_id = user_id
        self.timestamp = timestamp
        self.connectedness_before = connectedness_before
        self.connectedness_after = connectedness_after
        self.time_between_minutes = time_between_minutes
        
        # Computed
        self.connectedness_delta = None
        if connectedness_before is not None and connectedness_after is not None:
            self.connectedness_delta = connectedness_after - connectedness_before
    
    def to_dict(self) -> Dict:
        return {
            'from_story_id': self.from_story_id,
            'to_story_id': self.to_story_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'connectedness_before': self.connectedness_before,
            'connectedness_after': self.connectedness_after,
            'time_between_minutes': self.time_between_minutes,
            'connectedness_delta': self.connectedness_delta
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StoryTransition':
        transition = cls(
            data['from_story_id'],
            data['to_story_id'],
            data['user_id'],
            datetime.fromisoformat(data['timestamp']),
            data.get('connectedness_before'),
            data.get('connectedness_after'),
            data['time_between_minutes']
        )
        transition.connectedness_delta = data.get('connectedness_delta')
        return transition


class RecommendationRecord:
    """Records a recommendation that was shown to a user"""
    def __init__(self, story_id: str, method: str, slot_position: int, timestamp: datetime):
        self.story_id = story_id
        self.method = method  # 'content', 'collaborative', 'topical', 'wildcard', 'sequence'
        self.slot_position = slot_position  # 0-5 (or 0-N for configurable slots)
        self.timestamp = timestamp
        self.selected = False  # Updated when user views the story
        
    def to_dict(self) -> Dict:
        return {
            'story_id': self.story_id,
            'method': self.method,
            'slot_position': self.slot_position,
            'timestamp': self.timestamp.isoformat(),
            'selected': self.selected
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RecommendationRecord':
        rec = cls(
            data['story_id'],
            data['method'],
            data['slot_position'],
            datetime.fromisoformat(data['timestamp'])
        )
        rec.selected = data.get('selected', False)
        return rec


class AnalyticsEvent:
    """Represents a user interaction event"""
    def __init__(self, user_id: str, event_type: str, timestamp: datetime, **kwargs):
        self.user_id = user_id
        self.event_type = event_type
        # Event types: 'view', 'story_progress', 'question_response', 'bookmark', 'search'
        self.timestamp = timestamp
        self.data = kwargs
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalyticsEvent':
        return cls(
            data['user_id'],
            data['event_type'],
            datetime.fromisoformat(data['timestamp']),
            **data['data']
        )


class UserProfile:
    """Represents a user's preferences and history"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.viewed_stories = {}  # story_id -> timestamp
        self.story_progress = {}  # story_id -> (completion_percentage, timestamp)
        self.bookmarked_stories = {}  # story_id -> timestamp
        
        # Tag preferences (based on completions and connectedness)
        self.tag_interactions = defaultdict(list)  # tag -> [(score, timestamp)]
        
        # Connectedness tracking
        self.story_connectedness = {}  # story_id -> (connectedness_score, timestamp)
        
        # Question responses
        self.question_responses = []  # List of (story_id, question_number, response, timestamp)
        
        # Recent interactions for temporal filtering
        self.recent_story_views = []  # Last N viewed stories with timestamps
        
        # Recommendation tracking
        self.recommendations_shown = []  # List of RecommendationRecord
        self.story_ignore_count = defaultdict(int)  # story_id -> times shown but not selected
        
        # Sequential preferences
        self.story_sequences = []  # List of StoryTransition objects
        self.preferred_transitions = defaultdict(list)  # from_story_id -> [(to_story_id, connectedness_delta, timestamp)]
        self.tag_transition_preferences = defaultdict(lambda: defaultdict(list))  # from_tag -> to_tag -> [connectedness_deltas]
        
        # Last completed story (for next-story recommendations)
        self.last_completed_story = None
        self.last_completed_timestamp = None
        
    def get_avoided_tags(self, threshold: float = -1.0, current_time: datetime = None) -> List[str]:
        """Get tags the user seems to avoid, with time decay"""
        current_time = current_time or datetime.now()
        tag_scores = self._get_decayed_tag_scores(current_time)
        return [tag for tag, score in tag_scores.items() if score < threshold]
    
    def get_preferred_tags(self, threshold: float = 1.0, current_time: datetime = None) -> List[str]:
        """Get tags the user prefers, with time decay"""
        current_time = current_time or datetime.now()
        tag_scores = self._get_decayed_tag_scores(current_time)
        return [tag for tag, score in tag_scores.items() if score > threshold]
    
    def _get_decayed_tag_scores(self, current_time: datetime, half_life_days: float = 30.0) -> Dict[str, float]:
        """Calculate tag scores with exponential time decay"""
        tag_scores = defaultdict(float)
        
        for tag, interactions in self.tag_interactions.items():
            total_score = 0.0
            for score, timestamp in interactions:
                days_ago = (current_time - timestamp).total_seconds() / 86400
                decay_factor = 0.5 ** (days_ago / half_life_days)
                total_score += score * decay_factor
            tag_scores[tag] = total_score
        
        return tag_scores
    
    def get_recent_story_path(self, n: int = 3) -> List[str]:
        """Get the last N completed stories as a path"""
        # Get stories with 100% completion
        completed = [(sid, ts) for sid, (pct, ts) in self.story_progress.items() if pct >= 100]
        sorted_completions = sorted(completed, key=lambda x: x[1])
        return [story_id for story_id, _ in sorted_completions[-n:]]


class StoryRecommender:
    """Main recommender system with configurable recommendation slots"""
    
    def __init__(self, 
                 event_half_life_days: float = 30.0,
                 connectedness_half_life_days: float = 14.0,
                 transition_window_minutes: float = 1440.0,
                 recommendation_config: Dict = None):
        """
        Args:
            event_half_life_days: Half-life for general event decay
            connectedness_half_life_days: Half-life for connectedness-related events
            transition_window_minutes: Max time between stories to count as sequence
            recommendation_config: Dict specifying recommendation slot allocation
                Example: {'content': 2, 'collaborative': 2, 'topical': 1, 'wildcard': 1}
        """
        self.stories: Dict[str, Story] = {}
        self.users: Dict[str, UserProfile] = {}
        self.events: List[AnalyticsEvent] = []
        
        # Decay parameters
        self.event_half_life_days = event_half_life_days
        self.connectedness_half_life_days = connectedness_half_life_days
        self.transition_window_minutes = transition_window_minutes
        
        # Recommendation configuration
        if recommendation_config is None:
            recommendation_config = {
                'content': 2,
                'collaborative': 2,
                'topical': 1,
                'wildcard': 1
            }
        self.recommendation_config = recommendation_config
        self.total_recommendations = sum(recommendation_config.values())
        
        # Ignore decay parameters (configurable)
        self.ignore_threshold = 3  # Start applying decay after 3 ignores
        self.ignore_decay_rate = 0.2  # Reduce score by 20% per additional ignore
        
        # Global transition tracking
        self.story_transitions = []  # List of StoryTransition objects
        self.global_transition_graph = defaultdict(lambda: defaultdict(list))
        
        # Tag tracking
        self.available_tags = set()  # Dynamically populated from stories
        
        # Caches
        self._story_similarity_cache = {}
        self._tag_to_stories = defaultdict(list)
        
    def add_story(self, story_id: str, title: str, tags: List[str]):
        """Add a new story to the catalog"""
        story = Story(story_id, title, tags)
        self.stories[story_id] = story
        
        # Update available tags
        for tag in tags:
            self.available_tags.add(tag)
            self._tag_to_stories[tag].append(story_id)
        
        # Invalidate similarity cache
        self._story_similarity_cache = {}
    
    def add_event(self, event: AnalyticsEvent):
        """Process an analytics event"""
        self.events.append(event)
        user_id = event.user_id
        
        # Ensure user profile exists
        if user_id not in self.users:
            self.users[user_id] = UserProfile(user_id)
        
        user = self.users[user_id]
        
        # Process based on event type
        if event.event_type == 'view':
            story_id = event.data['story_id']
            user.viewed_stories[story_id] = event.timestamp
            user.recent_story_views.append((event.timestamp, story_id))
            
            # Update tag exposure (neutral at first)
            if story_id in self.stories:
                for tag in self.stories[story_id].tags:
                    user.tag_interactions[tag].append((0.1, event.timestamp))
            
            # Mark any recommendations for this story as selected
            for rec in user.recommendations_shown:
                if rec.story_id == story_id and not rec.selected:
                    rec.selected = True
                    # Reset ignore count since they selected it
                    user.story_ignore_count[story_id] = 0
                    
        elif event.event_type == 'story_progress':
            story_id = event.data['story_id']
            completion_pct = event.data['completion_percentage']
            user.story_progress[story_id] = (completion_pct, event.timestamp)
            
            # If completed (100%), treat as strong positive signal
            if completion_pct >= 100:
                if story_id in self.stories:
                    for tag in self.stories[story_id].tags:
                        user.tag_interactions[tag].append((1.0, event.timestamp))
                
                # Check for story transition (sequence)
                if user.last_completed_story and user.last_completed_timestamp:
                    time_diff = (event.timestamp - user.last_completed_timestamp).total_seconds() / 60.0
                    
                    if time_diff <= self.transition_window_minutes:
                        self._record_story_transition(
                            user,
                            user.last_completed_story,
                            story_id,
                            event.timestamp,
                            time_diff
                        )
                
                # Update last completed
                user.last_completed_story = story_id
                user.last_completed_timestamp = event.timestamp
            
            # Partial completion is still a moderate signal
            elif completion_pct >= 50:
                if story_id in self.stories:
                    for tag in self.stories[story_id].tags:
                        user.tag_interactions[tag].append((0.5, event.timestamp))
                        
        elif event.event_type == 'question_response':
            story_id = event.data['story_id']
            question_number = event.data['question_number']
            response = event.data['response']  # 1-5
            
            # Store all question responses
            user.question_responses.append((story_id, question_number, response, event.timestamp))
            
            # Question 1 (compulsory) is the connectedness score
            if question_number == 1:
                user.story_connectedness[story_id] = (response, event.timestamp)
                
                # Update story's connectedness stats
                if story_id in self.stories:
                    self.stories[story_id].connectedness_scores.append((response, event.timestamp))
                    self._update_story_connectedness_stats(story_id)
                    
                    # Adjust tag preference based on connectedness
                    for tag in self.stories[story_id].tags:
                        # High connectedness (4-5) = strong positive, Low (1-2) = negative
                        tag_score = (response - 3) * 0.5  # Range: -1.0 to +1.0
                        user.tag_interactions[tag].append((tag_score, event.timestamp))
                
                # Update transition with connectedness if applicable
                self._update_recent_transition_connectedness(user, story_id, response)
                        
        elif event.event_type == 'bookmark':
            story_id = event.data['story_id']
            user.bookmarked_stories[story_id] = event.timestamp
            
            # Moderate positive signal
            if story_id in self.stories:
                for tag in self.stories[story_id].tags:
                    user.tag_interactions[tag].append((0.7, event.timestamp))
                    
        elif event.event_type == 'search':
            # Track tag searches
            if 'tag' in event.data:
                tag = event.data['tag']
                user.tag_interactions[tag].append((0.3, event.timestamp))
    
    def _record_story_transition(self, user: UserProfile, from_story_id: str, 
                                 to_story_id: str, timestamp: datetime, 
                                 time_between_minutes: float):
        """Record a story-to-story transition"""
        # Get connectedness scores if available
        connectedness_before = None
        connectedness_after = None
        
        if from_story_id in user.story_connectedness:
            connectedness_before, _ = user.story_connectedness[from_story_id]
        
        # Current connectedness will be updated when question is answered
        
        # Create transition
        transition = StoryTransition(
            from_story_id,
            to_story_id,
            user.user_id,
            timestamp,
            connectedness_before,
            connectedness_after,
            time_between_minutes
        )
        
        # Store in user profile
        user.story_sequences.append(transition)
        user.preferred_transitions[from_story_id].append(
            (to_story_id, transition.connectedness_delta, timestamp)
        )
        
        # Store tag transitions
        if from_story_id in self.stories and to_story_id in self.stories:
            from_tags = self.stories[from_story_id].tags
            to_tags = self.stories[to_story_id].tags
            
            for from_tag in from_tags:
                for to_tag in to_tags:
                    if transition.connectedness_delta is not None:
                        user.tag_transition_preferences[from_tag][to_tag].append(
                            (transition.connectedness_delta, timestamp)
                        )
        
        # Store globally
        self.story_transitions.append(transition)
        self.global_transition_graph[from_story_id][to_story_id].append(transition)
        
        # Update story's "best next" statistics
        self._update_story_transition_stats(from_story_id)
    
    def _update_recent_transition_connectedness(self, user: UserProfile, story_id: str, 
                                               connectedness: float):
        """Update the most recent transition with connectedness information"""
        if not user.story_sequences:
            return
        
        last_transition = user.story_sequences[-1]
        if last_transition.to_story_id == story_id and last_transition.connectedness_after is None:
            last_transition.connectedness_after = connectedness
            if last_transition.connectedness_before is not None:
                last_transition.connectedness_delta = connectedness - last_transition.connectedness_before
                
                # Update the corresponding entry in preferred_transitions
                from_story = last_transition.from_story_id
                for i, (to_id, delta, ts) in enumerate(user.preferred_transitions[from_story]):
                    if to_id == story_id and delta is None and ts == last_transition.timestamp:
                        user.preferred_transitions[from_story][i] = (
                            to_id, last_transition.connectedness_delta, ts
                        )
                        break
                
                # Update story transition stats
                self._update_story_transition_stats(from_story)
    
    def _update_story_transition_stats(self, from_story_id: str):
        """Update statistics about what stories work well after this one"""
        if from_story_id not in self.stories:
            return
        
        from_story = self.stories[from_story_id]
        current_time = datetime.now()
        
        # Collect all transitions from this story
        transitions = self.global_transition_graph[from_story_id]
        
        # Calculate average connectedness for each next story (with time decay)
        next_story_effects = defaultdict(list)
        next_tag_effects = defaultdict(list)
        
        for to_story_id, transition_list in transitions.items():
            for transition in transition_list:
                if transition.connectedness_after is None:
                    continue
                
                # Apply time decay
                days_ago = (current_time - transition.timestamp).total_seconds() / 86400
                decay_factor = 0.5 ** (days_ago / self.connectedness_half_life_days)
                
                weighted_connectedness = transition.connectedness_after * decay_factor
                next_story_effects[to_story_id].append(weighted_connectedness)
                
                # Also track by tag
                if to_story_id in self.stories:
                    for to_tag in self.stories[to_story_id].tags:
                        next_tag_effects[to_tag].append(weighted_connectedness)
        
        # Store averages
        from_story.best_next_stories = {
            story_id: np.mean(scores) 
            for story_id, scores in next_story_effects.items()
            if scores
        }
        
        from_story.best_next_tags = {
            tag: np.mean(scores)
            for tag, scores in next_tag_effects.items()
            if scores
        }
    
    def _update_story_connectedness_stats(self, story_id: str):
        """Update story's average connectedness statistics with time decay"""
        story = self.stories[story_id]
        if not story.connectedness_scores:
            return
        
        current_time = datetime.now()
        
        # Calculate weighted average with time decay
        weighted_scores = []
        total_weight = 0.0
        
        for score, timestamp in story.connectedness_scores:
            # Apply exponential decay
            days_ago = (current_time - timestamp).total_seconds() / 86400
            decay_factor = 0.5 ** (days_ago / self.connectedness_half_life_days)
            
            weighted_scores.append(score * decay_factor)
            total_weight += decay_factor
        
        if total_weight > 0:
            story.avg_connectedness = sum(weighted_scores) / total_weight
    
    def get_recommendations(self, 
                           user_id: str, 
                           context: Dict = None,
                           n_recommendations: int = None) -> List[Tuple[str, float, str, int]]:
        """
        Get personalized story recommendations with method tracking.
        
        Args:
            user_id: User requesting recommendations
            context: Optional context dict
            n_recommendations: Number of recommendations (defaults to config total)
            
        Returns:
            List of (story_id, score, method, slot_position) tuples
        """
        context = context or {}
        current_time = context.get('current_time', datetime.now())
        
        if n_recommendations is None:
            n_recommendations = self.total_recommendations
        
        # Ensure user exists
        if user_id not in self.users:
            self.users[user_id] = UserProfile(user_id)
        
        user = self.users[user_id]
        
        # Get recommendations for each slot type
        recommendations = []
        slot_position = 0
        
        for method, count in self.recommendation_config.items():
            method_recs = self._get_recommendations_by_method(
                user, method, count, current_time, context
            )
            
            for story_id, score in method_recs:
                recommendations.append((story_id, score, method, slot_position))
                slot_position += 1
        
        # Record these recommendations
        for story_id, score, method, slot in recommendations:
            rec_record = RecommendationRecord(story_id, method, slot, current_time)
            user.recommendations_shown.append(rec_record)
        
        # Update ignore counts for stories that were shown before but not selected
        self._update_ignore_counts(user, [r[0] for r in recommendations])
        
        return recommendations
    
    def _get_recommendations_by_method(self, user: UserProfile, method: str, 
                                       count: int, current_time: datetime,
                                       context: Dict) -> List[Tuple[str, float]]:
        """Get recommendations using a specific method"""
        
        # Get all candidate stories (exclude recently viewed)
        recent_story_ids = [sid for _, sid in user.recent_story_views[-10:]]
        candidates = {
            sid: story for sid, story in self.stories.items()
            if sid not in recent_story_ids
        }
        
        if not candidates:
            return []
        
        if method == 'content':
            return self._content_based_recommendations(user, candidates, count, current_time)
        elif method == 'collaborative':
            return self._collaborative_recommendations(user, candidates, count, current_time)
        elif method == 'topical':
            return self._topical_recommendations(user, candidates, count, current_time, context)
        elif method == 'wildcard':
            return self._wildcard_recommendations(user, candidates, count, current_time)
        elif method == 'sequence':
            return self._sequence_recommendations(user, candidates, count, current_time)
        else:
            # Unknown method, return random
            return self._wildcard_recommendations(user, candidates, count, current_time)
    
    def _content_based_recommendations(self, user: UserProfile, candidates: Dict[str, Story],
                                       count: int, current_time: datetime) -> List[Tuple[str, float]]:
        """Recommend stories with tags similar to high-connectedness stories"""
        
        # Find user's high-connectedness stories
        high_connectedness_stories = [
            sid for sid, (score, ts) in user.story_connectedness.items()
            if score >= 4  # 4 or 5 = high connectedness
        ]
        
        if not high_connectedness_stories:
            # Fallback: use bookmarked or completed stories
            high_connectedness_stories = list(user.bookmarked_stories.keys())
            if not high_connectedness_stories:
                completed = [sid for sid, (pct, _) in user.story_progress.items() if pct >= 100]
                high_connectedness_stories = completed[:5]
        
        if not high_connectedness_stories:
            # No history, return random
            return self._wildcard_recommendations(user, candidates, count, current_time)
        
        # Score each candidate by tag similarity to high-connectedness stories
        scores = {}
        for candidate_id, candidate in candidates.items():
            total_similarity = 0.0
            for liked_id in high_connectedness_stories:
                if liked_id in self.stories:
                    similarity = self._story_similarity(candidate_id, liked_id)
                    total_similarity += similarity
            
            scores[candidate_id] = total_similarity / len(high_connectedness_stories)
        
        # Apply ignore decay
        scores = self._apply_ignore_decay(user, scores)
        
        # Return top N
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:count]
    
    def _collaborative_recommendations(self, user: UserProfile, candidates: Dict[str, Story],
                                       count: int, current_time: datetime) -> List[Tuple[str, float]]:
        """Recommend stories that similar users gave high connectedness scores"""
        
        # Find user's high-connectedness stories
        user_high_connected = {
            sid for sid, (score, _) in user.story_connectedness.items()
            if score >= 4
        }
        
        if not user_high_connected:
            # Fallback to completed stories
            user_high_connected = {
                sid for sid, (pct, _) in user.story_progress.items() if pct >= 100
            }
        
        if not user_high_connected:
            return self._wildcard_recommendations(user, candidates, count, current_time)
        
        # Find similar users (users who also gave high connectedness to same stories)
        similar_user_scores = []
        
        for other_user_id, other_user in self.users.items():
            if other_user_id == user.user_id:
                continue
            
            other_high_connected = {
                sid for sid, (score, _) in other_user.story_connectedness.items()
                if score >= 4
            }
            
            # Calculate Jaccard similarity
            intersection = len(user_high_connected & other_high_connected)
            union = len(user_high_connected | other_high_connected)
            
            if union == 0:
                continue
            
            similarity = intersection / union
            
            # Find candidates this similar user connected with
            for candidate_id in candidates.keys():
                if candidate_id in other_user.story_connectedness:
                    score, timestamp = other_user.story_connectedness[candidate_id]
                    if score >= 4:
                        # Weight by user similarity and time decay
                        days_ago = (current_time - timestamp).total_seconds() / 86400
                        decay_factor = 0.5 ** (days_ago / self.event_half_life_days)
                        
                        similar_user_scores.append((candidate_id, similarity * score * decay_factor))
        
        if not similar_user_scores:
            return self._wildcard_recommendations(user, candidates, count, current_time)
        
        # Aggregate scores by candidate
        candidate_scores = defaultdict(list)
        for candidate_id, score in similar_user_scores:
            candidate_scores[candidate_id].append(score)
        
        final_scores = {
            candidate_id: np.mean(scores)
            for candidate_id, scores in candidate_scores.items()
        }
        
        # Apply ignore decay
        final_scores = self._apply_ignore_decay(user, final_scores)
        
        sorted_recs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:count]
    
    def _topical_recommendations(self, user: UserProfile, candidates: Dict[str, Story],
                                 count: int, current_time: datetime, 
                                 context: Dict) -> List[Tuple[str, float]]:
        """
        Recommend stories based on topical boosting (e.g., new stories, specific tags).
        Context can specify: 'boost_tags', 'prefer_new', etc.
        """
        
        scores = {}
        
        # Check for boost tags in context
        boost_tags = context.get('boost_tags', [])
        prefer_new = context.get('prefer_new', True)
        
        for candidate_id, candidate in candidates.items():
            score = 0.0
            
            # Boost if story has boosted tags
            if boost_tags:
                matching_tags = set(candidate.tags) & set(boost_tags)
                score += len(matching_tags) * 2.0
            
            # Boost new stories (stories with few views)
            if prefer_new:
                total_views = sum(1 for u in self.users.values() if candidate_id in u.viewed_stories)
                # Higher score for fewer views
                newness_score = max(0, 5 - total_views) / 5.0
                score += newness_score * 1.5
            
            # Add base tag preference
            tag_scores = user._get_decayed_tag_scores(current_time)
            for tag in candidate.tags:
                score += tag_scores.get(tag, 0) * 0.5
            
            scores[candidate_id] = score
        
        # Apply ignore decay
        scores = self._apply_ignore_decay(user, scores)
        
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:count]
    
    def _wildcard_recommendations(self, user: UserProfile, candidates: Dict[str, Story],
                                  count: int, current_time: datetime) -> List[Tuple[str, float]]:
        """
        Recommend exploration stories (currently random from unseen).
        Future: could implement serendipity logic here.
        """
        
        # Get unseen stories
        unseen = {
            sid: story for sid, story in candidates.items()
            if sid not in user.viewed_stories
        }
        
        if not unseen:
            unseen = candidates  # All seen, just use all candidates
        
        # Currently: random selection
        # Future serendipity options:
        # - Stories from tags user hasn't explored
        # - Stories with diverse tags
        # - High avg_connectedness from other users
        
        import random
        selected = random.sample(list(unseen.keys()), min(count, len(unseen)))
        
        # Give them low but non-zero scores
        return [(sid, 0.1) for sid in selected]
    
    def _sequence_recommendations(self, user: UserProfile, candidates: Dict[str, Story],
                                  count: int, current_time: datetime) -> List[Tuple[str, float]]:
        """Recommend stories that work well after the last completed story"""
        
        if not user.last_completed_story or user.last_completed_story not in self.stories:
            return self._wildcard_recommendations(user, candidates, count, current_time)
        
        last_story = self.stories[user.last_completed_story]
        scores = {}
        
        for candidate_id, candidate in candidates.items():
            score = 0.0
            
            # Story-level patterns
            if candidate_id in last_story.best_next_stories:
                score += last_story.best_next_stories[candidate_id] * 2.0
            
            # Tag-level patterns
            for candidate_tag in candidate.tags:
                if candidate_tag in last_story.best_next_tags:
                    score += last_story.best_next_tags[candidate_tag] * 1.0
            
            scores[candidate_id] = score
        
        # Apply ignore decay
        scores = self._apply_ignore_decay(user, scores)
        
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:count]
    
    def _apply_ignore_decay(self, user: UserProfile, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply decay penalty to stories that have been repeatedly shown but not selected"""
        
        decayed_scores = {}
        for story_id, score in scores.items():
            ignore_count = user.story_ignore_count[story_id]
            
            if ignore_count >= self.ignore_threshold:
                # Apply exponential decay
                excess_ignores = ignore_count - self.ignore_threshold
                decay_factor = (1 - self.ignore_decay_rate) ** excess_ignores
                decayed_scores[story_id] = score * decay_factor
            else:
                decayed_scores[story_id] = score
        
        return decayed_scores
    
    def _update_ignore_counts(self, user: UserProfile, current_recommendations: List[str]):
        """Update ignore counts based on previous recommendations"""
        
        # Get stories from last recommendation set that weren't selected
        if len(user.recommendations_shown) < self.total_recommendations:
            return  # First recommendation set
        
        # Look at previous recommendation set
        prev_recs_start = -(self.total_recommendations * 2)
        prev_recs_end = -self.total_recommendations
        prev_recs = user.recommendations_shown[prev_recs_start:prev_recs_end]
        
        for rec in prev_recs:
            if not rec.selected:
                user.story_ignore_count[rec.story_id] += 1
    
    def _story_similarity(self, story_id1: str, story_id2: str) -> float:
        """
        Calculate similarity between two stories based on tags.
        Uses caching for efficiency.
        """
        if story_id1 == story_id2:
            return 1.0
        
        # Check cache (bidirectional)
        cache_key = tuple(sorted([story_id1, story_id2]))
        if cache_key in self._story_similarity_cache:
            return self._story_similarity_cache[cache_key]
        
        story1 = self.stories.get(story_id1)
        story2 = self.stories.get(story_id2)
        
        if not story1 or not story2:
            return 0.0
        
        # Jaccard similarity on tags
        tags1 = set(story1.tags)
        tags2 = set(story2.tags)
        
        intersection = len(tags1 & tags2)
        union = len(tags1 | tags2)
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Cache result
        self._story_similarity_cache[cache_key] = similarity
        
        return similarity
    
    def get_sequence_insights(self, user_id: str = None) -> Dict:
        """Get insights about story sequences for analysis"""
        insights = {
            'global_transitions': {},
            'effective_tag_transitions': {}
        }
        
        # Global transition statistics
        for from_id, to_dict in self.global_transition_graph.items():
            if from_id not in self.stories:
                continue
            
            from_story = self.stories[from_id]
            insights['global_transitions'][from_story.title] = {
                'best_next': [
                    (self.stories[to_id].title, avg_connectedness)
                    for to_id, avg_connectedness in sorted(
                        from_story.best_next_stories.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                ],
                'best_next_tags': from_story.best_next_tags
            }
        
        # User-specific insights
        if user_id and user_id in self.users:
            user = self.users[user_id]
            insights['user_sequences'] = [
                {
                    'from': self.stories[t.from_story_id].title if t.from_story_id in self.stories else t.from_story_id,
                    'to': self.stories[t.to_story_id].title if t.to_story_id in self.stories else t.to_story_id,
                    'connectedness_delta': t.connectedness_delta,
                    'time_between_min': t.time_between_minutes
                }
                for t in user.story_sequences[-10:]
            ]
            
            insights['user_ignore_counts'] = dict(user.story_ignore_count)
        
        return insights
    
    # State management
    def save_state(self, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """
        Export system state as a dictionary.
        Optionally filter by date range for daily exports.
        """
        # Filter events by date range if specified
        events_to_export = self.events
        if start_date or end_date:
            events_to_export = [
                e for e in self.events
                if (not start_date or e.timestamp >= start_date) and
                   (not end_date or e.timestamp <= end_date)
            ]
        
        return {
            'stories': {sid: story.to_dict() for sid, story in self.stories.items()},
            'users': {
                uid: {
                    'user_id': user.user_id,
                    'viewed_stories': {sid: ts.isoformat() for sid, ts in user.viewed_stories.items()},
                    'story_progress': {
                        sid: (pct, ts.isoformat()) 
                        for sid, (pct, ts) in user.story_progress.items()
                    },
                    'bookmarked_stories': {sid: ts.isoformat() for sid, ts in user.bookmarked_stories.items()},
                    'tag_interactions': {
                        tag: [(score, ts.isoformat()) for score, ts in interactions]
                        for tag, interactions in user.tag_interactions.items()
                    },
                    'story_connectedness': {
                        sid: (score, ts.isoformat())
                        for sid, (score, ts) in user.story_connectedness.items()
                    },
                    'question_responses': [
                        (sid, qnum, resp, ts.isoformat())
                        for sid, qnum, resp, ts in user.question_responses
                    ],
                    'recent_story_views': [(ts.isoformat(), sid) for ts, sid in user.recent_story_views],
                    'recommendations_shown': [rec.to_dict() for rec in user.recommendations_shown],
                    'story_ignore_count': dict(user.story_ignore_count),
                    'story_sequences': [t.to_dict() for t in user.story_sequences],
                    'last_completed_story': user.last_completed_story,
                    'last_completed_timestamp': user.last_completed_timestamp.isoformat() if user.last_completed_timestamp else None
                }
                for uid, user in self.users.items()
            },
            'story_transitions': [t.to_dict() for t in self.story_transitions],
            'events': [event.to_dict() for event in events_to_export],
            'available_tags': list(self.available_tags),
            'config': {
                'event_half_life_days': self.event_half_life_days,
                'connectedness_half_life_days': self.connectedness_half_life_days,
                'transition_window_minutes': self.transition_window_minutes,
                'recommendation_config': self.recommendation_config,
                'ignore_threshold': self.ignore_threshold,
                'ignore_decay_rate': self.ignore_decay_rate
            },
            'export_metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None
            }
        }
    
    def load_state(self, state: Dict):
        """Load system state from a dictionary"""
        # Load config
        config = state.get('config', {})
        self.event_half_life_days = config.get('event_half_life_days', 30.0)
        self.connectedness_half_life_days = config.get('connectedness_half_life_days', 14.0)
        self.transition_window_minutes = config.get('transition_window_minutes', 1440.0)
        self.recommendation_config = config.get('recommendation_config', {
            'content': 2, 'collaborative': 2, 'topical': 1, 'wildcard': 1
        })
        self.total_recommendations = sum(self.recommendation_config.values())
        self.ignore_threshold = config.get('ignore_threshold', 3)
        self.ignore_decay_rate = config.get('ignore_decay_rate', 0.2)
        
        # Load stories
        self.stories = {
            sid: Story.from_dict(data)
            for sid, data in state.get('stories', {}).items()
        }
        
        # Load available tags
        self.available_tags = set(state.get('available_tags', []))
        
        # Rebuild tag index
        self._tag_to_stories = defaultdict(list)
        for sid, story in self.stories.items():
            for tag in story.tags:
                self._tag_to_stories[tag].append(sid)
        
        # Load story transitions (global)
        self.story_transitions = [
            StoryTransition.from_dict(t_data)
            for t_data in state.get('story_transitions', [])
        ]
        
        # Rebuild global transition graph
        self.global_transition_graph = defaultdict(lambda: defaultdict(list))
        for transition in self.story_transitions:
            self.global_transition_graph[transition.from_story_id][transition.to_story_id].append(transition)
        
        # Load users
        self.users = {}
        for uid, user_data in state.get('users', {}).items():
            user = UserProfile(uid)
            user.viewed_stories = {
                sid: datetime.fromisoformat(ts) 
                for sid, ts in user_data['viewed_stories'].items()
            }
            user.story_progress = {
                sid: (pct, datetime.fromisoformat(ts))
                for sid, (pct, ts) in user_data.get('story_progress', {}).items()
            }
            user.bookmarked_stories = {
                sid: datetime.fromisoformat(ts) 
                for sid, ts in user_data.get('bookmarked_stories', {}).items()
            }
            user.tag_interactions = defaultdict(list)
            for tag, interactions in user_data.get('tag_interactions', {}).items():
                user.tag_interactions[tag] = [
                    (score, datetime.fromisoformat(ts)) 
                    for score, ts in interactions
                ]
            user.story_connectedness = {
                sid: (score, datetime.fromisoformat(ts))
                for sid, (score, ts) in user_data.get('story_connectedness', {}).items()
            }
            user.question_responses = [
                (sid, qnum, resp, datetime.fromisoformat(ts))
                for sid, qnum, resp, ts in user_data.get('question_responses', [])
            ]
            user.recent_story_views = [
                (datetime.fromisoformat(ts), sid)
                for ts, sid in user_data.get('recent_story_views', [])
            ]
            user.recommendations_shown = [
                RecommendationRecord.from_dict(rec_data)
                for rec_data in user_data.get('recommendations_shown', [])
            ]
            user.story_ignore_count = defaultdict(int, user_data.get('story_ignore_count', {}))
            
            # Load sequences
            user.story_sequences = [
                StoryTransition.from_dict(t_data)
                for t_data in user_data.get('story_sequences', [])
            ]
            
            # Rebuild preferred_transitions from sequences
            user.preferred_transitions = defaultdict(list)
            user.tag_transition_preferences = defaultdict(lambda: defaultdict(list))
            for transition in user.story_sequences:
                user.preferred_transitions[transition.from_story_id].append(
                    (transition.to_story_id, transition.connectedness_delta, transition.timestamp)
                )
                
                if transition.from_story_id in self.stories and transition.to_story_id in self.stories:
                    from_tags = self.stories[transition.from_story_id].tags
                    to_tags = self.stories[transition.to_story_id].tags
                    
                    for from_tag in from_tags:
                        for to_tag in to_tags:
                            if transition.connectedness_delta is not None:
                                user.tag_transition_preferences[from_tag][to_tag].append(
                                    (transition.connectedness_delta, transition.timestamp)
                                )
            
            user.last_completed_story = user_data.get('last_completed_story')
            if user_data.get('last_completed_timestamp'):
                user.last_completed_timestamp = datetime.fromisoformat(user_data['last_completed_timestamp'])
            
            self.users[uid] = user
        
        # Load events
        self.events = [
            AnalyticsEvent.from_dict(event_data)
            for event_data in state.get('events', [])
        ]

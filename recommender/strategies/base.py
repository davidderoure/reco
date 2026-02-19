"""Abstract base class for all recommendation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from recommender.models import Story, UserProfile


class RecommendationStrategy(ABC):
    """Abstract base class for all recommendation strategies.

    Each strategy encapsulates a single recommendation approach
    (content-based, collaborative, topical, or wildcard).  The
    :class:`~recommender.engine.RecommendationEngine` calls each strategy
    in turn, passing the accumulated ``exclude_ids`` set so strategies
    can avoid returning already-selected stories.
    """

    @abstractmethod
    def recommend(
        self,
        profile: UserProfile,
        catalogue: list[Story],
        all_profiles: list[UserProfile],
        n: int,
        exclude_ids: set[str],
    ) -> list[str]:
        """Return up to *n* story IDs recommended for *profile*.

        Args:
            profile: The target user's current profile.
            catalogue: All available stories (pre-filtered from the catalogue).
            all_profiles: All user profiles (needed by group-based strategies).
            n: Maximum number of recommendations to return.
            exclude_ids: Story IDs already chosen by earlier strategies in
                this recommendation pass.  Strategies **should** avoid
                returning these, but **may** include them if not enough
                distinct candidates exist (small catalogue edge case).

        Returns:
            List of up to *n* story IDs, ordered by descending relevance.
        """

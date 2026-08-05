"""User feedback loop for CGX (Subsystem H).

Captures thumbs up/down + comments on ``ask``/``plan`` results, joined to the
provenance keys from Subsystem F (``run_id``, ``prompt_version``, ``model``).
The :mod:`~cgx.feedback.flywheel` closes the loop: it unifies feedback with the
cross-session ``lessons.jsonl`` store and exports down-votes as candidate rows
for the offline eval golden sets.
"""

from cgx.feedback.flywheel import (
    default_candidates_path,
    export_eval_candidates,
    unify_with_lessons,
)
from cgx.feedback.store import (
    RATINGS,
    Feedback,
    FeedbackStore,
    default_feedback_db_path,
    get_default_store,
)

__all__ = [
    "Feedback",
    "FeedbackStore",
    "RATINGS",
    "default_feedback_db_path",
    "get_default_store",
    "export_eval_candidates",
    "unify_with_lessons",
    "default_candidates_path",
]

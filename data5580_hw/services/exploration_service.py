"""Epsilon-greedy explore/exploit for cold-start candidate scoring.

When feedback data is sparse the system surfaces diverse candidates (exploration)
by randomly perturbing fit scores.  As ground-truth hiring outcomes accumulate
via the PATCH /actual feedback loop, epsilon decays automatically and the model's
own score takes over (exploitation).

    epsilon = max(MIN_EPSILON, BASE_EPSILON * exp(-DECAY_RATE * n_feedback))

No hand-tuning required: on day 0 with zero outcomes epsilon ≈ 0.8, meaning 80%
of scores are randomised to ensure candidate diversity.  After ~60 labelled
outcomes epsilon drops to ~0.05 (near-pure exploitation).
"""
import logging
import math
import random

logger = logging.getLogger(__name__)

_BASE_EPSILON = 0.8
_DECAY_RATE = 0.05
_MIN_EPSILON = 0.05


class ExplorationService:

    def compute_epsilon(self, n_feedback: int) -> float:
        """Return the current exploration rate given accumulated feedback count."""
        return max(_MIN_EPSILON, _BASE_EPSILON * math.exp(-_DECAY_RATE * n_feedback))

    def adjust_score(self, raw_score: float, n_feedback: int) -> float:
        """Return a possibly-randomised fit score.

        With probability *epsilon* (decaying as feedback grows) returns a
        uniform random score in [0, 1] to ensure candidate diversity during
        cold start.  Otherwise returns *raw_score* unchanged.
        """
        epsilon = self.compute_epsilon(n_feedback)
        if random.random() < epsilon:
            adjusted = random.random()
            logger.debug(
                "exploration_service explore n_feedback=%d epsilon=%.3f raw=%.3f adjusted=%.3f",
                n_feedback, epsilon, raw_score, adjusted,
            )
            return adjusted
        logger.debug(
            "exploration_service exploit n_feedback=%d epsilon=%.3f score=%.3f",
            n_feedback, epsilon, raw_score,
        )
        return raw_score


exploration_service = ExplorationService()

"""GitHub signal connector.

Extracts longitudinal behavioral features from a public GitHub profile and
normalises them into a flat feature vector that the prediction pipeline
already understands.  Adding a new signal source means adding a new module
in this package that emits the same dict[str, float] format — nothing
downstream changes.
"""
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests

from data5580_hw.services.errors import ModelLoadError

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_DEFAULT_TIMEOUT = 10
_MAX_REPOS = 100  # single page — enough for behavioural signal
_CACHE_TTL_SECONDS = 3600  # 1 hour


class GitHubConnectorError(Exception):
    """Raised when the GitHub API cannot be reached or returns an error."""


class GitHubConnector:
    """Fetch public behavioural signals for a GitHub user.

    Feature vector produced
    -----------------------
    account_age_days          days since account creation
    followers                 follower count
    public_repos              total owned public repos
    stars_received            sum of stargazers_count across owned repos
    language_diversity        count of distinct non-null languages
    recency_days              days since the most-recently-pushed repo
    avg_repo_stars            stars_received / max(public_repos, 1)
    forked_ratio              fraction of repos that are forks
    has_description_ratio     fraction of repos that have a description
    watchers_received         sum of watchers_count across owned repos
    """

    def __init__(self, github_token: Optional[str] = None, timeout: int = _DEFAULT_TIMEOUT,
                 cache_ttl: int = _CACHE_TTL_SECONDS):
        self._token = github_token or os.environ.get("GITHUB_TOKEN")
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        # {username: (fetched_at_epoch, feature_dict)}
        self._cache: Dict[str, Tuple[float, Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_features(self, username: str) -> Dict[str, float]:
        """Return a normalised feature vector for *username*.

        Results are cached for *cache_ttl* seconds to avoid repeated API
        calls for the same candidate within the same process lifetime.

        Raises GitHubConnectorError on API errors or unknown users.
        """
        cached = self._cache.get(username)
        if cached is not None:
            fetched_at, features = cached
            if time.monotonic() - fetched_at < self._cache_ttl:
                logger.debug("github_cache_hit username=%s", username)
                return features

        user = self._get_user(username)
        repos = self._get_repos(username)
        features = self._compute_features(user, repos)
        self._cache[username] = (time.monotonic(), features)
        return features

    def invalidate_cache(self, username: Optional[str] = None) -> None:
        """Remove a single entry (or all entries) from the feature cache."""
        if username is None:
            self._cache.clear()
        else:
            self._cache.pop(username, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _get(self, url: str) -> Any:
        try:
            response = requests.get(url, headers=self._headers(), timeout=self._timeout)
        except requests.RequestException as exc:
            raise GitHubConnectorError(f"GitHub API request failed: {exc}") from exc

        if response.status_code == 404:
            raise GitHubConnectorError(f"GitHub user or resource not found: {url}")
        if response.status_code == 403:
            raise GitHubConnectorError("GitHub API rate limit exceeded or forbidden.")
        if not response.ok:
            raise GitHubConnectorError(
                f"GitHub API returned {response.status_code}: {response.text[:200]}"
            )
        return response.json()

    def _get_user(self, username: str) -> Dict[str, Any]:
        return self._get(f"{_GITHUB_API}/users/{username}")

    def _get_repos(self, username: str) -> list:
        return self._get(
            f"{_GITHUB_API}/users/{username}/repos"
            f"?type=owner&per_page={_MAX_REPOS}&sort=pushed"
        )

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _compute_features(self, user: Dict[str, Any], repos: list) -> Dict[str, float]:
        now = datetime.now(timezone.utc)

        created_at = self._parse_dt(user.get("created_at"))
        account_age_days = (now - created_at).days if created_at else 0.0

        followers = float(user.get("followers", 0))
        public_repos = float(user.get("public_repos", 0))

        if not repos:
            return {
                "account_age_days": float(account_age_days),
                "followers": followers,
                "public_repos": public_repos,
                "stars_received": 0.0,
                "language_diversity": 0.0,
                "recency_days": float(account_age_days),
                "avg_repo_stars": 0.0,
                "forked_ratio": 0.0,
                "has_description_ratio": 0.0,
                "watchers_received": 0.0,
            }

        stars_received = float(sum(r.get("stargazers_count", 0) for r in repos))
        watchers_received = float(sum(r.get("watchers_count", 0) for r in repos))
        languages = {r.get("language") for r in repos if r.get("language")}
        language_diversity = float(len(languages))

        pushed_dates = [self._parse_dt(r.get("pushed_at")) for r in repos]
        pushed_dates = [d for d in pushed_dates if d is not None]
        if pushed_dates:
            most_recent = max(pushed_dates)
            recency_days = float((now - most_recent).days)
        else:
            recency_days = float(account_age_days)

        n_repos = len(repos)
        forked_count = sum(1 for r in repos if r.get("fork", False))
        described_count = sum(1 for r in repos if r.get("description"))

        avg_repo_stars = stars_received / max(n_repos, 1)
        forked_ratio = forked_count / max(n_repos, 1)
        has_description_ratio = described_count / max(n_repos, 1)

        return {
            "account_age_days": float(account_age_days),
            "followers": followers,
            "public_repos": public_repos,
            "stars_received": stars_received,
            "language_diversity": language_diversity,
            "recency_days": recency_days,
            "avg_repo_stars": avg_repo_stars,
            "forked_ratio": forked_ratio,
            "has_description_ratio": has_description_ratio,
            "watchers_received": watchers_received,
        }


github_connector = GitHubConnector()

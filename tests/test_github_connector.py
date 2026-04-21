"""Tests for the GitHub signal connector."""
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests

from data5580_hw.services.connectors.github_connector import (
    GitHubConnector,
    GitHubConnectorError,
    github_connector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_user(created_days_ago=365, followers=50, public_repos=10):
    created_at = (datetime.now(timezone.utc) - timedelta(days=created_days_ago)).isoformat()
    return {
        "login": "testuser",
        "followers": followers,
        "public_repos": public_repos,
        "created_at": created_at,
    }


def _mock_repos(count=5, stars=10, watchers=5, forked=False, language="Python", description="A repo"):
    pushed_at = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    return [
        {
            "name": f"repo-{i}",
            "stargazers_count": stars,
            "watchers_count": watchers,
            "fork": forked,
            "language": language,
            "description": description,
            "pushed_at": pushed_at,
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Constructor + headers
# ---------------------------------------------------------------------------

def test_connector_uses_env_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "mytoken")
    connector = GitHubConnector()
    assert connector._token == "mytoken"


def test_connector_explicit_token_overrides_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")
    connector = GitHubConnector(github_token="explicit-token")
    assert connector._token == "explicit-token"


def test_connector_no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    connector = GitHubConnector()
    assert connector._token is None


def test_headers_with_token():
    connector = GitHubConnector(github_token="abc123")
    headers = connector._headers()
    assert headers["Authorization"] == "Bearer abc123"
    assert "Accept" in headers


def test_headers_without_token():
    connector = GitHubConnector(github_token=None)
    headers = connector._headers()
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# _get error handling
# ---------------------------------------------------------------------------

def test_get_raises_on_404():
    connector = GitHubConnector()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.ok = False
    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(GitHubConnectorError, match="not found"):
            connector._get("https://api.github.com/users/nobody")


def test_get_raises_on_403():
    connector = GitHubConnector()
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.ok = False
    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(GitHubConnectorError, match="rate limit"):
            connector._get("https://api.github.com/users/someone")


def test_get_raises_on_other_error_status():
    connector = GitHubConnector()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.ok = False
    mock_resp.text = "Internal Server Error"
    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(GitHubConnectorError, match="500"):
            connector._get("https://api.github.com/users/someone")


def test_get_raises_on_request_exception():
    connector = GitHubConnector()
    with patch("requests.get", side_effect=requests.ConnectionError("no network")):
        with pytest.raises(GitHubConnectorError, match="request failed"):
            connector._get("https://api.github.com/users/someone")


def test_get_returns_json_on_success():
    connector = GitHubConnector()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {"login": "testuser"}
    with patch("requests.get", return_value=mock_resp):
        result = connector._get("https://api.github.com/users/testuser")
    assert result["login"] == "testuser"


# ---------------------------------------------------------------------------
# extract_features: happy path
# ---------------------------------------------------------------------------

def test_extract_features_returns_all_keys():
    connector = GitHubConnector()
    user = _mock_user()
    repos = _mock_repos(count=5)

    with patch.object(connector, "_get_user", return_value=user), \
         patch.object(connector, "_get_repos", return_value=repos):
        features = connector.extract_features("testuser")

    expected_keys = {
        "account_age_days", "followers", "public_repos", "stars_received",
        "language_diversity", "recency_days", "avg_repo_stars",
        "forked_ratio", "has_description_ratio", "watchers_received",
    }
    assert expected_keys == set(features.keys())


def test_extract_features_all_float():
    connector = GitHubConnector()
    user = _mock_user()
    repos = _mock_repos()

    with patch.object(connector, "_get_user", return_value=user), \
         patch.object(connector, "_get_repos", return_value=repos):
        features = connector.extract_features("testuser")

    for key, val in features.items():
        assert isinstance(val, float), f"{key} is not float"


def test_extract_features_stars():
    connector = GitHubConnector()
    repos = _mock_repos(count=3, stars=10)  # 3 repos × 10 stars = 30
    with patch.object(connector, "_get_user", return_value=_mock_user()), \
         patch.object(connector, "_get_repos", return_value=repos):
        features = connector.extract_features("u")
    assert features["stars_received"] == 30.0


def test_extract_features_language_diversity():
    connector = GitHubConnector()
    repos = [
        {"stargazers_count": 0, "watchers_count": 0, "fork": False,
         "language": "Python", "description": "x", "pushed_at": "2024-01-01T00:00:00Z"},
        {"stargazers_count": 0, "watchers_count": 0, "fork": False,
         "language": "Go", "description": "x", "pushed_at": "2024-01-01T00:00:00Z"},
        {"stargazers_count": 0, "watchers_count": 0, "fork": False,
         "language": None, "description": "", "pushed_at": "2024-01-01T00:00:00Z"},
    ]
    with patch.object(connector, "_get_user", return_value=_mock_user()), \
         patch.object(connector, "_get_repos", return_value=repos):
        features = connector.extract_features("u")
    assert features["language_diversity"] == 2.0  # None excluded


def test_extract_features_forked_ratio():
    connector = GitHubConnector()
    repos = _mock_repos(count=4, forked=False)
    repos[0]["fork"] = True
    repos[1]["fork"] = True
    with patch.object(connector, "_get_user", return_value=_mock_user()), \
         patch.object(connector, "_get_repos", return_value=repos):
        features = connector.extract_features("u")
    assert features["forked_ratio"] == pytest.approx(0.5)


def test_extract_features_description_ratio():
    connector = GitHubConnector()
    repos = _mock_repos(count=4, description="")
    repos[0]["description"] = "Has one"
    with patch.object(connector, "_get_user", return_value=_mock_user()), \
         patch.object(connector, "_get_repos", return_value=repos):
        features = connector.extract_features("u")
    assert features["has_description_ratio"] == pytest.approx(0.25)


def test_extract_features_avg_stars():
    connector = GitHubConnector()
    repos = _mock_repos(count=4, stars=8)  # 32 stars / 4 repos = 8
    with patch.object(connector, "_get_user", return_value=_mock_user(public_repos=4)), \
         patch.object(connector, "_get_repos", return_value=repos):
        features = connector.extract_features("u")
    assert features["avg_repo_stars"] == pytest.approx(8.0)


def test_extract_features_recency_days_recent():
    connector = GitHubConnector()
    pushed_at = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    repos = [{"stargazers_count": 0, "watchers_count": 0, "fork": False,
               "language": "Python", "description": "x", "pushed_at": pushed_at}]
    with patch.object(connector, "_get_user", return_value=_mock_user()), \
         patch.object(connector, "_get_repos", return_value=repos):
        features = connector.extract_features("u")
    assert features["recency_days"] == pytest.approx(3.0, abs=1.0)


# ---------------------------------------------------------------------------
# Empty repos edge case
# ---------------------------------------------------------------------------

def test_extract_features_empty_repos():
    connector = GitHubConnector()
    with patch.object(connector, "_get_user", return_value=_mock_user()), \
         patch.object(connector, "_get_repos", return_value=[]):
        features = connector.extract_features("u")
    assert features["stars_received"] == 0.0
    assert features["language_diversity"] == 0.0
    assert features["forked_ratio"] == 0.0
    assert features["has_description_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Repos with null pushed_at
# ---------------------------------------------------------------------------

def test_extract_features_null_pushed_at_falls_back_to_account_age():
    connector = GitHubConnector()
    repos = [{"stargazers_count": 0, "watchers_count": 0, "fork": False,
               "language": "Python", "description": "x", "pushed_at": None}]
    with patch.object(connector, "_get_user", return_value=_mock_user(created_days_ago=200)), \
         patch.object(connector, "_get_repos", return_value=repos):
        features = connector.extract_features("u")
    assert features["recency_days"] == pytest.approx(200.0, abs=2.0)


# ---------------------------------------------------------------------------
# _parse_dt edge cases
# ---------------------------------------------------------------------------

def test_parse_dt_valid():
    dt = GitHubConnector._parse_dt("2023-01-15T12:00:00Z")
    assert dt is not None
    assert dt.year == 2023


def test_parse_dt_none():
    assert GitHubConnector._parse_dt(None) is None


def test_parse_dt_invalid():
    assert GitHubConnector._parse_dt("not-a-date") is None


# ---------------------------------------------------------------------------
# API errors propagate through extract_features
# ---------------------------------------------------------------------------

def test_extract_features_propagates_connector_error():
    connector = GitHubConnector()
    with patch.object(connector, "_get_user", side_effect=GitHubConnectorError("404")):
        with pytest.raises(GitHubConnectorError):
            connector.extract_features("ghost")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def test_module_singleton_exists():
    assert github_connector is not None
    assert isinstance(github_connector, GitHubConnector)


# ---------------------------------------------------------------------------
# Response caching
# ---------------------------------------------------------------------------

def test_cache_hit_skips_api_call():
    """Second call for same username should use cache, not hit the API."""
    connector = GitHubConnector(cache_ttl=60)
    user = _mock_user()
    repos = _mock_repos()

    with patch.object(connector, "_get_user", return_value=user) as mock_user, \
         patch.object(connector, "_get_repos", return_value=repos) as mock_repos:
        connector.extract_features("cached-user")
        connector.extract_features("cached-user")

    # API helpers called exactly once despite two extract_features calls
    mock_user.assert_called_once()
    mock_repos.assert_called_once()


def test_cache_miss_after_ttl_expired():
    """After TTL expires the API should be called again."""
    import time
    connector = GitHubConnector(cache_ttl=0)  # TTL=0 → always expired
    user = _mock_user()
    repos = _mock_repos()

    with patch.object(connector, "_get_user", return_value=user) as mock_user, \
         patch.object(connector, "_get_repos", return_value=repos):
        connector.extract_features("stale-user")
        connector.extract_features("stale-user")

    assert mock_user.call_count == 2


def test_cache_different_users_stored_separately():
    connector = GitHubConnector(cache_ttl=60)
    user_a = _mock_user(followers=10)
    user_b = _mock_user(followers=999)
    repos = _mock_repos()

    with patch.object(connector, "_get_user", side_effect=[user_a, user_b]), \
         patch.object(connector, "_get_repos", return_value=repos):
        feat_a = connector.extract_features("alice")
        feat_b = connector.extract_features("bob")

    assert feat_a["followers"] == 10.0
    assert feat_b["followers"] == 999.0


def test_invalidate_cache_single_entry():
    connector = GitHubConnector(cache_ttl=60)
    user = _mock_user()
    repos = _mock_repos()

    with patch.object(connector, "_get_user", return_value=user), \
         patch.object(connector, "_get_repos", return_value=repos):
        connector.extract_features("evict-me")

    assert "evict-me" in connector._cache
    connector.invalidate_cache("evict-me")
    assert "evict-me" not in connector._cache


def test_invalidate_cache_all_entries():
    connector = GitHubConnector(cache_ttl=60)
    user = _mock_user()
    repos = _mock_repos()

    with patch.object(connector, "_get_user", return_value=user), \
         patch.object(connector, "_get_repos", return_value=repos):
        connector.extract_features("user1")
        connector.extract_features("user2")

    connector.invalidate_cache()
    assert connector._cache == {}


def test_cache_returns_consistent_features():
    connector = GitHubConnector(cache_ttl=60)
    user = _mock_user()
    repos = _mock_repos()

    with patch.object(connector, "_get_user", return_value=user), \
         patch.object(connector, "_get_repos", return_value=repos):
        first = connector.extract_features("repeat-user")
        second = connector.extract_features("repeat-user")

    assert first == second

"""Tests for skills/git_ops.py — GitHub URL parsing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills.git_ops import parse_github_url


def test_parse_full_url():
    assert parse_github_url("https://github.com/owner/repo") == "owner/repo"


def test_parse_url_with_git_suffix():
    assert parse_github_url("https://github.com/owner/repo.git") == "owner/repo"


def test_parse_url_with_trailing_slash():
    assert parse_github_url("https://github.com/owner/repo/") == "owner/repo"


def test_parse_shorthand():
    assert parse_github_url("owner/repo") == "owner/repo"


def test_parse_with_www():
    assert parse_github_url("https://www.github.com/owner/repo") == "owner/repo"


def test_parse_http():
    assert parse_github_url("http://github.com/owner/repo") == "owner/repo"


def test_parse_invalid():
    assert parse_github_url("not a url") is None


def test_parse_empty():
    assert parse_github_url("") is None


def test_parse_other_domain():
    assert parse_github_url("https://gitlab.com/owner/repo") is None


def test_parse_with_dots_and_hyphens():
    assert parse_github_url("my-org/my-repo.js") == "my-org/my-repo.js"

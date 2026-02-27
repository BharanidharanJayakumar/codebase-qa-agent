"""
Git operations for cloning repositories from GitHub URLs.
Clones to a managed directory so indexed repos persist across sessions.
"""
import re
import subprocess
from pathlib import Path

REPOS_DIR = Path.home() / ".codebase-qa-agent" / "repos"

# Matches: https://github.com/user/repo, github.com/user/repo, user/repo
GITHUB_URL_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+?)(?:\.git)?/?$"
)
SHORTHAND_PATTERN = re.compile(
    r"^([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)$"
)


def parse_github_url(url: str) -> str | None:
    """Extract 'owner/repo' from a GitHub URL or shorthand. Returns None if invalid."""
    url = url.strip()
    m = GITHUB_URL_PATTERN.match(url)
    if m:
        return m.group(1)
    m = SHORTHAND_PATTERN.match(url)
    if m:
        return m.group(1)
    return None


def clone_repo(url: str) -> dict:
    """Clone a GitHub repo to the managed repos directory.
    Returns {"path": str, "owner_repo": str} on success, {"error": str} on failure."""
    owner_repo = parse_github_url(url)
    if not owner_repo:
        return {"error": f"Invalid GitHub URL: {url}. Expected format: https://github.com/owner/repo"}

    clone_url = f"https://github.com/{owner_repo}.git"
    repo_name = owner_repo.replace("/", "_")
    target_dir = REPOS_DIR / repo_name

    # If already cloned, pull latest
    if target_dir.exists() and (target_dir / ".git").exists():
        try:
            subprocess.run(
                ["git", "-C", str(target_dir), "pull", "--ff-only"],
                capture_output=True, text=True, timeout=120,
            )
            return {"path": str(target_dir), "owner_repo": owner_repo, "action": "updated"}
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            return {"error": f"Failed to update {owner_repo}: {e}"}

    # Fresh clone
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(target_dir)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return {"error": f"git clone failed: {result.stderr.strip()}"}
        return {"path": str(target_dir), "owner_repo": owner_repo, "action": "cloned"}
    except subprocess.TimeoutExpired:
        return {"error": f"Clone timed out for {owner_repo} (5min limit)"}
    except FileNotFoundError:
        return {"error": "git is not installed. Install git and try again."}

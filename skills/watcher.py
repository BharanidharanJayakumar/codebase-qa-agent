"""
File watcher for automatic incremental index updates.
Uses watchfiles (Rust-based, efficient) to monitor project directories.
"""
import asyncio
import threading
from pathlib import Path

from skills.scanner import SUPPORTED_EXTENSIONS, IGNORED_DIRS

_active_watchers: dict[str, threading.Event] = {}


async def start_watching(project_path: str, on_change_callback) -> str:
    """
    Start watching a project directory for file changes.
    Calls on_change_callback(project_path) when relevant files change.
    Returns a watcher ID for stopping later.
    """
    try:
        from watchfiles import awatch, Change
    except ImportError:
        return ""

    project = Path(project_path).resolve()
    watcher_id = str(project)

    if watcher_id in _active_watchers:
        return watcher_id  # Already watching

    stop_event = threading.Event()
    _active_watchers[watcher_id] = stop_event

    async def _watch_loop():
        try:
            async for changes in awatch(
                project,
                stop_event=stop_event,
                debounce=2000,  # 2s debounce to batch rapid saves
                recursive=True,
            ):
                # Filter to only relevant file changes
                relevant = False
                for change_type, path_str in changes:
                    p = Path(path_str)
                    # Skip ignored directories
                    if any(part in IGNORED_DIRS for part in p.parts):
                        continue
                    if p.suffix in SUPPORTED_EXTENSIONS:
                        relevant = True
                        break

                if relevant:
                    try:
                        await on_change_callback(project_path)
                    except Exception:
                        pass  # Don't crash the watcher on callback errors
        except Exception:
            pass
        finally:
            _active_watchers.pop(watcher_id, None)

    asyncio.create_task(_watch_loop())
    return watcher_id


def stop_watching(project_path: str) -> bool:
    """Stop watching a project directory. Returns True if a watcher was stopped."""
    watcher_id = str(Path(project_path).resolve())
    stop_event = _active_watchers.pop(watcher_id, None)
    if stop_event:
        stop_event.set()
        return True
    return False


def list_watchers() -> list[str]:
    """List all actively watched project paths."""
    return list(_active_watchers.keys())

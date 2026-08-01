

"""Agent Profile store for CGX.

An Agent Profile is a saved, reusable {objective, project root, mode,
skills} bundle that can be "launched" into a new agent session without
re-typing the task each time. This is unrelated to
:mod:`cgx.answer.profiles` (saved LLM provider connections) -- the two
happen to share the word "profile" but nothing else; this store holds
no secrets, so it's a plain JSON file with no keyring involvement.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

CONFIG_DIR = Path(os.environ.get("CGX_CONFIG_DIR", str(Path.home() / ".cgx")))
AGENT_PROFILES_PATH = CONFIG_DIR / "agent_profiles.json"


@dataclass
class AgentProfile:
    name: str
    objective: str
    project_root: str = ""
    mode: str = ""  # "" (auto) | "explore" | "greenfield"
    skills: List[str] = field(default_factory=list)

    def to_public_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, stat.S_IRWXU)
    except Exception as e:
        logger.warning("agent_profiles: chmod on %s failed: %s: %s",
                       CONFIG_DIR, type(e).__name__, e)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("agent_profiles: failed to parse %s: %s: %s",
                       path, type(e).__name__, e)
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir()
    payload = json.dumps(data, indent=2).encode("utf-8")
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    fd = os.open(str(path), flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


def list_agent_profiles() -> List[AgentProfile]:
    raw = _read_json(AGENT_PROFILES_PATH)
    out: List[AgentProfile] = []
    for name, p in (raw.get("profiles") or {}).items():
        if not isinstance(p, dict):
            continue
        skills = p.get("skills")
        out.append(AgentProfile(
            name=name,
            objective=str(p.get("objective", "")),
            project_root=str(p.get("project_root", "")),
            mode=str(p.get("mode", "")),
            skills=list(skills) if isinstance(skills, list) else [],
        ))
    out.sort(key=lambda x: x.name.lower())
    return out


def get_agent_profile(name: str) -> Optional[AgentProfile]:
    for p in list_agent_profiles():
        if p.name == name:
            return p
    return None


def save_agent_profile(profile: AgentProfile) -> AgentProfile:
    raw = _read_json(AGENT_PROFILES_PATH)
    profiles = raw.get("profiles") or {}
    profiles[profile.name] = {
        "objective": profile.objective,
        "project_root": profile.project_root,
        "mode": profile.mode,
        "skills": list(profile.skills),
    }
    raw["profiles"] = profiles
    _write_json(AGENT_PROFILES_PATH, raw)
    logger.info("agent_profiles: saved name=%r skills=%s", profile.name, profile.skills)
    return profile


def delete_agent_profile(name: str) -> bool:
    raw = _read_json(AGENT_PROFILES_PATH)
    profiles = raw.get("profiles") or {}
    if name not in profiles:
        return False
    profiles.pop(name, None)
    raw["profiles"] = profiles
    _write_json(AGENT_PROFILES_PATH, raw)
    logger.info("agent_profiles: deleted name=%r", name)
    return True

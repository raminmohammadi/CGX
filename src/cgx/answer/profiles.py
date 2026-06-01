"""Provider profile store for Averix.

Profiles let users save named LLM endpoint configurations (provider kind,
model, base URL, optional API key) and recall them from the UI without
re-entering credentials. API keys are stored in OS keyring when the optional
``keyring`` package is installed, otherwise in a permission-restricted file
under ``~/.cgx/``.

The store deliberately avoids surfacing raw secret values through tool/log
boundaries: only profile names and non-secret metadata are returned by the
public ``list_*`` helpers. ``load_profile`` materialises the key just-in-time
for an in-process provider instance.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

CONFIG_DIR = Path(os.environ.get("CGX_CONFIG_DIR", str(Path.home() / ".cgx")))
PROFILES_PATH = CONFIG_DIR / "profiles.json"
SECRETS_PATH = CONFIG_DIR / "secrets.json"
KEYRING_SERVICE = "averix"


@dataclass
class Profile:
    name: str
    kind: str  # "ollama" | "openai-compat" | "gemini" | "custom"
    model: str
    base_url: str
    temperature: float = 0.2
    num_predict: int = 1024
    has_api_key: bool = False  # metadata only; never holds the key
    # Optional client-side rate limiting + retry, persisted with the profile so
    # cloud providers can keep their per-tenant limits without re-entering them.
    rate_limit: Optional[float] = None  # requests/sec; None disables limiting
    max_retries: Optional[int] = None   # None = provider default
    # Phase 5: custom-server fields
    endpoint_path: str = "/v1/chat/completions"  # custom endpoint suffix
    allow_no_auth: bool = False  # skip Bearer auth for private-subnet servers

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, stat.S_IRWXU)
    except Exception as e:
        logger.warning("profiles: chmod on %s failed: %s: %s",
                       CONFIG_DIR, type(e).__name__, e)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("profiles: failed to parse %s: %s: %s",
                       path, type(e).__name__, e)
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


def _keyring():
    try:
        import keyring  # type: ignore
        return keyring
    except Exception:
        return None


def _store_secret(name: str, secret: str) -> None:
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, name, secret)
            return
        except Exception as e:
            logger.warning("profiles: keyring set_password failed for profile %r, "
                           "falling back to secrets file: %s: %s",
                           name, type(e).__name__, e)
    data = _read_json(SECRETS_PATH)
    data[name] = secret
    _write_json(SECRETS_PATH, data)


def _load_secret(name: str) -> Optional[str]:
    kr = _keyring()
    if kr is not None:
        try:
            val = kr.get_password(KEYRING_SERVICE, name)
            if val:
                return val
        except Exception as e:
            logger.warning("profiles: keyring get_password failed for profile %r, "
                           "falling back to secrets file: %s: %s",
                           name, type(e).__name__, e)
    data = _read_json(SECRETS_PATH)
    val = data.get(name)
    return val if isinstance(val, str) and val else None


def _delete_secret(name: str) -> None:
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(KEYRING_SERVICE, name)
        except Exception as e:
            logger.warning("profiles: keyring delete_password failed for profile %r: %s: %s",
                           name, type(e).__name__, e)
    data = _read_json(SECRETS_PATH)
    if name in data:
        data.pop(name, None)
        _write_json(SECRETS_PATH, data)


def list_profiles() -> List[Profile]:
    raw = _read_json(PROFILES_PATH)
    out: List[Profile] = []
    for name, p in (raw.get("profiles") or {}).items():
        if not isinstance(p, dict):
            continue
        rl = p.get("rate_limit")
        mr = p.get("max_retries")
        out.append(Profile(
            name=name,
            kind=str(p.get("kind", "ollama")),
            model=str(p.get("model", "")),
            base_url=str(p.get("base_url", "")),
            temperature=float(p.get("temperature", 0.2)),
            num_predict=int(p.get("num_predict", 1024)),
            has_api_key=bool(p.get("has_api_key", False)),
            rate_limit=(float(rl) if isinstance(rl, (int, float)) and rl > 0 else None),
            max_retries=(int(mr) if isinstance(mr, (int, float)) and mr >= 0 else None),
            endpoint_path=str(p.get("endpoint_path", "/v1/chat/completions")),
            allow_no_auth=bool(p.get("allow_no_auth", False)),
        ))
    out.sort(key=lambda x: x.name.lower())
    return out


def get_profile(name: str) -> Optional[Profile]:
    for p in list_profiles():
        if p.name == name:
            return p
    return None


def save_profile(profile: Profile, api_key: Optional[str] = None) -> Profile:
    raw = _read_json(PROFILES_PATH)
    profiles = raw.get("profiles") or {}
    has_key = False
    if api_key:
        _store_secret(profile.name, api_key)
        has_key = True
    elif profile.has_api_key:
        has_key = _load_secret(profile.name) is not None
    profile.has_api_key = has_key
    entry: Dict[str, Any] = {
        "kind": profile.kind,
        "model": profile.model,
        "base_url": profile.base_url,
        "temperature": profile.temperature,
        "num_predict": profile.num_predict,
        "has_api_key": has_key,
        "endpoint_path": profile.endpoint_path or "/v1/chat/completions",
        "allow_no_auth": bool(profile.allow_no_auth),
    }
    if profile.rate_limit is not None:
        entry["rate_limit"] = float(profile.rate_limit)
    if profile.max_retries is not None:
        entry["max_retries"] = int(profile.max_retries)
    profiles[profile.name] = entry
    raw["profiles"] = profiles
    _write_json(PROFILES_PATH, raw)
    logger.info("profiles: saved name=%r kind=%s model=%s has_api_key=%s",
                profile.name, profile.kind, profile.model, has_key)
    return profile


def delete_profile(name: str) -> bool:
    raw = _read_json(PROFILES_PATH)
    profiles = raw.get("profiles") or {}
    if name not in profiles:
        logger.info("profiles: delete no-op (not found) name=%r", name)
        return False
    profiles.pop(name, None)
    raw["profiles"] = profiles
    _write_json(PROFILES_PATH, raw)
    _delete_secret(name)
    logger.info("profiles: deleted name=%r", name)
    return True


def load_api_key(name: str) -> Optional[str]:
    """Load the API key for profile ``name`` from keyring/secrets file.

    Never log or echo the returned value back to UI/transcript surfaces.
    """
    return _load_secret(name)

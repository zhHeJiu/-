import json
import os
import sys
import threading
from pathlib import Path

from dotenv import dotenv_values


def _app_root() -> Path:
    configured = os.getenv("MORAL_ASSESSOR_CONFIG_DIR")
    if configured:
        return Path(configured).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).parent.parent


PROJECT_ROOT = _app_root()
CONFIG_PATH = PROJECT_ROOT / "app_config.json"
LEGACY_ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_CONFIG = {
    "minimax_api_key": "",
    "minimax_base_url": "https://api.minimaxi.com/v1",
    "model": "MiniMax-M2.7-highspeed",
    "default_temperature": 0.2,
    "judge_temperature": 0.1,
    "llm_max_retries": 2,
}

_lock = threading.Lock()


class ConfigError(ValueError):
    """Raised when persisted or submitted application config is invalid."""


def load_config() -> dict:
    """Read the persisted LLM config, creating it from defaults or legacy env values."""
    with _lock:
        if CONFIG_PATH.exists():
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            config = _normalize_config({**DEFAULT_CONFIG, **raw})
        else:
            config = _normalize_config({**DEFAULT_CONFIG, **_legacy_env_config()})
            _write_config_unlocked(config)
        return config


def public_config() -> dict:
    config = load_config()
    api_key = config.get("minimax_api_key", "")
    public = {key: value for key, value in config.items() if key != "minimax_api_key"}
    public["has_api_key"] = bool(api_key)
    public["api_key_hint"] = _mask_secret(api_key)
    public["config_path"] = str(CONFIG_PATH)
    return public


def update_config(changes: dict) -> dict:
    with _lock:
        current = load_config_unlocked()
        next_config = dict(current)

        if changes.get("clear_api_key"):
            next_config["minimax_api_key"] = ""
        elif "minimax_api_key" in changes and changes.get("minimax_api_key"):
            next_config["minimax_api_key"] = str(changes["minimax_api_key"]).strip()

        for key in ("minimax_base_url", "model"):
            if key in changes and changes[key] is not None:
                next_config[key] = str(changes[key]).strip()

        for key in ("default_temperature", "judge_temperature"):
            if key in changes and changes[key] is not None:
                next_config[key] = changes[key]

        if "llm_max_retries" in changes and changes["llm_max_retries"] is not None:
            next_config["llm_max_retries"] = changes["llm_max_retries"]

        normalized = _normalize_config(next_config)
        _write_config_unlocked(normalized)
        return normalized


def load_config_unlocked() -> dict:
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return _normalize_config({**DEFAULT_CONFIG, **raw})
    config = _normalize_config({**DEFAULT_CONFIG, **_legacy_env_config()})
    _write_config_unlocked(config)
    return config


def _legacy_env_config() -> dict:
    legacy = dotenv_values(LEGACY_ENV_PATH) if LEGACY_ENV_PATH.exists() else {}

    def value(env_name: str, default: str = "") -> str:
        return os.getenv(env_name) or legacy.get(env_name) or default

    return {
        "minimax_api_key": value("MINIMAX_API_KEY"),
        "minimax_base_url": value("MINIMAX_BASE_URL", DEFAULT_CONFIG["minimax_base_url"]),
        "model": value("MODEL", DEFAULT_CONFIG["model"]),
        "default_temperature": value("DEFAULT_TEMPERATURE", str(DEFAULT_CONFIG["default_temperature"])),
        "judge_temperature": value("JUDGE_TEMPERATURE", str(DEFAULT_CONFIG["judge_temperature"])),
        "llm_max_retries": value("LLM_MAX_RETRIES", str(DEFAULT_CONFIG["llm_max_retries"])),
    }


def _normalize_config(raw: dict) -> dict:
    api_key = str(raw.get("minimax_api_key") or "").strip()
    base_url = str(raw.get("minimax_base_url") or "").strip().rstrip("/")
    model = str(raw.get("model") or "").strip()

    if not base_url:
        raise ConfigError("MINIMAX_BASE_URL 不能为空")
    if not model:
        raise ConfigError("MODEL 不能为空")

    return {
        "minimax_api_key": api_key,
        "minimax_base_url": base_url,
        "model": model,
        "default_temperature": _coerce_float(raw.get("default_temperature"), "DEFAULT_TEMPERATURE", 0.0, 2.0),
        "judge_temperature": _coerce_float(raw.get("judge_temperature"), "JUDGE_TEMPERATURE", 0.0, 2.0),
        "llm_max_retries": _coerce_int(raw.get("llm_max_retries"), "LLM_MAX_RETRIES", 0, 5),
    }


def _coerce_float(value, label: str, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} 必须是数字") from exc
    if numeric < minimum or numeric > maximum:
        raise ConfigError(f"{label} 必须在 {minimum:g}-{maximum:g} 之间")
    return round(numeric, 3)


def _coerce_int(value, label: str, minimum: int, maximum: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} 必须是整数") from exc
    if numeric < minimum or numeric > maximum:
        raise ConfigError(f"{label} 必须在 {minimum}-{maximum} 之间")
    return numeric


def _write_config_unlocked(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CONFIG_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(CONFIG_PATH)


def _mask_secret(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"

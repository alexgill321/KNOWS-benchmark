"""Local configuration; never pass evaluator secrets to the agent process."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-6-luna", "gpt-6-astra", "gpt-6.1-sol")
SECRET_ENV_ALLOWLIST = {
    "GOOGLE_AI_API_KEY", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS",
    "USDA_API_KEY", "GOOGLE_MAPS_API_KEY", "S2_API_KEY", "HF_KEY",
}


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    config["config_path"] = str(path)
    for key in ("benchmark_root", "storage_state", "service_account_path", "token_path",
                "credentials_path", "output_dir", "python"):
        if config.get(key):
            p = Path(config[key]).expanduser()
            config[key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
    config.setdefault("benchmark_root", str(ROOT))
    config.setdefault("models", list(MODELS))
    config.setdefault("efforts", {model: "max" for model in MODELS})
    config.setdefault("max_actions", 150)
    config.setdefault("episode_timeout_seconds", 1800)
    config.setdefault("evaluation_timeout_seconds", 900)
    config.setdefault("headless", True)
    config.setdefault("channel", "chrome")
    return config


def apply_evaluator_environment(config: dict) -> None:
    # Explicit allowlist: unrelated API keys and Google passwords stay out of this process.
    from dotenv import dotenv_values
    for filename in config.get("env_files", []):
        p = Path(filename).expanduser()
        if not p.is_absolute():
            p = Path(config["config_path"]).parent / p
        for key, value in dotenv_values(p).items():
            if key in SECRET_ENV_ALLOWLIST and value:
                os.environ.setdefault(key, value)
    if not os.environ.get("GOOGLE_AI_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_AI_API_KEY"] = os.environ["GEMINI_API_KEY"]
    for key, env in (("service_account_path", "SERVICE_ACCOUNT_PATH"),
                     ("token_path", "TOKEN_PATH"),
                     ("credentials_path", "CLIENT_SECRETS_PATH")):
        if config.get(key):
            os.environ[env] = config[key]
    os.environ["PYTHONPATH"] = os.pathsep.join((config["benchmark_root"] + "/src", str(ROOT)))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)

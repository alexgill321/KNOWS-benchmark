"""Renew owner Google OAuth into this checkout without overwriting other projects."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import ROOT, load_config, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmark.local.json")
    args = parser.parse_args()
    config = load_config(args.config)
    from google_auth_oauthlib.flow import InstalledAppFlow
    # Same Workspace access the KNOWS owner provisioning/evaluation helpers use.
    scopes = ["https://www.googleapis.com/auth/drive",
              "https://www.googleapis.com/auth/documents",
              "https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/presentations"]
    flow = InstalledAppFlow.from_client_secrets_file(config["credentials_path"], scopes=scopes)
    credentials = flow.run_local_server(port=0, timeout_seconds=300,
        prompt="select_account consent", access_type="offline",
        authorization_prompt_message="Complete Google consent in the browser using your dedicated KNOWS account.",
        success_message="KNOWS Google authorization saved. You may close this tab.")
    output = ROOT / "auth-data" / "owner-token.json"
    output.parent.mkdir(exist_ok=True, mode=0o700)
    output.write_text(credentials.to_json())
    output.chmod(0o600)
    source = Path(config["config_path"])
    raw = json.loads(source.read_text())
    raw["token_path"] = str(output)
    write_json(source, raw)
    print("Owner OAuth renewed; benchmark.local.json now references the local token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

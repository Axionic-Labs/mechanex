import json
import os
from typing import Iterable, List

from mechanex import Mechanex


DEFAULT_BASE_URL = "https://axionic-backend-prod-594546489999.us-east4.run.app"


def require_env(keys: Iterable[str]) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required environment variables: {joined}")


def get_remote_client() -> Mechanex:
    require_env(["MECHANEX_API_KEY"])
    base_url = os.getenv("MECHANEX_BASE_URL", DEFAULT_BASE_URL)
    mx = Mechanex(base_url=base_url).set_execution_mode("remote")
    mx.set_key(os.getenv("MECHANEX_API_KEY"))
    return mx


def get_local_client() -> Mechanex:
    model_name = os.getenv("MECHANEX_LOCAL_MODEL", "gpt2-small")
    mx = Mechanex().set_execution_mode("local")
    try:
        mx.load_model(model_name)
    except Exception as exc:
        raise SystemExit(
            "Local model load failed. Install transformer-lens and ensure the model is available. "
            f"Error: {exc}"
        )
    return mx


def load_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

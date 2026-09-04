"""Do quota con lai cua tung account Kaggle bang phep thu DAT COC.

Proxy dat coc tien theo `max_output_tokens` chu khong theo token thuc tieu (CLAUDE.md
BAY 5). Nen account gan can se OK o cap nho va 403 o cap lon. Hai lan goi cung mot
model, cap=16 roi cap=4096, du de xep hang account ma ton vai xu.
"""
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

STORE = Path(r"D:/AI_PhD/GameTheory/kaggle_for_research")
CFG = Path(r"D:/tmp/kcfg")
MODEL = "gemini-3.5-flash-lite"


def tokens():
    out = []
    for f in sorted((STORE / "kaggle-api").glob("*.txt")):
        out.append((f.stem, f.read_text(encoding="utf-8").strip()))
    for f in sorted((STORE / "kaggle-api-2").glob("*.md")):
        tok = next((l.strip() for l in f.read_text(encoding="utf-8").splitlines()
                    if l.strip().startswith("KGAT_")), None)
        if tok:
            out.append((f.stem, tok))
    return out


def proxy_key(name, token):
    d = CFG / name
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KAGGLE_CONFIG_DIR=str(d), KAGGLE_API_TOKEN=token,
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    envf = d / "acct.env"
    r = subprocess.run([sys.executable, "-m", "kaggle", "benchmarks", "auth", "-y",
                        "--env-file", str(envf)],
                       capture_output=True, text=True, env=env, timeout=180)
    if not envf.exists():
        return None, None, (r.stderr or r.stdout).strip().splitlines()[-1:] or ["?"]
    kv = dict(re.findall(r"^(\w+)=(.*)$", envf.read_text(encoding="utf-8"), re.M))
    return kv.get("MODEL_PROXY_URL"), kv.get("MODEL_PROXY_API_KEY"), None


def call(url, key, cap):
    body = json.dumps({"model": MODEL, "max_tokens": cap,
                       "messages": [{"role": "user", "content": "Say OK"}]}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/openapi/chat/completions", body,
                                 {"Authorization": f"Bearer {key}",
                                  "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return "OK" if b"choices" in r.read() else "no-choices"
    except Exception as e:  # noqa: BLE001
        m = str(e)
        detail = getattr(e, "read", None)
        if detail:
            try:
                m += " " + detail().decode()[:160]
            except Exception:  # noqa: BLE001
                pass
        if "403" in m and "quota" in m:
            return "403-QUOTA"
        return m[:110]


def main():
    print(f"{'account':<16} {'cap=16':<12} {'cap=4096':<12} ghi chu")
    print("-" * 78)
    for name, tok in tokens():
        url, key, err = proxy_key(name, tok)
        if not key:
            print(f"{name:<16} {'-':<12} {'-':<12} auth that bai: {err}")
            continue
        small = call(url, key, 16)
        big = call(url, key, 4096) if small == "OK" else "-"
        verdict = ("CON NHIEU" if big == "OK" else
                   "GAN CAN" if small == "OK" else "KHONG DUNG DUOC")
        print(f"{name:<16} {small:<12} {big:<12} {verdict}")


if __name__ == "__main__":
    main()

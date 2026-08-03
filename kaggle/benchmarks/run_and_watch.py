#!/usr/bin/env python
"""Chạy sweep FAIRGAME PD trên Kaggle Benchmarks + theo dõi realtime + tải về local.

Vì sao cần script này: `kaggle b t run` chạy TRÊN SERVER Kaggle. Checkpoint mà
`pd_task.py` ghi nằm trong session của server, mất kết nối là không xem được tiến độ
và không lấy được kết quả. Script này bám theo log của run, in tiến độ ra màn hình
(bao nhiêu game / bao nhiêu %), và tải output về máy ngay khi run xong.

Dùng:
    # chạy 1 model, bám log tới khi xong rồi tải về
    python run_and_watch.py -m google/gemini-3.5-flash-lite

    # chạy lần lượt nhiều model
    python run_and_watch.py -m google/gemini-3.5-flash-lite google/gemini-3.6-flash

    # không chạy mới, chỉ bám theo run đang chạy dở + tải về
    python run_and_watch.py --attach

Kết quả tải về:
    <--dest>/<lambda>/<model>/x<lambda>_<lang>_<model>.csv     (đúng layout Dataset/)
    <--dest>/<model>/checkpoints/*.json                        (1 shard = 1 game)

Hết credit giữa chừng thì `pd_task.py` tự dừng sớm và GIỮ nguyên checkpoint; đổi API
key rồi chạy lại đúng lệnh này là resume tiếp, không tính tiền lại phần đã xong.
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TASK = "prisoner-dilemma-fairgame"
HERE = Path(__file__).resolve().parent

# Dòng tiến độ do pd_task.py in ra: "[123/600 | 20.5%] pd__...  coop=13/20 ..."
PROGRESS_RE = re.compile(r"\[(\d+)/(\d+) \| ([\d.]+)%\]\s+(\S+)\s+coop=(\d+)/(\d+)")
NOISE_RE = re.compile(r"cryptography|CryptographyDeprecation|TripleDES|Blowfish|"
                      r'"class": algorithms|"cipher": algorithms')


def kaggle(*args, timeout=1800):
    """Gọi kaggle CLI, trả (returncode, stdout+stderr) đã lọc bớt warning rác."""
    proc = subprocess.run(
        [sys.executable, "-m", "kaggle", "benchmarks", "tasks", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(HERE), timeout=timeout,
    )
    out = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines()
                    if not NOISE_RE.search(l))
    return proc.returncode, out


def watch(poll):
    """Bám log server, chỉ in DÒNG MỚI. Trả về True nếu run kết thúc bình thường."""
    seen = 0
    last_pct = -1.0
    stall = 0
    while True:
        rc, out = kaggle("log", TASK, timeout=600)
        lines = out.splitlines()

        for line in lines[seen:]:
            m = PROGRESS_RE.search(line)
            if m:
                done, total, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))
                calls = done * 20            # 10 vòng × 2 agent
                tcalls = total * 20
                bar_n = int(pct / 100 * 32)
                bar = "#" * bar_n + "." * (32 - bar_n)
                print(f"\r[{bar}] {pct:6.2f}%  {done}/{total} game  "
                      f"{calls}/{tcalls} lượt gọi", end="", flush=True)
                last_pct = pct
            elif line.strip():
                # Không phải dòng tiến độ (summary, cảnh báo, lỗi) -> in nguyên văn.
                print(f"\n{line}", flush=True)
        if len(lines) > seen:
            seen = len(lines)
            stall = 0
        else:
            stall += 1

        blob = out.lower()
        if "summary =====" in blob or "result:" in blob:
            print(f"\n[watch] run đã kết thúc (tiến độ cuối {last_pct:.1f}%).", flush=True)
            return True
        if "errored" in blob or "traceback (most recent call last)" in blob:
            print("\n[watch] run báo lỗi — xem log ở trên.", flush=True)
            return False
        if stall * poll > 3600:
            print("\n[watch] 60 phút không có log mới; dừng bám. "
                  "Run có thể vẫn đang chạy trên server.", flush=True)
            return False
        time.sleep(poll)


def download(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rc, out = kaggle("download", TASK, "-o", str(dest), timeout=3600)
    print(out)
    csvs = sorted(dest.rglob("x*_*.csv"))
    ckpts = list(dest.rglob("checkpoints/*.json"))
    print(f"[download] {len(csvs)} CSV, {len(ckpts)} checkpoint -> {dest}")
    for c in csvs:
        n = sum(1 for _ in c.open(encoding="utf-8")) - 1
        flag = "" if n == 40 else f"  <-- {n} game (chưa đủ 40)"
        print(f"    {c.relative_to(dest)}: {n} game{flag}")
    return csvs


def collect(csvs, dataset_dir):
    """Chép CSV vào Dataset/ theo đúng layout <lambda>/<model>/ đang dùng."""
    if not csvs:
        return
    dataset_dir = Path(dataset_dir)
    for c in csvs:
        # .../<lambda>/<model>/x<lambda>_<lang>_<model>.csv
        target = dataset_dir / c.parent.parent.name / c.parent.name / c.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c, target)
        print(f"[collect] {target}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--models", nargs="+", default=[],
                    help="slug model, ví dụ google/gemini-3.5-flash-lite")
    ap.add_argument("--attach", action="store_true",
                    help="không chạy mới, chỉ bám run đang chạy rồi tải về")
    ap.add_argument("--dest", default=str(HERE.parent.parent / "results" / "kbench_download"))
    ap.add_argument("--collect-into", default=None,
                    help="chép CSV vào thư mục Dataset (vd Dataset/data_fairgame_frontier_llm)")
    ap.add_argument("--poll", type=int, default=30, help="giây giữa 2 lần đọc log")
    args = ap.parse_args()

    if args.attach:
        watch(args.poll)
        collect(download(args.dest), args.collect_into) if args.collect_into \
            else download(args.dest)
        return

    if not args.models:
        ap.error("cần -m <model slug> (hoặc --attach)")

    for model in args.models:
        print(f"\n{'=' * 70}\n== chạy {model}\n{'=' * 70}", flush=True)
        rc, out = kaggle("run", TASK, "-m", model, timeout=900)
        print(out, flush=True)
        if rc != 0 and "scheduled" not in out.lower():
            print(f"[error] không submit được {model}; bỏ qua.", flush=True)
            continue
        ok = watch(args.poll)
        csvs = download(args.dest)
        if args.collect_into:
            collect(csvs, args.collect_into)
        if not ok:
            print(f"[warn] {model} chưa xong trọn vẹn. Checkpoint đã giữ — "
                  "chạy lại đúng lệnh này để resume.", flush=True)


if __name__ == "__main__":
    main()

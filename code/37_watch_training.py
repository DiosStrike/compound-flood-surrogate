#!/usr/bin/env python3
"""Read-only training monitor: refreshes every few seconds and shows process status, current epoch / step, time used and estimated remaining for this epoch, total estimated remaining, per-epoch loss, latest log.
Modifies no files and does not affect the running training. Press Ctrl+C to quit the monitor (training continues).

Process matching: only 34_modelB_train.py processes whose command line has `--exp <name>` or `--out result/models/<name>` matching --exp
(caffeinate wrapper processes are only shown, not timed).
Step-level progress: supports two train_stdout.log formats
  old (exp1 process):  "  ep 1 step 100 loss 0.023186 0.342s/step mps 2.44 GB" (step is cumulative; steps per epoch taken from the header "/ 1080 steps")
  new (training started from 2026-09-16 on): "[2026-09-16 18:30:00] ep 2/50 step 100/1080 | epoch elapsed ... | loss (last 100 steps) ... | 0.450 s/step"

Usage (from the project root):
    python3 code/37_watch_training.py --exp exp1                  # refresh every 30 s
    python3 code/37_watch_training.py --exp exp1 --interval 60
    python3 code/37_watch_training.py --exp exp1 --once           # show once only
"""
import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE_HEADER = re.compile(r"/\s*(\d+)\s*steps;.*epochs\s*≤\s*(\d+)")
RE_OLD = re.compile(r"^\s*ep (\d+) step (\d+) loss ([-+.\deE]+|nan) ([\d.]+)s/step")
RE_NEW = re.compile(r"^\[([\d\- :]+)\] ep (\d+)/(\d+) step (\d+)/(\d+) .*?loss\(last ?\d+ steps\) ([-+.\deE]+|nan).*?([\d.]+) s/step")
RE_EPOCH = re.compile(r"^epoch (\d+)\s")


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def etime_to_sec(s):
    """Convert ps etime format [[dd-]hh:]mm:ss to seconds."""
    s = s.strip()
    if not s:
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def fmt(sec):
    if sec is None:
        return "unknown"
    sec = max(int(sec), 0)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h} h {m} min" if h else f"{m} min {s} s"


def tail(path, n):
    if not os.path.exists(path):
        return []
    with open(path, "r", errors="replace") as f:
        return f.read().splitlines()[-n:]


def find_procs(exp):
    """Return training processes matching --exp [(pid, etime seconds, command line, is caffeinate)]."""
    out = []
    for line in run(["ps", "-axo", "pid=,etime=,command="]).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or "34_modelB_train.py" not in parts[2]:
            continue
        toks = parts[2].split()

        def arg(flag):
            return toks[toks.index(flag) + 1] if flag in toks and toks.index(flag) + 1 < len(toks) else None

        out_name = os.path.basename(os.path.normpath(arg("--out"))) if arg("--out") else None
        if exp in (arg("--exp"), out_name):
            out.append((int(parts[0]), etime_to_sec(parts[1]), parts[2], toks[0].endswith("caffeinate")))
    return out


def parse_log(lines):
    """Parse the stdout log: steps per epoch, max epochs, last step-level progress line, whether a resume occurred."""
    info = dict(steps_per_epoch=None, max_epochs=None, last=None, last_epoch_done=None)
    for ln in lines:
        m = RE_HEADER.search(ln)
        if m:
            info["steps_per_epoch"], info["max_epochs"] = int(m.group(1)), int(m.group(2))
        m = RE_NEW.search(ln)
        if m:
            info["last"] = dict(fmt="new", time=m.group(1), epoch=int(m.group(2)), max_epochs=int(m.group(3)),
                                step=int(m.group(4)), steps=int(m.group(5)), loss=m.group(6), sec_per_step=float(m.group(7)))
            continue
        m = RE_OLD.search(ln)
        if m:
            info["last"] = dict(fmt="old", epoch=int(m.group(1)), global_step=int(m.group(2)), loss=m.group(3), sec_per_step=float(m.group(4)))
        m = RE_EPOCH.search(ln)
        if m:
            info["last_epoch_done"] = int(m.group(1))
    last = info["last"]
    if last and last["fmt"] == "old" and info["steps_per_epoch"]:
        last["steps"] = info["steps_per_epoch"]
        last["step"] = last["global_step"] - (last["epoch"] - 1) * info["steps_per_epoch"]
    return info


def show(args):
    out = os.path.join(ROOT, "result", "models", args.exp)
    log_csv = os.path.join(out, "train_log.csv")
    log_txt = os.path.join(out, "train_stdout.log")
    now = datetime.now()
    print(f"=== {args.exp} training monitor  {now:%Y-%m-%d %H:%M:%S}  (refresh every {args.interval} s, Ctrl+C quits the monitor, training unaffected) ===\n")

    procs = find_procs(args.exp)
    main = [p for p in procs if not p[3]]
    elapsed = main[0][1] if main else None
    if main:
        print(f"Status: running   PID {main[0][0]}   process running for {fmt(elapsed)}" + ("   (caffeinate prevents sleep)" if any(p[3] for p in procs) else ""))
        if len(main) > 1:
            print(f"!! Note: {len(main)} {args.exp} training processes matched: {[p[0] for p in main]}")
    else:
        print(f"Status: no training process found for {args.exp} (finished, early-stopped, not started or failed; see latest log below)")

    rows, header = [], []
    if os.path.exists(log_csv):
        with open(log_csv, newline="") as f:
            reader = list(csv.reader(f))
        if reader:
            header, rows = reader[0], reader[1:]
    full_lines = tail(log_txt, 100000)
    info = parse_log(full_lines)
    max_ep = args.max_epochs or info["max_epochs"] or 50
    done = len(rows)
    ep_secs = [float(r[header.index("epoch_sec")]) for r in rows if "epoch_sec" in header and r[header.index("epoch_sec")]]
    val_secs = [float(r[header.index("val_sec")]) for r in rows if "val_sec" in header and r[header.index("val_sec")]]
    avg_ep = sum(ep_secs) / len(ep_secs) if ep_secs else None
    avg_val = sum(val_secs) / len(val_secs) if val_secs else 0.0
    total_done_sec = sum(ep_secs)
    print(f"Done: {done} / max {max_ep} epochs" + (f"   mean per epoch {fmt(avg_ep)}   total of finished epochs {fmt(total_done_sec)}" if avg_ep else ""))

    last = info["last"]
    cur_ep = done + 1
    if main and cur_ep <= max_ep:
        # start of this epoch: finished_at of the previous epoch; for epoch 1 use the process start time
        start = None
        if rows and "finished_at" in header:
            try:
                start = datetime.strptime(rows[-1][header.index("finished_at")], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                start = None
        if start is None and elapsed is not None:
            start = datetime.fromtimestamp(time.time() - elapsed)
        ep_used = (now - start).total_seconds() if start else None
        steps_total = (last or {}).get("steps") or info["steps_per_epoch"]
        if last and last.get("epoch") == cur_ep and last.get("step") is not None and steps_total:
            st, sps = last["step"], last["sec_per_step"]
            pct = 100.0 * st / steps_total
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            if avg_ep and ep_used is not None:
                ep_left = max(avg_ep - ep_used, (steps_total - st) * sps + avg_val)
            else:
                ep_left = (steps_total - st) * sps + avg_val
            print(f"Current: epoch {cur_ep} / {max_ep}   step {st} / {steps_total}  [{bar}] {pct:.0f}% (log line every 100 steps)")
            print(f"This epoch: used {fmt(ep_used)}   est. remaining {fmt(ep_left)}" + (" (incl. validation ~%s)" % fmt(avg_val) if avg_val else " (epoch-1 validation time unknown, not included)"))
            per_ep = avg_ep or (steps_total * sps + avg_val)
            all_left = ep_left + (max_ep - cur_ep) * per_ep
            print(f"Overall: est. {fmt(all_left)} left for all {max_ep} epochs (earlier if early-stopped)   last loss {last['loss']}   {sps:.3f} s/step")
        else:
            print(f"Current: epoch {cur_ep} / {max_ep}   epoch elapsed {fmt(ep_used)} (no step-level line yet for this epoch, or validating / saving)")
            if avg_ep:
                print(f"Overall: est. ~{fmt(max(avg_ep - (ep_used or 0), 0) + (max_ep - cur_ep) * avg_ep)} left for all {max_ep} epochs (earlier if early-stopped)")
    if os.path.exists(log_txt):
        age = time.time() - os.path.getmtime(log_txt)
        print(f"Log last updated: {fmt(age)} ago")

    if rows:
        print("\nRecent epochs (up to 10 shown):")
        cols = [c for c in ("epoch", "train_loss", "val_loss", "best_val_loss", "is_best", "bad_epochs", "epoch_sec", "finished_at") if c in header]
        table = [cols] + [[(r[header.index(c)][:10] if c.endswith("loss") else r[header.index(c)]) for c in cols] for r in rows[-10:]]
        widths = [max(len(r[i]) for r in table) for i in range(len(cols))]
        for r in table:
            print("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(cols))))

    lines = full_lines[-args.log_lines:]
    print(f"\nLatest log ({os.path.relpath(log_txt, ROOT)}, last {args.log_lines} lines):")
    for line in lines:
        print("  " + line)
    if any(k in "\n".join(lines) for k in ("Traceback", "Error", " nan", "NaN", "non-finite")):
        print("\n!! Traceback / Error / nan found in the log; please send the output above to the assistant for checking !!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp1")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--max-epochs", type=int, default=0, help="0 = read from the log header (50 if not found)")
    ap.add_argument("--log-lines", type=int, default=8)
    ap.add_argument("--once", action="store_true", help="show once and exit")
    args = ap.parse_args()
    if args.once:
        show(args)
        return
    try:
        while True:
            os.system("clear")
            show(args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor exited; training is still running in the background.")


if __name__ == "__main__":
    main()

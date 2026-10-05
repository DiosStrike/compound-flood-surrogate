#!/usr/bin/env python3
"""40_watch_colab.py -- Experiment 2 (Colab) training monitor, read-only (docs/PLAN.md v10 §7).

Reads train_stdout.log and train_log.csv from the output directory; does not rely on ps, so it can run inside a notebook cell.
Shows: current epoch / step, progress bar, elapsed / estimated remaining for this epoch, total elapsed, estimated remaining to run all epochs, recent per-epoch loss, latest log lines, and how long since the log was updated.
Log format: code/38_exp2_train.py lines "[time] epoch N / M started (this epoch K steps)" and "[time] ep N/M step k/K | ... | x.xxx s/step | ...".

Usage:
  python3 code/40_watch_colab.py --fold matthew --out-root /content/drive/MyDrive/Flood2/models            # show once
  python3 code/40_watch_colab.py --fold matthew --out-root ... --interval 60 --count 30                      # refresh every 60 s, 30 times
  python3 code/40_watch_colab.py --out-dir <any fold directory>
"""
import argparse, csv, os, re, sys, time
from datetime import datetime

RE_START = re.compile(r"^\[([\d\- :]+)\] epoch (\d+) / (\d+) started(?: \(this epoch (\d+) steps\))?")
RE_STEP = re.compile(r"^\[([\d\- :]+)\] ep (\d+)/(\d+) step (\d+)/(\d+) .*?loss\(last ?\d+ steps\) ([-+.\deE]+|nan).*?([\d.]+) s/step")
RE_HEADER = re.compile(r"/\s*(\d+)\s*steps;.*epochs\s*≤\s*(\d+)")
RE_VAL_START = re.compile(r"^\[([\d\- :]+)\] Epoch (\d+) training part finished.*validation started")
RE_END = re.compile(r"^\[([\d\- :]+)\] (training finished|Early stop|finished previously)")
TFMT = "%Y-%m-%d %H:%M:%S"


def fmt(sec):
    if sec is None:
        return "unknown"
    sec = max(int(sec), 0)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h} h {m} min" if h else f"{m} min {s} s"


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", errors="replace") as f:
        return f.read().splitlines()


def show(out_dir, log_lines=6):
    log_txt = os.path.join(out_dir, "train_stdout.log"); log_csv = os.path.join(out_dir, "train_log.csv")
    now = datetime.now()
    print(f"=== Training monitor {out_dir}   {now:%Y-%m-%d %H:%M:%S} ===")
    lines = read_lines(log_txt)
    if not lines:
        print("No log yet (training not started, or --out-dir is wrong)"); return
    steps_per_epoch = max_ep = None; ep_start = {}; last_step = None; val_started = None; ended = None
    for ln in lines:
        m = RE_HEADER.search(ln)
        if m: steps_per_epoch, max_ep = int(m.group(1)), int(m.group(2))
        m = RE_START.search(ln)
        if m:
            ep_start[int(m.group(2))] = datetime.strptime(m.group(1), TFMT); max_ep = int(m.group(3)); ended = None; val_started = None
            if m.group(4): steps_per_epoch = int(m.group(4))
        m = RE_STEP.search(ln)
        if m: last_step = dict(ep=int(m.group(2)), k=int(m.group(4)), n=int(m.group(5)), loss=m.group(6), sps=float(m.group(7)))
        m = RE_VAL_START.search(ln)
        if m: val_started = (int(m.group(2)), datetime.strptime(m.group(1), TFMT))
        m = RE_END.search(ln)
        if m: ended = ln
    rows, header = [], []
    if os.path.exists(log_csv):
        rd = list(csv.reader(open(log_csv, newline="")))
        if rd: header, rows = rd[0], rd[1:]
    col = lambda r, c: r[header.index(c)] if c in header else ""
    ep_secs = [float(col(r, "epoch_sec")) for r in rows if col(r, "epoch_sec")]
    val_secs = [float(col(r, "val_sec")) for r in rows if col(r, "val_sec")]
    avg_ep = sum(ep_secs) / len(ep_secs) if ep_secs else None; avg_val = sum(val_secs) / len(val_secs) if val_secs else 0.0
    done = len(rows); max_ep = max_ep or 50
    age = time.time() - os.path.getmtime(log_txt)
    print(f"Completed: {done} / at most {max_ep} epochs" + (f"   mean per epoch {fmt(avg_ep)}   completed epochs total {fmt(sum(ep_secs))}" if avg_ep else ""))
    if ended:
        print("Status: finished -- " + ended)
    else:
        cur = done + 1; st = ep_start.get(cur)
        ep_used = (now - st).total_seconds() if st else None
        n = steps_per_epoch
        if last_step and last_step["ep"] == cur and n:
            k, sps = last_step["k"], last_step["sps"]
            if val_started and val_started[0] == cur:
                ep_left = max(avg_val - (now - val_started[1]).total_seconds(), 0); stage = "validating"
                k = n
            else:
                ep_left = (n - k) * sps + avg_val; stage = "training"
            pct = 100.0 * k / n; bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            per_ep = avg_ep or (n * sps + avg_val)
            print(f"Current: epoch {cur} / {max_ep} ({stage})  step {k} / {n}  [{bar}] {pct:.0f}%   recent loss {last_step['loss']}   {sps:.3f} s/step")
            print(f"This epoch: elapsed {fmt(ep_used)}   est. remaining {fmt(ep_left)}" + ("" if avg_val else " (first-epoch validation time unknown, not included)"))
            print(f"Overall: total elapsed {fmt(sum(ep_secs) + (ep_used or 0))}   est. time left to run all {max_ep} epochs {fmt(ep_left + (max_ep - cur) * per_ep)} (early stopping will end sooner)")
        else:
            print(f"Current: epoch {cur} / {max_ep}   elapsed this epoch {fmt(ep_used)} (no step-level progress line yet: just started / computing statistics / validating / saving)")
        if age > 600:
            print(f"!! Log not updated for {fmt(age)}: training may have stopped (Colab disconnected?). After a disconnect, restart following the notebook's \"resume after disconnect\" steps.")
    print(f"Log last updated: {fmt(age)} ago")
    if rows:
        cols = [c for c in ("epoch", "train_loss", "val_loss", "best_val_loss", "is_best", "bad_epochs", "epoch_sec", "finished_at") if c in header]
        table = [cols] + [[(col(r, c)[:10] if c.endswith("loss") else col(r, c)) for c in cols] for r in rows[-8:]]
        w = [max(len(r[i]) for r in table) for i in range(len(cols))]
        print("\nRecent epochs:")
        for r in table:
            print("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(cols))))
    print(f"\nLatest log (last {log_lines} lines):")
    for ln in lines[-log_lines:]:
        print("  " + ln)
    if any(k in "\n".join(lines[-50:]) for k in ("Traceback", "Error", "non-finite", "CUDA out of memory")):
        print("\n!! Traceback / Error / non-finite value / out of GPU memory found in the log, please check !!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--fold", default="")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--interval", type=int, default=0, help="0 = show once")
    ap.add_argument("--count", type=int, default=60, help="maximum number of refreshes (when --interval > 0)")
    ap.add_argument("--log-lines", type=int, default=6)
    a = ap.parse_args()
    out = a.out_dir or os.path.join(a.out_root, "exp2_smoke" if a.smoke else "exp2", f"fold_{a.fold}")
    if not a.out_dir and not (a.fold and a.out_root):
        sys.exit("Give --out-dir, or both --fold and --out-root")
    if a.interval <= 0:
        show(out, a.log_lines); return
    try:
        from IPython.display import clear_output
    except Exception:
        clear_output = None
    try:
        for _ in range(a.count):
            if clear_output: clear_output(wait=True)
            else: os.system("clear")
            show(out, a.log_lines); sys.stdout.flush()
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\nMonitor exited; training is not affected.")


if __name__ == "__main__":
    main()

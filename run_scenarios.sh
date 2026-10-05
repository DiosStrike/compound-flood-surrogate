#!/usr/bin/env bash
# run_scenarios.sh [event=irma] [from=1] [to=81]
# Batch-run scenarios: for each scenario in manifest.csv, assemble runs/<event>/<scenario>/ (static grid + that scenario's bzs/dis/ampr/inp), run SFINCS in Docker,
# then run check_run.py and 21_rain_qc.py and write the status back to the manifest (sfincs_run_status / sfincs_version / rain_qc_ratio). Scenarios that already have sfincs_map.nc are skipped.
# Estimate: ~39 s per scenario (basis: measured baseline); 81 scenarios ~55 min.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; EVENT="${1:-irma}"; FROM="${2:-1}"; TO="${3:-81}"
BASE="$ROOT/data/static/sfincs_base"; SCD="$ROOT/data/$EVENT/scenarios"; IMG="deltares/sfincs-cpu"
[ -f "$SCD/manifest.csv" ] || { echo "missing $SCD/manifest.csv"; exit 1; }
LAST="$ROOT/runs/$EVENT/.last_elapsed_sec"; L=$(cat "$LAST" 2>/dev/null || echo 39)
N=$((TO-FROM+1)); echo "Estimated time $((L*N*8/10/60))-$((L*N*15/10/60+1)) min (basis: last measured ${L} s/run x $N runs)"
TALL=$(date +%s)
for i in $(seq -f "%03g" "$FROM" "$TO"); do
  S="${EVENT}_s$i"; RUN="$ROOT/runs/$EVENT/$S"
  if [ -f "$RUN/sfincs_map.nc" ]; then echo "[$S] results exist, skipping"; continue; fi
  mkdir -p "$RUN"
  cp "$BASE"/sfincs.{dep,msk,ind,bnd,src,obs} "$BASE"/check_run.py "$RUN"/
  cp "$SCD/$S"/sfincs.{bzs,dis,ampr,inp} "$RUN"/
  printf 'event=%s\nscenario=%s\nbase_dir=%s\nforcing_dir=%s\ncreated=%s\n' "$EVENT" "$S" "$BASE" "$SCD/$S" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN/RUN_SOURCE.txt"
  T0=$(date +%s); echo "[$S] start $(date '+%H:%M:%S')"
  docker run --rm --platform linux/amd64 -v "$RUN":/data "$IMG" > "$RUN/docker_stdout.txt" 2>&1 || true
  EL=$(( $(date +%s) - T0 )); echo "$EL" > "$LAST"
  if [ -f "$RUN/sfincs.log" ]; then NERR=$(grep -ciE 'error|cannot|fail' "$RUN/sfincs.log" || true); else NERR=99; fi
  VER=$(grep -m1 'Build-Revision' "$RUN/sfincs.log" 2>/dev/null | sed 's/.*Rev: //'; true)
  if [ -f "$RUN/sfincs_map.nc" ] && [ "$NERR" = "0" ]; then ST=ok; else ST=failed; fi
  (cd "$RUN" && python3 check_run.py > check_run_stdout.txt 2>&1) || true
  RATIO=$(python3 "$ROOT/code/21_rain_qc.py" "$S" "$EVENT" 2>/dev/null | python3 -c "import sys,json,re; t=sys.stdin.read(); m=re.search(r'\"mean_ratio\": ([0-9.]+)',t); print(m.group(1) if m else '')")
  python3 - "$SCD/manifest.csv" "$S" "$ST" "$VER" "$RATIO" "$EL" <<'PY'
import sys, pandas as pd
f, s, st, ver, ratio, el = sys.argv[1:]
m = pd.read_csv(f, dtype=str); m.loc[m.scenario == s, ['sfincs_run_status', 'sfincs_version', 'rain_qc_ratio']] = [st, ver, ratio]
m.to_csv(f, index=False)
PY
  echo "[$S] $ST  ${EL}s  log errors $NERR  rain ratio ${RATIO:-n/a}"
done
python3 - "$SCD/manifest.csv" "$ROOT/runs/$EVENT" <<'PY'
# Re-check: any scenario with sfincs_map.nc present and no error in the log is set to ok (fixes 'failed' caused by an early script bug)
import sys, os, re, pandas as pd
f, rd = sys.argv[1:]; m = pd.read_csv(f, dtype=str)
for i, r in m.iterrows():
    d = os.path.join(rd, r.scenario); lg = os.path.join(d, 'sfincs.log')
    if os.path.exists(os.path.join(d, 'sfincs_map.nc')) and os.path.exists(lg):
        n = sum(1 for l in open(lg, errors='replace') if re.search(r'error|cannot|fail', l, re.I))
        m.loc[i, 'sfincs_run_status'] = 'ok' if n == 0 else 'failed'
m.to_csv(f, index=False); print(m.sfincs_run_status.value_counts().to_dict())
PY
echo "All done, total time $(( ($(date +%s)-TALL)/60 )) min"; printf '\n## %s  batch scenarios %s s%s-s%s finished, total time %d min\n' "$(date '+%Y-%m-%d %H:%M')" "$EVENT" "$FROM" "$TO" "$(( ($(date +%s)-TALL)/60 ))" >> "$ROOT/docs/PROGRESS.md"

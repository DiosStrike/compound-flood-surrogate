#!/usr/bin/env bash
# run_sfincs.sh <run_name> [event]
#   run_name : runs/<event>/<run_name>/; not overwritten if it exists (edit its sfincs.inp first, then run)
#   event    : default irma   -- forcing taken from data/<event>/processed/sfincs_forcing/{bzs,dis,ampr,precip}
#   grid is fixed to data/static/sfincs_base (v1 block-averaged terrain, shared by all events)
# What it does: assemble inputs -> print expected runtime -> run SFINCS in Docker (timed) -> check_run.py -> append to docs/PROGRESS.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:-baseline}"; EVENT="${2:-irma}"; BASEV="v1"
BASE="$ROOT/data/static/sfincs_base"
FORC="$ROOT/data/$EVENT/processed/sfincs_forcing"
RUN="$ROOT/runs/$EVENT/$NAME"
IMG="deltares/sfincs-cpu"
LAST="$ROOT/runs/$EVENT/.last_elapsed_sec"
[ -d "$BASE" ] || { echo "Missing base grid directory $BASE"; exit 1; }
[ -d "$FORC" ] || { echo "Missing event forcing directory $FORC"; exit 1; }

mkdir -p "$ROOT/runs/$EVENT"
if [ ! -d "$RUN" ]; then
  mkdir -p "$RUN"
  cp "$BASE"/sfincs.inp "$BASE"/sfincs.dep "$BASE"/sfincs.msk "$BASE"/sfincs.ind \
     "$BASE"/sfincs.bnd "$BASE"/sfincs.src "$BASE"/sfincs.obs "$BASE"/check_run.py "$RUN"/
  cp "$FORC"/sfincs.bzs "$FORC"/sfincs.dis "$FORC"/sfincs.ampr "$FORC"/sfincs.precip "$RUN"/
  printf 'event=%s\nbase=%s\nbase_dir=%s\nforcing_dir=%s\ncreated=%s\n' "$EVENT" "$BASEV" "$BASE" "$FORC" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN/RUN_SOURCE.txt"
  echo "[1/4] Created $RUN  (grid $BASEV + forcing $EVENT)"
else
  echo "[1/4] Using existing $RUN (if inp was edited, the edited version is run)"
fi

if [ -f "$LAST" ]; then
  L=$(cat "$LAST"); LO=$((L*8/10/60)); HI=$((L*15/10/60+1))
  echo "[2/4] Expected runtime ${LO}-${HI} min (basis: last measured $((L/60)) min $((L%60)) s)"
else
  echo "[2/4] Expected runtime 1-5 min (basis: first estimate; 255x195 cells @200 m, 10 days)"
fi
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "      First run needs to pull image $IMG (about 1-3 min)"; docker pull "$IMG"
fi

echo "[3/4] Starting SFINCS  $(date '+%H:%M:%S')   directory $RUN"
T0=$(date +%s)
docker run --rm --platform linux/amd64 -v "$RUN":/data "$IMG" 2>&1 | tee "$RUN/docker_stdout.txt"
T1=$(date +%s); EL=$((T1-T0)); echo "$EL" > "$LAST"
echo "[3/4] Finished  $(date '+%H:%M:%S')   measured runtime $((EL/60)) min $((EL%60)) s"

echo "[4/4] Check"
if [ -f "$RUN/sfincs.log" ]; then
  (cd "$RUN" && python3 check_run.py) || echo "      check_run.py did not complete (usually missing xarray/netCDF4: pip install xarray netCDF4 matplotlib)"
else
  echo "      No sfincs.log produced -- SFINCS did not start, see the docker output above"
fi
{
  echo ""; echo "## $(date '+%Y-%m-%d %H:%M')  SFINCS run: $EVENT/$NAME (base $BASEV)"
  echo "- Directory runs/$EVENT/$NAME; measured runtime $((EL/60)) min $((EL%60)) s"
  echo "- log error lines: $(grep -ciE 'error|cannot|fail' "$RUN/sfincs.log" 2>/dev/null || echo 'n/a')"
  echo "- Outputs: $(ls "$RUN" 2>/dev/null | grep -E 'sfincs_(map|his)\.nc' | tr '\n' ' ')"
  echo "- Next step: see runs/$EVENT/$NAME/check_*.png; python3 code/18_validate_obs.py $NAME $EVENT"
} >> "$ROOT/docs/PROGRESS.md"
echo "Logged to docs/PROGRESS.md"

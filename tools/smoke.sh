#!/bin/zsh
# Automated check of the built app: launches it in the background with a throwaway store,
# types one line through the normal key path (wrong press first), exports CSV + JSON and
# verifies both. Usage: tools/smoke.sh [path/to/Reflex.app]
set -euo pipefail
app="${1:-$HOME/Applications/Reflex.app}"
work="$(mktemp -d "${TMPDIR:-/tmp}/reflex-smoke.XXXXXX")"
report="$work/result.txt"
codesign --verify --strict "$app"
open -g -n "$app" --args --smoke-test "$report"
for _ in {1..30}; do
  [[ -f "$report" ]] && break
  sleep 1
done
if [[ ! -f "$report" ]]; then
  echo "smoke: no result at $report" >&2
  exit 1
fi
cat "$report"
grep -q '^passed=true' "$report"
uv run --no-project python -I - "$work" <<'PY'
import csv, json, sys
from pathlib import Path

work = Path(sys.argv[1])
report = dict(l.split("=", 1) for l in (work / "result.txt").read_text().splitlines() if "=" in l)
with open(work / "export" / "keystrokes.csv", newline="") as f:
    rows = list(csv.DictReader(f))
keys = json.loads((work / "export" / "export.json").read_text())["keystrokes"]
n = int(report["keys"])
assert len(rows) == len(keys) == n, (len(rows), len(keys), n)
# Wrong press first, then the correction at the same position, then the rest in order.
assert rows[0]["correct"] == "0" and rows[1]["correct"] == "1", rows[:2]
assert rows[0]["pos"] == rows[1]["pos"] == "0"
assert [int(r["pos"]) for r in rows[1:]] == list(range(n - 1))
assert "".join(r["typed"] for r in rows[1:]) == report["line"]
assert all(r["line_no"] == "1" and r["status"] == "finished" for r in rows)
for r, k in zip(rows, keys):
    assert r["typed"] == k["typed"] and (r["correct"] == "1") == k["correct"]
print(f"smoke: {n} keystrokes verified in CSV and JSON")
PY
rm -rf "$work"

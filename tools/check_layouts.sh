#!/bin/zsh
# Check Reflex's layout tables against the real macOS layouts (UCKeyTranslate) on this Mac's
# keyboard. Cells marked "-" in a table are not checked. EurKEY is checked only if installed.
# The U.S. table describes ANSI hardware; on ISO hardware the core swaps the keys left of
# "1" and left of "Z" (layout::outputs), and so does this check.
set -euo pipefail
cd "${0:A:h}/.."
work="$(mktemp -d)"
swiftc -O tools/layoutcheck/dump.swift -o "$work/dump"
geo=$([[ "$("$work/dump" --kbd-type)" == iso ]] && echo iso || echo ansi)
us_swap=$([[ $geo == iso ]] && echo swap || echo "")
rc=0
for pair in "de-iso:com.apple.keylayout.German:" "us-ansi:com.apple.keylayout.US:$us_swap" "eurkey:com.apple.keylayout.EurKEY:"; do
  table=${pair%%:*}; rest=${pair#*:}; id=${rest%%:*}; swap=${rest#*:}
  if "$work/dump" "$id" > "$work/$table.tsv" 2>/dev/null; then
    echo "› $table vs $id ($geo keyboard)"
    uv run --no-project python -I tools/layoutcheck/compare.py "core/data/layouts/$table.tsv" "$work/$table.tsv" "$geo" $swap || rc=1
  else
    echo "› $table: $id not installed, skipped"
  fi
done
rm -rf "$work"
exit $rc

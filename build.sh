#!/bin/zsh
# Build Reflex: Rust core → Swift bindings → SwiftUI app bundle in ./build and ~/Applications.
#   ./build.sh            build + install + launch in the background
#   ./build.sh --no-open  build + install only
#   ./build.sh --smoke    build + install, then run the automated app check (tools/smoke.sh)
# Environment: REFLEX_INSTALL_DIR (default ~/Applications), CARGO_HOME (e.g. an offline cache).
set -euo pipefail
OPEN=1 SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN=0 ;;
    --smoke) OPEN=0 SMOKE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
INSTALL_DIR="${REFLEX_INSTALL_DIR:-$HOME/Applications}"
cd "${0:A:h}"
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"

VERSION="0.1.0"
BUILD_NUMBER="1"
APP="build/Reflex.app"
GEN="mac/Sources/ReflexMac/Generated"
FFI="mac/Sources/ReflexCoreFFI"

echo "› core"
(cd core && cargo build --release -p reflex-core --quiet)
(cd core && cargo run --release -p uniffi-bindgen --quiet -- generate \
    --library target/release/libreflex_core.dylib --language swift --out-dir ../build/bindings >/dev/null)
mkdir -p "$GEN"
cp build/bindings/reflex_core.swift "$GEN/ReflexCore.swift"
cp build/bindings/reflex_coreFFI.h "$FFI/reflex_coreFFI.h"
cat > "$FFI/module.modulemap" <<'MAP'
module reflex_coreFFI {
    header "reflex_coreFFI.h"
    export *
}
MAP
# Link statically: hide the dylib from the linker while building the app.
mv core/target/release/libreflex_core.dylib core/target/release/libreflex_core.dylib.off

echo "› app"
(cd mac && swift build -c release --quiet) || { mv core/target/release/libreflex_core.dylib.off core/target/release/libreflex_core.dylib; exit 1; }
mv core/target/release/libreflex_core.dylib.off core/target/release/libreflex_core.dylib

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp mac/.build/release/ReflexMac "$APP/Contents/MacOS/Reflex"
[[ -f mac/Resources/AppIcon.icns ]] && cp mac/Resources/AppIcon.icns "$APP/Contents/Resources/"
[[ -d core/data/models ]] && cp -R core/data/models "$APP/Contents/Resources/models"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Reflex</string>
  <key>CFBundleDisplayName</key><string>Reflex</string>
  <key>CFBundleIdentifier</key><string>com.hsxcix.reflex</string>
  <key>CFBundleExecutable</key><string>Reflex</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$BUILD_NUMBER</string>
  <key>LSMinimumSystemVersion</key><string>26.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.education</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$APP" >/dev/null
codesign --verify --strict "$APP"

pkill -x Reflex 2>/dev/null || true
mkdir -p "$INSTALL_DIR"
rm -rf "$INSTALL_DIR/Reflex.app"
cp -R "$APP" "$INSTALL_DIR/"
# Drop the build copy: only the installed app should show up in Spotlight / Launchpad.
rm -rf "$APP"
echo "› installed $INSTALL_DIR/Reflex.app"
(( SMOKE )) && tools/smoke.sh "$INSTALL_DIR/Reflex.app"
(( OPEN )) && open -g "$INSTALL_DIR/Reflex.app"
exit 0

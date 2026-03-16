#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP_DIR="${HOME}/Desktop"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TERMINAL_BIN="${TERMINAL_BIN:-gnome-terminal}"
DEMO_ICON_PATH="$PROJECT_DIR/static/system180-desktop-icon.png"

mkdir -p "$DESKTOP_DIR"

RUN_NORMAL_CMD="cd \"$PROJECT_DIR\" && demonstrator --mode normal"
RUN_GAME_CMD="cd \"$PROJECT_DIR\" && bash \"$PROJECT_DIR/tools/launch_game_kiosk.sh\""
UPDATE_CMD="cd \"$PROJECT_DIR\" && \"$PROJECT_DIR/tools/update_local_install.sh\""
CONVERT_CMD="cd \"$PROJECT_DIR\" && bash \"$PROJECT_DIR/tools/convert_models.sh\""

cat > "$DESKTOP_DIR/System180 Demonstrator.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=System180 Demonstrator
Comment=Start the System180 demonstrator in normal mode
Exec=$TERMINAL_BIN -- bash -lc '$RUN_NORMAL_CMD; EXIT_CODE=\$?; echo; echo "Press Enter to close..."; read'
Icon=$DEMO_ICON_PATH
Terminal=false
StartupNotify=true
Categories=Utility;Development;
Path=$PROJECT_DIR
EOF

cat > "$DESKTOP_DIR/System180 Game.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=System180 Game
Comment=Start the System180 demonstrator in game mode
Exec=$TERMINAL_BIN -- bash -lc '$RUN_GAME_CMD; EXIT_CODE=\$?; echo; echo "Press Enter to close..."; read'
Icon=applications-games
Terminal=false
StartupNotify=true
Categories=Utility;Development;
Path=$PROJECT_DIR
EOF

cat > "$DESKTOP_DIR/System180 Update.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=System180 Update
Comment=Update the local System180 demonstrator checkout and reinstall it
Exec=$TERMINAL_BIN -- bash -lc '$UPDATE_CMD; EXIT_CODE=\$?; echo; echo "Press Enter to close..."; read'
Icon=view-refresh
Terminal=false
StartupNotify=true
Categories=Utility;Development;
Path=$PROJECT_DIR
EOF

cat > "$DESKTOP_DIR/System180 Convert Models.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=System180 Convert Models
Comment=Convert YOLO source models into local TensorRT engines
Exec=$TERMINAL_BIN -- bash -lc '$CONVERT_CMD; EXIT_CODE=\$?; echo; echo "Press Enter to close..."; read'
Icon=applications-engineering
Terminal=false
StartupNotify=true
Categories=Utility;Development;
Path=$PROJECT_DIR
EOF

chmod +x \
    "$PROJECT_DIR/tools/update_local_install.sh" \
    "$PROJECT_DIR/tools/install_desktop_icons.sh" \
    "$PROJECT_DIR/tools/convert_models.sh" \
    "$PROJECT_DIR/tools/launch_game_kiosk.sh" \
    "$DESKTOP_DIR/System180 Demonstrator.desktop" \
    "$DESKTOP_DIR/System180 Game.desktop" \
    "$DESKTOP_DIR/System180 Update.desktop" \
    "$DESKTOP_DIR/System180 Convert Models.desktop"

if command -v gio >/dev/null 2>&1; then
    gio set "$DESKTOP_DIR/System180 Demonstrator.desktop" metadata::trusted true 2>/dev/null || true
    gio set "$DESKTOP_DIR/System180 Game.desktop" metadata::trusted true 2>/dev/null || true
    gio set "$DESKTOP_DIR/System180 Update.desktop" metadata::trusted true 2>/dev/null || true
    gio set "$DESKTOP_DIR/System180 Convert Models.desktop" metadata::trusted true 2>/dev/null || true
fi

rm -f \
    "$DESKTOP_DIR/System180 Demonstrator (Game).desktop" \
    "$DESKTOP_DIR/System180 Demonstrator Update.desktop"

echo "Desktop launchers installed to $DESKTOP_DIR"
echo "- System180 Demonstrator"
echo "- System180 Game"
echo "- System180 Update"
echo "- System180 Convert Models"

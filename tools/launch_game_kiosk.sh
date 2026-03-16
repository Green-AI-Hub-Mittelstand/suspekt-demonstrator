#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

demonstrator --mode game &
SERVER_PID=$!

URL="${SYSTEM180_GAME_URL:-http://127.0.0.1:8001}"

echo "Starting System180 Game server (PID: $SERVER_PID)..."
echo "Waiting for $URL ..."

# Wait up to ~60s for the server port to accept connections.
for _ in {1..60}; do
    if (echo > /dev/tcp/127.0.0.1/8001) >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if command -v firefox >/dev/null 2>&1; then
    firefox --kiosk "$URL" >/dev/null 2>&1 &
elif command -v chromium-browser >/dev/null 2>&1; then
    chromium-browser --kiosk --start-fullscreen "$URL" >/dev/null 2>&1 &
elif command -v google-chrome >/dev/null 2>&1; then
    google-chrome --kiosk --start-fullscreen "$URL" >/dev/null 2>&1 &
else
    echo "No supported browser found (firefox/chromium-browser/google-chrome)."
    echo "Open $URL manually."
fi

echo "Browser launch triggered. Close this terminal to stop the server."


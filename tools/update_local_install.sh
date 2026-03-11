#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "System180 update helper"
echo "Project: $PROJECT_DIR"
echo

cd "$PROJECT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "This directory is not a git repository."
    exit 1
fi

CURRENT_HEAD="$(git rev-parse HEAD)"

echo "Fetching latest changes..."
git fetch --all --prune

UPSTREAM_REF=""
if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    UPSTREAM_REF="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}')"
    UPSTREAM_HEAD="$(git rev-parse '@{u}')"
else
    echo "No upstream branch configured. Falling back to git pull."
    git pull
    NEW_HEAD="$(git rev-parse HEAD)"
    if [[ "$NEW_HEAD" == "$CURRENT_HEAD" ]]; then
        echo "No updates found."
        exit 0
    fi
    echo "Installing updated package..."
    python3 -m pip install -e . --user
    echo "Update complete."
    exit 0
fi

if [[ "$CURRENT_HEAD" == "$UPSTREAM_HEAD" ]]; then
    echo "No updates found."
    exit 0
fi

echo "Updates available from $UPSTREAM_REF."
git pull --ff-only

echo "Installing updated package..."
python3 -m pip install -e . --user

echo "Update complete."

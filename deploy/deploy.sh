#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/evelyn/complex"
PYTHON="/usr/local/bin/python3.14"

cd "$APP_DIR"

sudo "$PYTHON" -m pip install --upgrade pip wheel --quiet --root-user-action=ignore --break-system-packages
if [[ -f requirements.txt ]]; then
  sudo "$PYTHON" -m pip install -r requirements.txt --quiet --root-user-action=ignore --break-system-packages
elif [[ -f pyproject.toml ]]; then
  sudo "$PYTHON" -m pip install . --quiet --root-user-action=ignore --break-system-packages
fi

sudo nginx -t
sudo systemctl reload nginx
sudo systemctl daemon-reload

sudo systemctl enable --now complex
sudo systemctl restart complex

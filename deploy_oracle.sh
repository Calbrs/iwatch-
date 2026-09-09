#!/usr/bin/env bash
# deploy_oracle.sh - one-shot deploy of the Clinic Doctor Time Tracker on an
# Oracle Cloud Always Free (Ampere A1 / ARM64) VM.
#
# Run (on the VM, after SSH):
#   bash deploy_oracle.sh
#
# Result: a systemd service "iwatch-tracker" serving tracker.py on :5000.
# NOTE: you must also open TCP 5000 in the OCI Security List
# (VCN -> Public Subnet -> Security List -> Ingress Rule) or the port stays
# unreachable from the internet even though the app runs.
# HTTPS for phones (getUserMedia needs a secure context) is a separate step -
# see the README "Oracle Cloud (free ARM VM)" section.
set -euo pipefail

APP_DIR="$HOME/iwatch"
REPO_URL="https://github.com/Calbrs/iwatch-"

echo "==> Detecting OS"
if command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="apt-get"
elif command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
else
    echo "Unsupported OS - need apt-get (Ubuntu) or dnf (Oracle Linux)."; exit 1
fi

echo "==> Installing prerequisites (git, python3-pip, python3-venv)"
if [ "$PKG_MGR" = "apt-get" ]; then
    sudo apt-get update -y
    sudo apt-get install -y git python3-pip python3-venv
else
    sudo dnf install -y git python3-pip python3-virtualenv
fi

echo "==> Clone/update the repo"
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$APP_DIR"
fi

echo "==> Install Python deps into a virtualenv (ARM64 wheels)"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
"$APP_DIR/.venv/bin/pip" install --upgrade opencv-python-headless

echo "==> Open firewall port 5000 (still required: OCI Security List rule!)"
if command -v firewall-cmd >/dev/null 2>&1; then
    sudo firewall-cmd --permanent --add-port=5000/tcp || true
    sudo firewall-cmd --reload || true
elif command -v ufw >/dev/null 2>&1; then
    sudo ufw allow 5000/tcp || true
fi
sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT 2>/dev/null || true
if command -v netfilter-persistent >/dev/null 2>&1; then
    sudo netfilter-persistent save || true
fi

echo "==> Install systemd service (auto-start + auto-restart)"
sudo tee /etc/systemd/system/iwatch-tracker.service >/dev/null <<EOF
[Unit]
Description=Clinic Doctor Time Tracker
After=network.target

[Service]
User=$(id -un)
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/python tracker.py
Environment=PORT=5000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable iwatch-tracker
sudo systemctl restart iwatch-tracker
sleep 2
sudo systemctl status iwatch-tracker --no-pager || true

echo ""
echo "==> DONE. Tracker is running on:"
echo "    http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "    If 0.0.0.0:5000 shows here but the URL times out from outside, open"
echo "    TCP 5000 in the OCI Security List (VCN -> Public Subnet -> Security"
echo "    List -> Add Ingress Rule: source 0.0.0.0/0, port 5000)."
echo ""
echo "    Phones need HTTPS next (trycloudflare tunnel) - see the README."
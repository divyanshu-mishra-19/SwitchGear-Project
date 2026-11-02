#!/usr/bin/env bash
# Render Build Script for CircuitGuard AI
set -e

echo "=== CircuitGuard AI: Starting Render Build ==="

# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install lightweight CPU PyTorch on Linux to save memory and disk
echo "Installing PyTorch CPU..."
pip install --no-cache-dir torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining Python dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# 4. Build React frontend if not already present or if node is installed
if command -v npm &> /dev/null; then
    echo "Node.js detected. Building frontend..."
    cd frontend
    npm install
    npm run build
    cd ..
else
    echo "Using pre-built frontend in frontend/dist/"
fi

echo "=== Build Complete! Ready for Deployment ==="


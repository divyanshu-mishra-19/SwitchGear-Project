#!/usr/bin/env bash
# CircuitGuard AI - Master Execution Script
# Launches FastAPI backend + React frontend dashboard

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=True

# 1. Detect the Python interpreter that has the required ML libraries (pytorch_tabnet, torch, fastapi)
PYTHON_BIN=""
if [ -x "/opt/anaconda3/bin/python3" ] && /opt/anaconda3/bin/python3 -c "import pytorch_tabnet, fastapi, uvicorn" >/dev/null 2>&1; then
    PYTHON_BIN="/opt/anaconda3/bin/python3"
elif python3 -c "import pytorch_tabnet, fastapi, uvicorn" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif python -c "import pytorch_tabnet, fastapi, uvicorn" >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif [ -x "/opt/anaconda3/bin/python" ]; then
    PYTHON_BIN="/opt/anaconda3/bin/python"
else
    PYTHON_BIN="python3"
fi

ACTION="${1:-all}"

# Function to clean up lingering processes on a port
kill_port() {
  local port=$1
  local pids=$(lsof -ti :$port 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
}

case "$ACTION" in
    all|dev|start)
        echo "============================================================"
        echo "⚡ CircuitGuard AI - Starting System (Backend + React)"
        echo "============================================================"
        echo "Using Python: $PYTHON_BIN"

        kill_port 8000
        kill_port 3000

        # Ensure node modules exist
        if [ ! -d "frontend/node_modules" ]; then
            echo "Installing React frontend dependencies..."
            (cd frontend && npm install)
        fi

        # Start FastAPI backend in background using detected Python
        echo "1. Starting FastAPI backend on http://127.0.0.1:8000..."
        $PYTHON_BIN -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > /tmp/circuitguard_backend.log 2>&1 &
        BACKEND_PID=$!

        # Cleanup backend when script terminates
        cleanup() {
            echo ""
            echo "Shutting down CircuitGuard AI..."
            kill $BACKEND_PID 2>/dev/null || true
            kill_port 8000
            kill_port 3000
            echo "Shutdown complete."
            exit 0
        }
        trap cleanup EXIT INT TERM

        # Wait up to 6 seconds for backend to be healthy
        echo "2. Loading ML models into memory..."
        for i in {1..12}; do
            if curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
                echo "   Backend healthy and all models loaded!"
                break
            fi
            sleep 0.5
        done

        echo ""
        echo "============================================================"
        echo "🚀 React Dashboard: http://localhost:3000"
        echo "📖 Backend API Docs: http://localhost:8000/docs"
        echo "============================================================"
        echo "(Opening dashboard in browser... Press Ctrl+C to stop)"
        echo ""

        # Open browser automatically on macOS
        if command -v open >/dev/null 2>&1; then
            (sleep 1 && open "http://localhost:3000") &
        fi

        # Start Vite React frontend in foreground
        cd frontend
        npm run dev -- --host 127.0.0.1 --port 3000
        ;;

    pipeline)
        echo "=== Running Full ML Pipeline from Scratch ==="
        $PYTHON_BIN training/inspect_dataset.py
        $PYTHON_BIN training/preprocess.py
        $PYTHON_BIN training/train.py
        $PYTHON_BIN training/evaluate.py
        $PYTHON_BIN training/create_scenarios.py
        echo "=== Rebuilding React Production Assets ==="
        (cd frontend && npm run build)
        echo "=== Pipeline Completed Successfully! ==="
        ;;

    backend)
        echo "=== Starting FastAPI Backend Only on http://127.0.0.1:8000 ==="
        kill_port 8000
        $PYTHON_BIN -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
        ;;

    help|-h|--help)
        echo "Usage: ./run.sh [all | backend | pipeline]"
        echo ""
        echo "Commands:"
        echo "  ./run.sh          : Start FastAPI backend + React dashboard on http://localhost:3000 (Default)"
        echo "  ./run.sh backend  : Start FastAPI backend only on http://localhost:8000"
        echo "  ./run.sh pipeline : Re-run entire ML training pipeline and rebuild React"
        ;;

    *)
        echo "Unknown option: '$1'"
        echo "Run './run.sh help' for available commands."
        exit 1
        ;;
esac

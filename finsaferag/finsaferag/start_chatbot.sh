#!/usr/bin/env bash

set -u

echo "=================================================="
echo "   Federated RAG Chatbot with Privacy Protection"
echo "=================================================="
echo ""

# Determine script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

echo "✓ Python: $(python3 --version)"
echo ""

# Check project structure
if [ ! -f "config.toml" ]; then
    echo "❌ config.toml not found. Run from project root"
    exit 1
fi

# Setup logs
mkdir -p logs
echo "✓ Logs directory ready"
echo ""

FLOWER_PID=""
STREAMLIT_PID=""

cleanup() {
    echo ""
    echo "=================================================="
    echo "   Shutting down..."
    echo "=================================================="
    echo ""

    if [ -n "$STREAMLIT_PID" ]; then
        kill "$STREAMLIT_PID" 2>/dev/null && echo "✓ Stopped Streamlit" || true
    fi

    if [ -n "$FLOWER_PID" ]; then
        kill "$FLOWER_PID" 2>/dev/null && echo "✓ Stopped Flower" || true
    fi

    pkill -f "flwr run" 2>/dev/null || true
    pkill -f "streamlit" 2>/dev/null || true

    echo ""
    echo "✓ All services stopped"
    echo ""
}

trap cleanup INT TERM

# ============================================================
# Start Flower
# ============================================================
echo "🌸 Starting Flower Federated Server..."
echo "=================================================="

flwr run . > logs/flower.log 2>&1 &
FLOWER_PID=$!

echo "✓ Flower started (PID: $FLOWER_PID)"
echo "  Logs: logs/flower.log"
echo ""
echo "⏳ Waiting for server & clients..."

max_wait=40
waited=0

while [ $waited -lt $max_wait ]; do
    if grep -q "Federated RAG bridge initialized\|Server ready" logs/flower.log 2>/dev/null; then
        echo "✓ Flower ready!"
        sleep 2
        break
    fi
    sleep 1
    ((waited++))
done

echo ""

# ============================================================
# Start Streamlit
# ============================================================
echo "🎨 Starting Streamlit UI..."
echo "=================================================="
echo ""
echo "✅ Open: http://localhost:8501"
echo ""
echo "Shortcuts:"
echo "  Ctrl+C: Stop all"
echo ""
echo "Logs: logs/flower.log"
echo ""
echo "=================================================="
echo ""

cd "$SCRIPT_DIR/ui"
streamlit run streamlit_app.py

cleanup

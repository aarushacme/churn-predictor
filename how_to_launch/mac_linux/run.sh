#!/bin/bash
# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/../../frontend" ]; then
    cd "$SCRIPT_DIR/../.."
elif [ -d "$SCRIPT_DIR/../frontend" ]; then
    cd "$SCRIPT_DIR/.."
else
    cd "$SCRIPT_DIR"
fi

echo "===================================================="
echo "Starting Telecom Customer Churn Predictor..."
echo "Project Folder: $(pwd)"
echo "===================================================="
echo ""

# Check Python executable
if command -v python3 &>/dev/null; then
    PY_CMD=python3
elif command -v python &>/dev/null; then
    PY_CMD=python
else
    echo "[ERROR] Python is not installed or not in PATH."
    exit 1
fi

echo "Step 1: Installing required packages..."
$PY_CMD -m pip install -r requirements.txt || pip install -r requirements.txt
echo ""
echo "Step 2: Launching Streamlit frontend..."
echo "Application will open in your browser at http://localhost:8501"
$PY_CMD -m streamlit run frontend/app.py || streamlit run frontend/app.py

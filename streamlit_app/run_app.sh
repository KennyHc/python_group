#!/bin/bash
# Run script for Bike Sharing Streamlit Dashboard

echo "🚴 Starting Bike Sharing Dashboard..."
echo "=================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please run this script from the streamlit_app directory"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit is not installed"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Run setup check
echo "🔍 Checking setup..."
python check_setup.py

# If setup check passed, run the app
if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Launching Streamlit app..."
    echo "The app will open in your browser automatically"
    echo "If not, navigate to: http://localhost:8501"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "=================================="
    streamlit run app.py
else
    echo "❌ Setup check failed. Please fix the issues before running the app."
    exit 1
fi

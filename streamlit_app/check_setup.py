#!/usr/bin/env python3
"""
Setup verification script for Bike Sharing Streamlit Dashboard
This script checks if all required files are present before running the app.
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    
    print("🔍 Checking Bike Sharing Dashboard Setup...\n")
    
    # Define required files
    required_data_files = [
        'data/bike_data_processed.csv',
        'data/bike_data_original.csv',
        'data/model_results.csv',
        'data/tuned_model_results.csv',
        'data/predictions.csv',
        'data/summary_stats.csv',
        'data/feature_names.csv'
    ]
    
    optional_data_files = [
        'data/feature_importance.csv'
    ]
    
    required_model_files = [
        'models/best_model.pkl'
    ]
    
    required_page_files = [
        'pages/1_📊_Data_Overview.py',
        'pages/2_🔍_Exploratory_Analysis.py',
        'pages/3_🤖_Model_Performance.py',
        'pages/4_🔮_Predictions.py',
        'pages/5_💡_Recommendations.py'
    ]
    
    all_good = True
    
    # Check main app file
    if os.path.exists('app.py'):
        print("✅ Main app file found: app.py")
    else:
        print("❌ Main app file missing: app.py")
        all_good = False
    
    print("\n📁 Checking data files:")
    for file in required_data_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            all_good = False
    
    print("\n📁 Checking optional data files:")
    for file in optional_data_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  {file} - Optional, not found")
    
    print("\n🤖 Checking model files:")
    for file in required_model_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            all_good = False
    
    print("\n📄 Checking page files:")
    for file in required_page_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            all_good = False
    
    print("\n" + "="*50)
    
    if all_good:
        print("✅ All required files found! You're ready to run the app.")
        print("\nTo start the dashboard, run:")
        print("  streamlit run app.py")
    else:
        print("❌ Some required files are missing!")
        print("\nPlease ensure you have:")
        print("1. Run the Jupyter notebook (Group0_Notebook_Assignment2.ipynb)")
        print("2. Executed the export cell at the end of the notebook")
        print("3. Copied all files to the correct directories")
        print("\nRefer to README.md for detailed setup instructions.")
        return False
    
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    
    print("\n\n📦 Checking Python dependencies...\n")
    
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\nOr install all dependencies with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

if __name__ == "__main__":
    print("="*50)
    print("🚴 Bike Sharing Dashboard Setup Checker")
    print("="*50)
    
    files_ok = check_files()
    deps_ok = check_dependencies()
    
    print("\n" + "="*50)
    
    if files_ok and deps_ok:
        print("🎉 Everything looks good! Your dashboard is ready to run.")
        print("\nStart the dashboard with:")
        print("  streamlit run app.py")
        print("\nThe dashboard will open in your browser at http://localhost:8501")
    else:
        print("⚠️  Please fix the issues above before running the dashboard.")
        sys.exit(1)

In the `streamlit_app/` folder we have.

1. **Main Application Files:**
   - `app.py` - Main dashboard with overview and KPIs
   - `requirements.txt` - Python dependencies
   - `README.md` - Comprehensive documentation

2. **Dashboard Pages (in `pages/` folder):**
   - `1_Data_Overview.py` - Data quality and statistics
   - `2_Exploratory_Analysis.py` - Interactive EDA
   - `3_Model_Performance.py` - Model comparison
   - `4_Predictions.py` - Real-time predictions
   - `5_Recommendations.py` - Business insights

3. **Data Structure (empty folders):**
   - `data/` - For your exported CSV files
   - `models/` - For your saved model files

### Quick Start Steps:

1. **Run the Jupyter Notebook**, such that the CSV files are produced.

2. **Navigate to the app folder:**
```bash
cd streamlit_app
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the app:**
```bash
streamlit run app.py
```

### Features Implemented:

 **Multi-page navigation** with sidebar
 **Interactive Plotly charts** throughout
 **Real-time predictions** with sliders and inputs
 **Batch prediction** via CSV upload
 **What-if scenarios** for business planning
 **Comprehensive business recommendations**
 **Model performance comparison**
 **Feature importance visualization**
 **Weather and temporal analysis**
 **User segmentation insights**

###  Notes:

- All visualizations are interactive (zoom, pan, hover)
- The predictions page includes scenario analysis
- Business recommendations are data-driven
- Color coding helps identify patterns quickly

### Troubleshooting:

If you encounter issues:
1. Ensure all CSV exports completed successfully
2. Check that model pickle files are compatible
3. Use virtual environment to avoid conflicts

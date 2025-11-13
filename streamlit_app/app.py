import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Bike Sharing Demand Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main Header: Bold, modern font with a slight shadow for depth */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-color); /* Adapts to dark/light mode */
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Metric Card: Glassmorphism effect with borders and shadows */
    .metric-card {
        background-color: var(--secondary-background-color); /* Use Streamlit's secondary bg */
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle drop shadow */
        border: 1px solid rgba(128, 128, 128, 0.2); /* Subtle border for definition */
        transition: transform 0.2s; /* Smooth hover effect */
    }
    
    /* Hover effect for interactivity */
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }

    /* Insight Box: Uses semi-transparent background for readability in both modes */
    .insight-box {
        background-color: rgba(31, 119, 180, 0.15); /* Blue tint that works on dark & light */
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4; /* Solid accent line */
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_summary_stats():
    return pd.read_csv('data/summary_stats.csv')

@st.cache_data
def load_processed_data():
    return pd.read_csv('data/bike_data_processed.csv')

# Main page content
st.markdown('<h1 class="main-header">🚴 Bike Sharing Demand Analysis Dashboard</h1>', unsafe_allow_html=True)

# Load data
try:
    stats = load_summary_stats()
    df = load_processed_data()
    
    # Convert date column
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    # Key Metrics Section
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records Analyzed",
            value=f"{stats['total_records'].values[0]:,}",
            delta="2 years of hourly data"
        )
    
    with col2:
        st.metric(
            label="Average Hourly Demand",
            value=f"{stats['avg_demand'].values[0]:.0f} bikes",
            delta=f"Peak: {stats['peak_demand'].values[0]:.0f} bikes"
        )
    
    with col3:
        st.metric(
            label="Best Model Performance",
            value=f"R² = {stats['best_r2'].values[0]:.3f}",
            delta=f"RMSE: {stats['best_rmse'].values[0]:.1f} bikes"
        )
    
    with col4:
        best_model = stats['best_model_name'].values[0]
        st.metric(
            label="Selected Model",
            value=best_model,
            delta="After hyperparameter tuning"
        )
    
    # Visual Overview
    st.markdown("---")
    st.markdown("## 📈 Demand Trends Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily demand trend
        daily_demand = df.groupby(df['dteday'].dt.date)['cnt'].sum().reset_index()
        daily_demand.columns = ['date', 'total_rentals']
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_demand['date'],
            y=daily_demand['total_rentals'],
            mode='lines',
            name='Daily Rentals',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_daily.update_layout(
            title="Daily Bike Rentals Over Time",
            xaxis_title="Date",
            yaxis_title="Total Rentals",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        # Hourly pattern
        hourly_avg = df.groupby('hr')['cnt'].mean().reset_index()
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=hourly_avg['hr'],
            y=hourly_avg['cnt'],
            name='Average Rentals',
            marker_color='#ff7f0e'
        ))
        
        fig_hourly.update_layout(
            title="Average Hourly Demand Pattern",
            xaxis_title="Hour of Day",
            yaxis_title="Average Rentals",
            height=400
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Key Insights Section
    st.markdown("---")
    st.markdown("## 💡 Key Insights from Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🕐 Time Patterns</h4>
        <ul>
        <li>Rush hours (8-9 AM, 5-6 PM) show peak demand</li>
        <li>Weekend patterns differ significantly</li>
        <li>Year-over-year growth: 64.9%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>🌡️ Weather Impact</h4>
        <ul>
        <li>Temperature is 2nd most important factor</li>
        <li>Optimal range: 20-25°C (normalized)</li>
        <li>Bad weather reduces demand by ~66%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Model Accuracy</h4>
        <ul>
        <li>Explains 88.9% of demand variation</li>
        <li>Average error: ±67 bikes/hour</li>
        <li>Suitable for capacity planning</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation Guide
    st.markdown("---")
    st.markdown("## 🗺️ Dashboard Navigation Guide")
    
    st.info("""
    **Explore the complete analysis using the sidebar navigation:**
    
    1. **📊 Data Overview** - Examine data quality, missing values, and basic statistics
    2. **🔍 Exploratory Analysis** - Interactive visualizations of demand patterns and relationships
    3. **🤖 Model Performance** - Compare different models and see detailed performance metrics
    4. **🔮 Predictions** - Make real-time predictions with custom inputs
    5. **💡 Recommendations** - Business insights and actionable recommendations
    """)
    
    # Year comparison
    st.markdown("---")
    st.markdown("## 📅 Year-over-Year Comparison")
    
    # Calculate yearly stats
    df['year'] = df['yr'].map({0: '2011', 1: '2012'})
    yearly_stats = df.groupby('year').agg({
        'cnt': ['sum', 'mean', 'std', 'max'],
        'casual': 'sum',
        'registered': 'sum'
    }).round(0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_year_total = go.Figure()
        
        years = ['2011', '2012']
        totals = [yearly_stats.loc['2011', ('cnt', 'sum')], 
                  yearly_stats.loc['2012', ('cnt', 'sum')]]
        
        fig_year_total.add_trace(go.Bar(
            x=years,
            y=totals,
            text=[f'{int(t):,}' for t in totals],
            textposition='auto',
            marker_color=['#1f77b4', '#ff7f0e']
        ))
        
        fig_year_total.update_layout(
            title="Total Annual Rentals",
            yaxis_title="Total Rentals",
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig_year_total, use_container_width=True)
    
    with col2:
        # User type breakdown
        fig_user_type = go.Figure()
        
        categories = ['Casual', 'Registered']
        
        fig_user_type.add_trace(go.Bar(
            name='2011',
            x=categories,
            y=[yearly_stats.loc['2011', ('casual', 'sum')],
               yearly_stats.loc['2011', ('registered', 'sum')]],
            marker_color='#1f77b4'
        ))
        
        fig_user_type.add_trace(go.Bar(
            name='2012',
            x=categories,
            y=[yearly_stats.loc['2012', ('casual', 'sum')],
               yearly_stats.loc['2012', ('registered', 'sum')]],
            marker_color='#ff7f0e'
        ))
        
        fig_user_type.update_layout(
            title="User Type Distribution by Year",
            yaxis_title="Total Rentals",
            barmode='group',
            height=350
        )
        
        st.plotly_chart(fig_user_type, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>Dashboard created with Streamlit | Data: Capital Bikeshare System (2011-2012)</p>
    </div>
    """, unsafe_allow_html=True)
    
except FileNotFoundError:
    st.error("""
    ⚠️ **Data files not found!**
    
    Please make sure you have exported the data from your Jupyter notebook by running the export code.
    The app expects to find data files in the `data/` subdirectory.
    """)
    
    st.info("""
    **Required files:**
    - `data/summary_stats.csv`
    - `data/bike_data_processed.csv`
    - `data/model_results.csv`
    - `data/predictions.csv`
    - `models/best_model.pkl`
    """)
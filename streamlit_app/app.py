import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
from utils import load_css


st.set_page_config(
    page_title="Bike Sharing Demand Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

script_dir = Path(__file__).parent

@st.cache_data
def load_summary_stats():
    """Loads the summary statistics file."""
    try:
        path_summary = script_dir / 'data/summary_stats.csv'
        return pd.read_csv(path_summary)
    except FileNotFoundError:
        st.error("Error: The summary statistics file ('summary_stats.csv') was not found.", icon=":material/error:")
        st.info("Please check the 'data' folder in your repo.", icon=":material/info:")
        return None
    except Exception as e:
        st.error(f"An error occurred loading summary stats: {e}", icon=":material/error:")
        return None

@st.cache_data
def load_processed_data():
    """Loads the processed dataset."""
    try:
        path_processed = script_dir / 'data/bike_data_processed.csv'
        return pd.read_csv(path_processed)
    except FileNotFoundError:
        st.error("Error: The processed data file ('bike_data_processed.csv') was not found.", icon=":material/error:")
        st.info("Please check the 'data' folder in your repo.", icon=":material/info:")
        return None
    except Exception as e:
        st.error(f"An error occurred loading processed data: {e}", icon=":material/error:")
        return None

# --- Main Page Content ---

st.markdown('<h1 class="main-header">Bike Sharing Demand Analysis</h1>', unsafe_allow_html=True)

# Image loading with prompt tag
image_path = script_dir / 'bikes.jpg'



try:
    st.image(str(image_path), 
            caption="Strategic Analysis for Department of Transportation Services", 
            use_column_width=True) 
except FileNotFoundError:
    st.error("Header image not found.", icon=":material/broken_image:")


# Load data
try:
    stats = load_summary_stats()
    df = load_processed_data()
    
    # Convert date column
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    # Key Metrics Section
    st.markdown("## :material/monitoring: Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=":material/dataset: Total Records",
            value=f"{stats['total_records'].values[0]:,}",
            delta="2 years of hourly data"
        )
    
    with col2:
        st.metric(
            label=":material/schedule: Avg Hourly Demand",
            value=f"{stats['avg_demand'].values[0]:.0f} bikes",
            delta=f"Peak: {stats['peak_demand'].values[0]:.0f} bikes"
        )
    
    with col3:
        st.metric(
            label=":material/model_training: Best Model R²",
            value=f"{stats['best_r2'].values[0]:.3f}",
            delta=f"RMSE: {stats['best_rmse'].values[0]:.1f} bikes"
        )
    
    with col4:
        best_model = stats['best_model_name'].values[0]
        st.metric(
            label=":material/check_circle: Selected Model",
            value=best_model,
            delta="Optimized Gradient Boosting"
        )
    
    # Visual Overview
    st.divider()
    st.markdown("## :material/timeline: Demand Trends Overview")
    
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
    st.divider()
    st.markdown("## :material/lightbulb: Key Insights from Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    # Using HTML for Custom Cards with Material Symbols
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4><span class="material-symbols-rounded">schedule</span> Time Patterns</h4>
            <ul>
                <li><span class="material-symbols-rounded">chevron_right</span> Rush hours (8-9 AM, 5-6 PM) show peak demand</li>
                <li><span class="material-symbols-rounded">chevron_right</span> Weekend patterns differ significantly</li>
                <li><span class="material-symbols-rounded">trending_up</span> Year-over-year growth: 64.9%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <h4><span class="material-symbols-rounded">thermostat</span> Weather Impact</h4>
            <ul>
                <li><span class="material-symbols-rounded">chevron_right</span> Temperature is 2nd most important factor</li>
                <li><span class="material-symbols-rounded">wb_sunny</span> Optimal range: 20-25°C (normalized)</li>
                <li><span class="material-symbols-rounded">umbrella</span> Bad weather reduces demand by ~66%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
            <h4><span class="material-symbols-rounded">smart_toy</span> Model Accuracy</h4>
            <ul>
                <li><span class="material-symbols-rounded">chevron_right</span> Explains 88.9% of demand variation</li>
                <li><span class="material-symbols-rounded">chevron_right</span> Average error: ±67 bikes/hour</li>
                <li><span class="material-symbols-rounded">verified</span> Suitable for capacity planning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation Guide
    st.divider()
    st.markdown("## :material/map: Dashboard Navigation Guide")
    
    st.info("""
    **Explore the complete analysis using the sidebar navigation:**
    
    1. **:material/table_view: Data Overview** - Examine data quality, missing values, and basic statistics
    2. **:material/analytics: Exploratory Analysis** - Interactive visualizations of demand patterns and relationships
    3. **:material/speed: Model Performance** - Compare different models and see detailed performance metrics
    4. **:material/online_prediction: Predictions** - Make real-time predictions with custom inputs
    5. **:material/assistant: Recommendations** - Business insights and actionable recommendations
    """, icon=":material/info:")
    
    # Year comparison
    st.divider()
    st.markdown("## :material/compare_arrows: Year-over-Year Comparison")
    
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
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
    <p>Dashboard created with Streamlit | Data: Capital Bikeshare System (2011-2012)</p>
    </div>
    """, unsafe_allow_html=True)
    
except FileNotFoundError:
    st.error("""
    **Data files not found!**
    
    Please make sure you have exported the data from your Jupyter notebook by running the export code.
    The app expects to find data files in the `data/` subdirectory.
    """, icon=":material/database:")
    
    st.info("""
    **Required files:**
    - `data/summary_stats.csv`
    - `data/bike_data_processed.csv`
    - `data/model_results.csv`
    - `data/predictions.csv`
    - `models/best_model.pkl`
    """, icon=":material/description:")
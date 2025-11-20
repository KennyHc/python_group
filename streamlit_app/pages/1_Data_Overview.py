import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from utils import load_css

# Title
st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

load_css()


script_dir = Path(__file__).parent.parent

# Title
st.title("Data Overview & Quality Analysis")
st.markdown("Explore the bike-sharing dataset structure, quality, and basic statistics")

@st.cache_data
def load_data():
    """Loads both original and processed datasets."""
    try:
        path_original = script_dir / 'data/bike_data_original.csv'
        path_processed = script_dir / 'data/bike_data_processed.csv'
        
        df_original = pd.read_csv(path_original)
        df_processed = pd.read_csv(path_processed)
        
        df_original['dteday'] = pd.to_datetime(df_original['dteday'])
        df_processed['dteday'] = pd.to_datetime(df_processed['dteday'])
        
        return df_original, df_processed
        
    except FileNotFoundError as e:
        # Removed 'icon' parameter for compatibility
        st.error(f"Error: A data file was not found.")
        st.error(f"Details: {e}")
        st.info("Please check that 'bike_data_original.csv' and 'bike_data_processed.csv' exist in the 'data' folder.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred loading the data: {e}")
        return None, None

try:
    df_original, df_processed = load_data()
    
    # Data Overview Section
    st.markdown("## :material/dataset: Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df_original):,}")
    with col2:
        st.metric("Original Features", f"{len(df_original.columns)}")
    with col3:
        st.metric("Engineered Features", f"{len(df_processed.columns) - len(df_original.columns)}")
    with col4:
        date_range = f"{df_original['dteday'].min().date()} to {df_original['dteday'].max().date()}"
        st.metric("Date Range", date_range)
    
    # Data Quality Section
    st.markdown("---")
    st.markdown("## :material/fact_check: Data Quality Analysis")
    
    # Missing values check
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### :material/rule: Missing Values Check")
        missing_values = df_original.isnull().sum()
        if missing_values.sum() == 0:
            st.success("No missing values found in the dataset")
            st.markdown("All 17,379 records are complete with no null values.")
        else:
            fig_missing = px.bar(
                x=missing_values[missing_values > 0].index,
                y=missing_values[missing_values > 0].values,
                title="Missing Values by Column",
                labels={'x': 'Column', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig_missing, use_container_width=True)
    
    with col2:
        st.markdown("### :material/donut_large: Data Types")
        dtype_counts = df_original.dtypes.value_counts()
        
        fig_dtype = go.Figure(data=[
            go.Pie(
                labels=[str(dt) for dt in dtype_counts.index],
                values=dtype_counts.values,
                hole=0.4,
                marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
        ])
        
        fig_dtype.update_layout(
            title="Distribution of Data Types",
            height=300
        )
        st.plotly_chart(fig_dtype, use_container_width=True)
    
    # Feature Descriptions
    st.markdown("---")
    st.markdown("## :material/description: Feature Descriptions")
    
    feature_descriptions = {
        'instant': 'Record index',
        'dteday': 'Date',
        'season': '1: Winter, 2: Spring, 3: Summer, 4: Fall',
        'yr': 'Year (0: 2011, 1: 2012)',
        'mnth': 'Month (1 to 12)',
        'hr': 'Hour (0 to 23)',
        'holiday': 'Whether day is holiday or not',
        'weekday': 'Day of the week',
        'workingday': 'If day is neither weekend nor holiday is 1, otherwise 0',
        'weathersit': '1: Clear, 2: Mist/Cloudy, 3: Light Rain/Snow, 4: Heavy Rain/Snow',
        'temp': 'Normalized temperature in Celsius (divided by 41)',
        'atemp': 'Normalized feeling temperature (divided by 50)',
        'hum': 'Normalized humidity (divided by 100)',
        'windspeed': 'Normalized wind speed (divided by 67)',
        'casual': 'Count of casual users',
        'registered': 'Count of registered users',
        'cnt': 'Count of total rental bikes'
    }
    
    # Removed 'icon' parameter
    with st.expander("View detailed feature descriptions"):
        for feature, description in feature_descriptions.items():
            if feature in df_original.columns:
                st.markdown(f"**{feature}**: {description}")
    
    # Statistical Summary
    st.markdown("---")
    st.markdown("## :material/analytics: Statistical Summary")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "Numerical Variables", 
        "Categorical Variables", 
        "Target Variable"
    ])
    
    with tab1:
        # Numerical features
        numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
        
        selected_numerical = st.multiselect(
            "Select numerical features to analyze:",
            numerical_features,
            default=['temp', 'hum', 'cnt']
        )
        
        if selected_numerical:
            # Summary statistics
            summary_stats = df_original[selected_numerical].describe().round(2)
            st.dataframe(summary_stats, use_container_width=True)
            
            st.markdown("### :material/bar_chart: Distribution Analysis")
            
            fig = make_subplots(
                rows=len(selected_numerical),
                cols=2,
                subplot_titles=[f'{feat} - Histogram' for feat in selected_numerical] + 
                              [f'{feat} - Box Plot' for feat in selected_numerical],
                vertical_spacing=0.1
            )
            
            for i, feature in enumerate(selected_numerical):
                # Histogram
                fig.add_trace(
                    go.Histogram(x=df_original[feature], name=feature, showlegend=False),
                    row=i+1, col=1
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(y=df_original[feature], name=feature, showlegend=False),
                    row=i+1, col=2
                )
            
            fig.update_layout(height=300*len(selected_numerical))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Categorical features
        categorical_features = ['season', 'yr', 'holiday', 'weekday', 'workingday', 'weathersit']
        
        selected_categorical = st.multiselect(
            "Select categorical features to analyze:",
            categorical_features,
            default=['season', 'weathersit']
        )
        
        if selected_categorical:
            cols = st.columns(2)
            
            for idx, feature in enumerate(selected_categorical):
                with cols[idx % 2]:
                    # Value counts
                    value_counts = df_original[feature].value_counts().sort_index()
                    
                    # Map values to meaningful labels
                    labels_map = {
                        'season': {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'},
                        'yr': {0: '2011', 1: '2012'},
                        'weathersit': {
                            1: 'Clear',
                            2: 'Mist/Cloudy',
                            3: 'Light Rain/Snow',
                            4: 'Heavy Rain/Snow'
                        },
                        'weekday': {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
                    }
                    
                    if feature in labels_map:
                        labels = [labels_map[feature].get(val, str(val)) for val in value_counts.index]
                    else:
                        labels = [str(val) for val in value_counts.index]
                    
                    fig = go.Figure([go.Bar(
                        x=labels,
                        y=value_counts.values,
                        text=value_counts.values,
                        textposition='auto',
                        marker_color='lightblue'
                    )])
                    
                    fig.update_layout(
                        title=f'Distribution of {feature.title()}',
                        xaxis_title=feature.title(),
                        yaxis_title='Count',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add percentage info
                    st.caption(f"**{feature}** value distribution:")
                    for label, count in zip(labels, value_counts.values):
                        percentage = (count / len(df_original)) * 100
                        st.caption(f"- {label}: {count:,} ({percentage:.1f}%)")
    
    with tab3:
        st.markdown("### :material/target: Target Variable: Total Bike Rentals (cnt)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution
            fig_target_dist = go.Figure()
            
            fig_target_dist.add_trace(go.Histogram(
                x=df_original['cnt'],
                nbinsx=50,
                name='Distribution',
                marker_color='green'
            ))
            
            fig_target_dist.update_layout(
                title='Distribution of Total Rentals',
                xaxis_title='Number of Rentals',
                yaxis_title='Frequency',
                showlegend=False
            )
            
            st.plotly_chart(fig_target_dist, use_container_width=True)
            
            # Summary stats
            st.markdown("**Summary Statistics:**")
            target_stats = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness'],
                'Value': [
                    f"{df_original['cnt'].mean():.1f}",
                    f"{df_original['cnt'].median():.1f}",
                    f"{df_original['cnt'].std():.1f}",
                    f"{df_original['cnt'].min()}",
                    f"{df_original['cnt'].max()}",
                    f"{df_original['cnt'].skew():.2f}"
                ]
            })
            st.dataframe(target_stats, hide_index=True)
        
        with col2:
            # Time series of target
            daily_avg = df_original.groupby(df_original['dteday'].dt.date)['cnt'].mean()
            
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=daily_avg.index,
                y=daily_avg.values,
                mode='lines',
                line=dict(color='darkgreen', width=2)
            ))
            
            fig_ts.update_layout(
                title='Average Daily Rentals Over Time',
                xaxis_title='Date',
                yaxis_title='Average Rentals',
                height=400
            )
            
            st.plotly_chart(fig_ts, use_container_width=True)
    
    # Data Sample
    st.markdown("---")
    st.markdown("## :material/table_view: Data Sample")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        sample_size = st.slider("Sample size", 5, 100, 20)
    with col2:
        sort_by = st.selectbox("Sort by", df_original.columns, index=1)
    with col3:
        ascending = st.checkbox("Ascending order", value=True)
    
    # Display sample
    sample_df = df_original.sort_values(by=sort_by, ascending=ascending).head(sample_size)
    st.dataframe(sample_df, use_container_width=True, height=400)
    
    # Export option
    csv = sample_df.to_csv(index=False)
    # Removed 'icon' parameter
    st.download_button(
        label="Download Sample Data as CSV",
        data=csv,
        file_name="bike_sharing_sample.csv",
        mime="text/csv"
    )
    
    # Feature Engineering Summary
    st.markdown("---")
    st.markdown("## :material/engineering: Feature Engineering Summary")
    
    # Removed 'icon' parameter
    st.info("""
    **New features created during analysis:**
    - **Temporal Features**: `hour_sin`, `hour_cos`, `month_sin`, `month_cos` (cyclical encoding)
    - **Rush Hour Indicators**: `is_rush_hour` (binary flag for peak commute times)
    - **Weather Categories**: `is_good_weather`, `is_bad_weather`, `is_comfortable_temp`
    - **Interaction Terms**: `temp_humidity_interaction`, `temp_weather_interaction`
    - **Derived Features**: `feels_like_diff` (difference between temp and atemp)
    - **Time Flags**: `is_weekend` (weekend indicator)
    
    These engineered features help capture non-linear patterns and interactions in the data.
    """)

except Exception as e:
    # Removed 'icon' parameter
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure the data files are properly exported from the Jupyter notebook.")
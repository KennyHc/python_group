import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from utils import load_css

# Note: page_icon best supports emojis for the browser tab
st.set_page_config(page_title="Exploratory Analysis", page_icon="🔍", layout="wide")

load_css()


# Title with Material Icon
st.title(":material/analytics: Exploratory Data Analysis")
st.markdown("Discover patterns, trends, and relationships in bike-sharing demand")

script_dir = Path(__file__).parent.parent

# --- COLOR PALETTE FOR DARK MODE ---
COLORS = {
    'cyan': '#00F0FF',
    'pink': '#FF007A',
    'purple': '#9D00FF',
    'amber': '#FFBC00',
    'green': '#00FF94',
    'slate': '#8892b0',
    'dark_bg': '#0e1117'
}

@st.cache_data
def load_data():
    """Loads the main processed dataset."""
    try:
        data_path = script_dir / 'data/bike_data_processed.csv'
        
        df = pd.read_csv(data_path)
        df['dteday'] = pd.to_datetime(df['dteday'])
        return df
        
    except FileNotFoundError:
        st.error("Error: The main data file ('bike_data_processed.csv') was not found.")
        st.info("Please check that the file exists in the 'data' folder in your GitHub repo.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the data: {e}")
        return None

try:
    df = load_data()
    
    # Add helper columns for visualization
    df['year'] = df['yr'].map({0: '2011', 1: '2012'})
    df['season_name'] = df['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
    df['weather_name'] = df['weathersit'].map({
        1: 'Clear',
        2: 'Mist/Cloudy',
        3: 'Light Rain/Snow',
        4: 'Heavy Rain/Snow'
    })
    df['day_type'] = df['workingday'].map({0: 'Holiday/Weekend', 1: 'Working Day'})
    
    # Sidebar filters
    st.sidebar.markdown("## :material/tune: Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(df['dteday'].min(), df['dteday'].max()),
        min_value=df['dteday'].min(),
        max_value=df['dteday'].max()
    )
    
    # User type filter
    user_type = st.sidebar.radio(
        "Select user type to analyze",
        ["Total (cnt)", "Casual Users", "Registered Users"]
    )
    
    user_col_map = {
        "Total (cnt)": "cnt",
        "Casual Users": "casual",
        "Registered Users": "registered"
    }
    selected_col = user_col_map[user_type]
    
    # Filter data based on date range
    if len(date_range) == 2:
        mask = (df['dteday'].dt.date >= date_range[0]) & (df['dteday'].dt.date <= date_range[1])
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()
    
    # Analysis sections with Material Icons in Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        ":material/schedule: Temporal Patterns",
        ":material/thermostat: Weather Impact",
        ":material/groups: User Behavior",
        ":material/hub: Correlations",
        ":material/science: Advanced Analysis"
    ])
    
    with tab1:
        st.markdown("## :material/calendar_month: Temporal Patterns Analysis")
        
        # Hourly patterns
        st.markdown("### :material/access_time: Hourly Demand Patterns")
        
        # Add day type filter
        day_type_filter = st.selectbox(
            "Filter by day type",
            ["All Days", "Working Days", "Weekends/Holidays"]
        )
        
        if day_type_filter == "Working Days":
            hourly_data = df_filtered[df_filtered['workingday'] == 1]
        elif day_type_filter == "Weekends/Holidays":
            hourly_data = df_filtered[df_filtered['workingday'] == 0]
        else:
            hourly_data = df_filtered
        
        hourly_avg = hourly_data.groupby('hr')[selected_col].agg(['mean', 'std']).reset_index()
        
        fig_hourly = go.Figure()
        
        # Add mean line (Bright Cyan)
        fig_hourly.add_trace(go.Scatter(
            x=hourly_avg['hr'],
            y=hourly_avg['mean'],
            mode='lines+markers',
            name='Average',
            line=dict(color=COLORS['cyan'], width=3),
            marker=dict(size=8, color=COLORS['dark_bg'], line=dict(width=2, color=COLORS['cyan']))
        ))
        
        # Add confidence interval (Cyan with transparency)
        fig_hourly.add_trace(go.Scatter(
            x=hourly_avg['hr'].tolist() + hourly_avg['hr'].tolist()[::-1],
            y=(hourly_avg['mean'] + hourly_avg['std']).tolist() + 
              (hourly_avg['mean'] - hourly_avg['std']).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 240, 255, 0.1)', # Cyan transparent
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
        
        # Highlight rush hours (Amber/Yellow)
        for hour in [8, 9, 17, 18]:
            fig_hourly.add_vrect(
                x0=hour-0.5, x1=hour+0.5,
                fillcolor=COLORS['amber'], opacity=0.15,
                layer="below", line_width=0
            )
        
        fig_hourly.update_layout(
            title=f'Hourly {user_type} Pattern - {day_type_filter}',
            xaxis_title='Hour of Day',
            yaxis_title=f'Average {user_type}',
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Key insights
        peak_hour = hourly_avg.loc[hourly_avg['mean'].idxmax(), 'hr']
        peak_value = hourly_avg.loc[hourly_avg['mean'].idxmax(), 'mean']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(":material/vertical_align_top: Peak Hour", f"{peak_hour}:00", f"{peak_value:.0f} avg rentals")
        with col2:
            morning_peak = hourly_avg[(hourly_avg['hr'] >= 7) & (hourly_avg['hr'] <= 9)]['mean'].max()
            st.metric(":material/wb_twilight: Morning Peak (7-9 AM)", f"{morning_peak:.0f} rentals")
        with col3:
            evening_peak = hourly_avg[(hourly_avg['hr'] >= 17) & (hourly_avg['hr'] <= 19)]['mean'].max()
            st.metric(":material/wb_twilight: Evening Peak (5-7 PM)", f"{evening_peak:.0f} rentals")
        
        # Weekly patterns
        st.divider()
        st.markdown("### :material/date_range: Weekly Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week average
            weekday_avg = df_filtered.groupby('weekday')[selected_col].mean().reset_index()
            weekday_avg['day_name'] = weekday_avg['weekday'].map({
                0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
                4: 'Thursday', 5: 'Friday', 6: 'Saturday'
            })
            
            # Weekend = Pink, Weekday = Muted Cyan
            bar_colors = [COLORS['pink'] if i in [0, 6] else 'rgba(0, 240, 255, 0.6)' for i in range(7)]
            
            fig_weekday = go.Figure([go.Bar(
                x=weekday_avg['day_name'],
                y=weekday_avg[selected_col],
                marker_color=bar_colors,
                text=weekday_avg[selected_col].round(0),
                textposition='auto'
            )])
            
            fig_weekday.update_layout(
                title=f'Average {user_type} by Day of Week',
                xaxis_title='Day of Week',
                yaxis_title=f'Average {user_type}'
            )
            
            st.plotly_chart(fig_weekday, use_container_width=True)
        
        with col2:
            # Monthly trend
            monthly_avg = df_filtered.groupby(['year', 'mnth'])[selected_col].mean().reset_index()
            monthly_avg['month_year'] = monthly_avg.apply(
                lambda x: f"{x['year']}-{x['mnth']:02d}", axis=1
            )
            
            fig_monthly = go.Figure()
            
            # 2011 (Gray/Slate), 2012 (Neon Green)
            colors_year = {'2011': COLORS['slate'], '2012': COLORS['green']}
            
            for year in ['2011', '2012']:
                year_data = monthly_avg[monthly_avg['year'] == year]
                fig_monthly.add_trace(go.Scatter(
                    x=year_data['mnth'],
                    y=year_data[selected_col],
                    mode='lines+markers',
                    name=year,
                    line=dict(width=3, color=colors_year[year]),
                    marker=dict(size=8)
                ))
            
            fig_monthly.update_layout(
                title=f'Monthly {user_type} Trend',
                xaxis_title='Month',
                yaxis_title=f'Average {user_type}',
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Seasonal patterns
        st.markdown("### :material/eco: Seasonal Analysis")
        
        seasonal_stats = df_filtered.groupby('season_name')[selected_col].agg(['mean', 'sum', 'std']).round(0)
        seasonal_stats['cv'] = (seasonal_stats['std'] / seasonal_stats['mean'] * 100).round(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Custom seasonal palette
            season_colors = [COLORS['cyan'], COLORS['green'], COLORS['amber'], COLORS['pink']]
            
            fig_season = go.Figure([go.Bar(
                x=seasonal_stats.index,
                y=seasonal_stats['mean'],
                text=seasonal_stats['mean'].astype(int),
                textposition='auto',
                marker_color=season_colors
            )])
            
            fig_season.update_layout(
                title=f'Average {user_type} by Season',
                xaxis_title='Season',
                yaxis_title=f'Average {user_type}'
            )
            
            st.plotly_chart(fig_season, use_container_width=True)
        
        with col2:
            st.markdown("**Seasonal Statistics:**")
            display_stats = seasonal_stats[['mean', 'sum', 'cv']].copy()
            display_stats.columns = ['Avg Rentals', 'Total Rentals', 'Variability (%)']
            st.dataframe(display_stats, use_container_width=True)
            
            best_season = seasonal_stats['mean'].idxmax()
            worst_season = seasonal_stats['mean'].idxmin()
            
            st.success(f":material/trophy: Best season: **{best_season}** ({seasonal_stats.loc[best_season, 'mean']:.0f} avg rentals)")
            st.info(f":material/trending_down: Lowest season: **{worst_season}** ({seasonal_stats.loc[worst_season, 'mean']:.0f} avg rentals)")
    
    with tab2:
        st.markdown("## :material/thermostat: Weather Impact Analysis")
        
        # Temperature impact
        st.markdown("### :material/thermometer: Temperature vs Demand")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create temperature bins for better visualization
            df_filtered['temp_celsius'] = df_filtered['temp'] * 41  # Denormalize
            df_filtered['temp_bin'] = pd.cut(df_filtered['temp_celsius'], bins=20)
            
            temp_avg = df_filtered.groupby('temp_bin')[selected_col].mean().reset_index()
            temp_avg['temp_mid'] = temp_avg['temp_bin'].apply(lambda x: x.mid)
            
            fig_temp = go.Figure()
            
            # Scatter plot: Transparent Cyan
            fig_temp.add_trace(go.Scatter(
                x=df_filtered['temp_celsius'],
                y=df_filtered[selected_col],
                mode='markers',
                name='Hourly Data',
                marker=dict(
                    size=4,
                    opacity=0.3,
                    color=COLORS['cyan']
                )
            ))
            
            # Average line: Hot Pink
            fig_temp.add_trace(go.Scatter(
                x=temp_avg['temp_mid'],
                y=temp_avg[selected_col],
                mode='lines',
                name='Average',
                line=dict(color=COLORS['pink'], width=4)
            ))
            
            fig_temp.update_layout(
                title=f'Temperature Impact on {user_type}',
                xaxis_title='Temperature (°C)',
                yaxis_title=user_type,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            # Temperature stats
            optimal_temp_bin = temp_avg.loc[temp_avg[selected_col].idxmax(), 'temp_bin']
            optimal_temp = temp_avg.loc[temp_avg[selected_col].idxmax(), 'temp_mid']
            
            st.markdown("**Temperature Insights:**")
            st.metric(":material/wb_sunny: Optimal Temperature", f"{optimal_temp:.1f}°C")
            
            # Comfort zones
            comfort_zones = {
                'Cold (< 10°C)': df_filtered[df_filtered['temp_celsius'] < 10][selected_col].mean(),
                'Cool (10-20°C)': df_filtered[(df_filtered['temp_celsius'] >= 10) & 
                                             (df_filtered['temp_celsius'] < 20)][selected_col].mean(),
                'Warm (20-30°C)': df_filtered[(df_filtered['temp_celsius'] >= 20) & 
                                             (df_filtered['temp_celsius'] < 30)][selected_col].mean(),
                'Hot (> 30°C)': df_filtered[df_filtered['temp_celsius'] >= 30][selected_col].mean()
            }
            
            for zone, avg in comfort_zones.items():
                st.caption(f"{zone}: {avg:.0f} avg rentals")
        
        # Weather situation impact
        st.divider()
        st.markdown("### :material/cloud: Weather Conditions Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weather_avg = df_filtered.groupby('weather_name')[selected_col].agg(['mean', 'count']).reset_index()
            
            # Good -> Green, Bad -> Pink/Purple
            weather_colors = [COLORS['green'], COLORS['amber'], COLORS['pink'], COLORS['purple']]
            
            fig_weather = go.Figure([go.Bar(
                x=weather_avg['weather_name'],
                y=weather_avg['mean'],
                text=weather_avg['mean'].round(0),
                textposition='auto',
                marker_color=weather_colors
            )])
            
            fig_weather.update_layout(
                title=f'Average {user_type} by Weather Condition',
                xaxis_title='Weather Condition',
                yaxis_title=f'Average {user_type}'
            )
            
            st.plotly_chart(fig_weather, use_container_width=True)
        
        with col2:
            # Weather frequency and impact
            st.markdown("**Weather Statistics:**")
            weather_stats = pd.DataFrame({
                'Condition': weather_avg['weather_name'],
                'Frequency': (weather_avg['count'] / len(df_filtered) * 100).round(1),
                'Avg Rentals': weather_avg['mean'].round(0),
                'Impact': ((weather_avg['mean'] / weather_avg['mean'].iloc[0] - 1) * 100).round(1)
            })
            
            st.dataframe(weather_stats, use_container_width=True, hide_index=True)
        
        # Combined weather factors
        st.divider()
        st.markdown("### :material/water_drop: Combined Weather Factors")
        
        # Create 2D heatmap of temp vs humidity
        temp_bins = pd.qcut(df_filtered['temp'], q=10, duplicates='drop')
        hum_bins = pd.qcut(df_filtered['hum'], q=10, duplicates='drop')
        
        heatmap_data = df_filtered.groupby([temp_bins, hum_bins])[selected_col].mean().unstack()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['Low Humidity'] + [f'Hum {i}' for i in range(2, 10)] + ['High Humidity'],
            y=['Low Temp'] + [f'Temp {i}' for i in range(2, 10)] + ['High Temp'],
            colorscale='Viridis', # Viridis pops in dark mode
            text=heatmap_data.values.round(0),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig_heatmap.update_layout(
            title=f'Average {user_type} by Temperature and Humidity',
            xaxis_title='Humidity Level',
            yaxis_title='Temperature Level',
            height=500
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.markdown("## :material/groups: User Behavior Analysis")
        
        # Casual vs Registered comparison
        st.markdown("### :material/compare_arrows: Casual vs Registered Users")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern comparison
            hourly_user = df_filtered.groupby('hr')[['casual', 'registered']].mean().reset_index()
            
            fig_user_hourly = go.Figure()
            
            # Casual = Pink, Registered = Cyan
            fig_user_hourly.add_trace(go.Scatter(
                x=hourly_user['hr'],
                y=hourly_user['casual'],
                mode='lines+markers',
                name='Casual Users',
                line=dict(color=COLORS['pink'], width=3)
            ))
            
            fig_user_hourly.add_trace(go.Scatter(
                x=hourly_user['hr'],
                y=hourly_user['registered'],
                mode='lines+markers',
                name='Registered Users',
                line=dict(color=COLORS['cyan'], width=3)
            ))
            
            fig_user_hourly.update_layout(
                title='Hourly Patterns: Casual vs Registered',
                xaxis_title='Hour of Day',
                yaxis_title='Average Users',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_user_hourly, use_container_width=True)
        
        with col2:
            # User type proportion
            total_casual = df_filtered['casual'].sum()
            total_registered = df_filtered['registered'].sum()
            
            fig_pie = go.Figure([go.Pie(
                labels=['Casual', 'Registered'],
                values=[total_casual, total_registered],
                hole=0.4,
                marker_colors=[COLORS['pink'], COLORS['cyan']]
            )])
            
            fig_pie.update_layout(
                title='User Type Distribution',
                annotations=[dict(text=f'{total_casual + total_registered:,}', 
                                x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # User behavior by conditions
        st.markdown("### :material/ads_click: User Preferences")
        
        # Calculate preferences
        conditions = ['workingday', 'holiday', 'season_name', 'weather_name']
        
        preference_data = []
        for condition in conditions:
            for value in df_filtered[condition].unique():
                subset = df_filtered[df_filtered[condition] == value]
                casual_ratio = subset['casual'].sum() / subset['cnt'].sum() * 100
                preference_data.append({
                    'Condition': condition.replace('_name', '').title(),
                    'Value': value,
                    'Casual %': casual_ratio,
                    'Registered %': 100 - casual_ratio
                })
        
        preference_df = pd.DataFrame(preference_data)
        
        # Sort by casual percentage
        preference_df = preference_df.sort_values('Casual %', ascending=False)
        
        fig_preferences = go.Figure()
        
        fig_preferences.add_trace(go.Bar(
            x=preference_df['Value'],
            y=preference_df['Casual %'],
            name='Casual %',
            marker_color=COLORS['pink']
        ))
        
        fig_preferences.add_trace(go.Bar(
            x=preference_df['Value'],
            y=preference_df['Registered %'],
            name='Registered %',
            marker_color=COLORS['cyan']
        ))
        
        fig_preferences.update_layout(
            title='User Type Distribution by Conditions',
            xaxis_title='Condition',
            yaxis_title='Percentage',
            barmode='stack',
            height=500
        )
        
        st.plotly_chart(fig_preferences, use_container_width=True)
        
        # Key insights
        st.markdown("### :material/lightbulb: Key User Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            casual_weekend = df_filtered[df_filtered['workingday'] == 0]['casual'].mean()
            casual_weekday = df_filtered[df_filtered['workingday'] == 1]['casual'].mean()
            weekend_increase = ((casual_weekend / casual_weekday - 1) * 100)
            
            st.metric(
                ":material/trending_up: Casual Weekend Increase",
                f"{weekend_increase:.1f}%",
                "vs weekdays"
            )
        
        with col2:
            reg_morning = df_filtered[(df_filtered['hr'] >= 7) & (df_filtered['hr'] <= 9)]['registered'].mean()
            reg_other = df_filtered[~((df_filtered['hr'] >= 7) & (df_filtered['hr'] <= 9))]['registered'].mean()
            morning_ratio = reg_morning / reg_other
            
            st.metric(
                ":material/commute: Registered Morning Commute",
                f"{morning_ratio:.1f}x",
                "vs other hours"
            )
        
        with col3:
            good_weather_ratio = df_filtered[df_filtered['weathersit'] == 1]['casual'].mean()
            bad_weather_ratio = df_filtered[df_filtered['weathersit'] > 2]['casual'].mean()
            weather_sensitivity = ((good_weather_ratio / bad_weather_ratio - 1) * 100)
            
            st.metric(
                ":material/water: Casual Weather Sensitivity",
                f"{weather_sensitivity:.0f}%",
                "good vs bad weather"
            )
    
    with tab4:
        st.markdown("## :material/hub: Correlation Analysis")
        
        # Correlation matrix
        st.markdown("### :material/dataset: Feature Correlations")
        
        # Select features for correlation
        corr_features = st.multiselect(
            "Select features for correlation analysis",
            ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'workingday', 'weekday', 
             'weathersit', 'season', 'casual', 'registered', 'cnt'],
            default=['temp', 'hum', 'windspeed', 'hr', 'casual', 'registered', 'cnt']
        )
        
        if len(corr_features) > 1:
            corr_matrix = df_filtered[corr_features].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu', # Standard for correlation (Red=Neg, Blue=Pos)
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig_corr.update_layout(
                title='Correlation Matrix',
                height=600,
                xaxis={'side': 'bottom'}
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Key correlations
            st.markdown("### :material/key: Key Correlations with Demand")
            
            # Get correlations with target
            target_corr = corr_matrix['cnt'].drop('cnt').sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Positive Correlations:**")
                positive_corr = target_corr[target_corr > 0].head(5)
                for feature, corr in positive_corr.items():
                    st.caption(f":material/add_circle: {feature}: {corr:.3f}")
            
            with col2:
                st.markdown("**Negative Correlations:**")
                negative_corr = target_corr[target_corr < 0].head(5)
                for feature, corr in negative_corr.items():
                    st.caption(f":material/remove_circle: {feature}: {corr:.3f}")
        
        # Scatter plot matrix
        st.divider()
        st.markdown("### :material/scatter_plot: Relationship Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("Select X-axis feature", 
                                    ['temp', 'atemp', 'hum', 'windspeed', 'hr'],
                                    index=0)
        with col2:
            y_feature = st.selectbox("Select Y-axis feature",
                                    ['cnt', 'casual', 'registered'],
                                    index=0)
        
        # Create scatter plot with trend line
        fig_scatter = px.scatter(
            df_filtered,
            x=x_feature,
            y=y_feature,
            color='weather_name',
            size='cnt',
            trendline="lowess",
            title=f'{y_feature} vs {x_feature}',
            opacity=0.6,
            color_discrete_map={
                'Clear': COLORS['green'],
                'Mist/Cloudy': COLORS['amber'],
                'Light Rain/Snow': COLORS['pink'],
                'Heavy Rain/Snow': COLORS['purple']
            }
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab5:
        st.markdown("## :material/science: Advanced Analysis")
   
        st.markdown("### :material/trending_up: Year-over-Year Growth Analysis")
        
        yoy_data = df.groupby(['yr', 'mnth']).agg({
            'cnt': 'sum',
            'casual': 'sum',
            'registered': 'sum'
        }).reset_index()
        
        yoy_pivot = yoy_data.pivot(index='mnth', columns='yr', values=['cnt', 'casual', 'registered'])
        
        growth_rates = pd.DataFrame({
            'Month': range(1, 13),
            'Total Growth %': ((yoy_pivot['cnt'][1] / yoy_pivot['cnt'][0] - 1) * 100).round(1),
            'Casual Growth %': ((yoy_pivot['casual'][1] / yoy_pivot['casual'][0] - 1) * 100).round(1),
            'Registered Growth %': ((yoy_pivot['registered'][1] / yoy_pivot['registered'][0] - 1) * 100).round(1)
        })
        
        fig_growth = go.Figure()
        
        # Distinct neon colors for growth lines
        growth_colors = {
            'Total Growth %': COLORS['cyan'],
            'Casual Growth %': COLORS['pink'],
            'Registered Growth %': COLORS['purple']
        }

        for col in ['Total Growth %', 'Casual Growth %', 'Registered Growth %']:
            fig_growth.add_trace(go.Scatter(
                x=growth_rates['Month'],
                y=growth_rates[col],
                mode='lines+markers',
                name=col.replace(' %', ''),
                line=dict(width=3, color=growth_colors[col]),
                marker=dict(size=8)
            ))
        
        fig_growth.add_hline(y=0, line_dash="dash", line_color=COLORS['slate'])
        
        fig_growth.update_layout(
            title='Year-over-Year Growth by Month (2012 vs 2011)',
            xaxis_title='Month',
            yaxis_title='Growth Percentage (%)',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Growth summary
        col1, col2, col3 = st.columns(3)
        
        total_2011 = df[df['yr'] == 0]['cnt'].sum()
        total_2012 = df[df['yr'] == 1]['cnt'].sum()
        overall_growth = ((total_2012 / total_2011 - 1) * 100)
        
        with col1:
            st.metric(":material/show_chart: Overall Growth", f"{overall_growth:.1f}%", "2012 vs 2011")
        with col2:
            st.metric("2011 Total", f"{total_2011:,} rentals")
        with col3:
            st.metric("2012 Total", f"{total_2012:,} rentals")
        
        # Demand forecasting insights
        st.divider()
        st.markdown("### :material/insights: Demand Pattern Insights")
        
        # Calculate some advanced metrics
        # Weekday vs weekend ratio
        weekday_avg = df_filtered[df_filtered['workingday'] == 1]['cnt'].mean()
        weekend_avg = df_filtered[df_filtered['workingday'] == 0]['cnt'].mean()
        
        # Peak hour concentration
        hourly_totals = df_filtered.groupby('hr')['cnt'].sum()
        top_3_hours = hourly_totals.nlargest(3)
        peak_concentration = (top_3_hours.sum() / hourly_totals.sum() * 100)
        
        # Weather impact
        clear_avg = df_filtered[df_filtered['weathersit'] == 1]['cnt'].mean()
        bad_weather_avg = df_filtered[df_filtered['weathersit'] >= 3]['cnt'].mean()
        weather_impact = ((clear_avg - bad_weather_avg) / clear_avg * 100)
        
        # Display insights
        st.markdown("#### :material/lightbulb: Key Operational Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.info(f"""
            ** Demand Concentration**
            - Top 3 hours account for {peak_concentration:.1f}% of daily demand
            - Peak hours: {', '.join([f"{h}:00" for h in top_3_hours.index])}
            - Weekday demand is {(weekday_avg/weekend_avg):.1f}x weekend demand
            """)
        
        with insight_col2:
            st.warning(f"""
            ** Weather Sensitivity**
            - Bad weather reduces demand by {weather_impact:.1f}%
            - Temperature explains {(df['temp'].corr(df['cnt'])**2 * 100):.1f}% of variance
            - Optimal operations window: 20-30°C, low humidity
            """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure the data files are properly exported from the Jupyter notebook.")
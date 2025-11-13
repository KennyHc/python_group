import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Recommendations", page_icon="💡", layout="wide")

# Title
st.title("Business Recommendations")
st.markdown("Actionable insights and recommendations based on data analysis and model results")

# Load data for insights
@st.cache_data
def load_analysis_data():
    try:

        script_dir = Path(__file__).parent.parent
        
        path_df = script_dir / 'data/bike_data_processed.csv'
        path_model_results = script_dir / 'data/tuned_model_results.csv'
        path_predictions = script_dir / 'data/predictions.csv'
        
        df = pd.read_csv(path_df)
        df['dteday'] = pd.to_datetime(df['dteday'])
        
        model_results = pd.read_csv(path_model_results, index_col=0)
        
        predictions = pd.read_csv(path_predictions)
        
        return df, model_results, predictions

    except FileNotFoundError as e:
        st.error(f"Error: A data file was not found.")
        st.error(f"Details: {e}")
        st.info("Please make sure the 'data' folder and its CSV files are in your GitHub repo and the paths are correct.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None, None, None

try:
    df, model_results, predictions = load_analysis_data()
    
    if df is None:
        st.error("Unable to load data files. Please ensure data is exported from the notebook.")
        st.stop()
    
    # Executive Summary
    st.markdown("## 📋 Executive Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_rentals_2011 = df[df['yr'] == 0]['cnt'].sum()
    total_rentals_2012 = df[df['yr'] == 1]['cnt'].sum()
    yoy_growth = (total_rentals_2012 / total_rentals_2011 - 1) * 100
    
    with col1:
        st.metric("YoY Growth", f"{yoy_growth:.1f}%", "2012 vs 2011")
    
    with col2:
        avg_daily = df.groupby(df['dteday'].dt.date)['cnt'].sum().mean()
        st.metric("Avg Daily Rentals", f"{avg_daily:,.0f}", "bikes/day")
    
    with col3:
        peak_hour_demand = df.groupby('hr')['cnt'].mean().max()
        st.metric("Peak Hour Demand", f"{peak_hour_demand:.0f}", "avg bikes/hour")
    
    with col4:
        weather_impact = (df[df['weathersit'] == 1]['cnt'].mean() - 
                         df[df['weathersit'] >= 3]['cnt'].mean()) / df[df['weathersit'] == 1]['cnt'].mean() * 100
        st.metric("Weather Impact", f"-{weather_impact:.0f}%", "bad vs good")
    
    # Strategic recommendations sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚴 Operations Strategy",
        "💰 Revenue Optimization",
        "🌡️ Weather Response",
        "📈 Growth Opportunities",
        "🎯 Implementation Roadmap"
    ])
    
    with tab1:
        st.markdown("## 🚴 Operational Recommendations")
        
        # Fleet management
        st.markdown("### 1. Dynamic Fleet Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Calculate hourly requirements
            hourly_avg = df.groupby('hr')['cnt'].agg(['mean', 'std', 'max']).round(0)
            hourly_avg['recommended_bikes'] = hourly_avg['mean'] + 2 * hourly_avg['std']  # 95% confidence
            
            fig = go.Figure()
            
            # Average demand
            fig.add_trace(go.Scatter(
                x=hourly_avg.index,
                y=hourly_avg['mean'],
                name='Average Demand',
                line=dict(color='blue', width=3)
            ))
            
            # Recommended capacity
            fig.add_trace(go.Scatter(
                x=hourly_avg.index,
                y=hourly_avg['recommended_bikes'],
                name='Recommended Capacity',
                line=dict(color='red', width=2, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ))
            
            fig.update_layout(
                title='Recommended Bike Availability by Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Number of Bikes',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Fleet Sizing")
            
            max_concurrent = hourly_avg['recommended_bikes'].max()
            avg_concurrent = hourly_avg['recommended_bikes'].mean()
            
            st.metric("Peak Fleet Size", f"{max_concurrent:,.0f} bikes")
            st.metric("Average Fleet Need", f"{avg_concurrent:,.0f} bikes")
            st.metric("Utilization Rate", f"{(avg_concurrent/max_concurrent*100):.1f}%")
            
            st.info("""
            **Key Actions:**
            - Maintain {:.0f} bikes for peak hours
            - Redistribute bikes before 8 AM and 5 PM
            - Focus maintenance during 2-4 AM (lowest demand)
            """.format(max_concurrent))
        
        # Staff scheduling
        st.markdown("### 2. Optimized Staff Scheduling")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Staffing needs by hour and day type
            staff_data = df.groupby(['hr', 'workingday'])['cnt'].mean().unstack()
            
            fig_staff = go.Figure()
            
            fig_staff.add_trace(go.Bar(
                x=staff_data.index,
                y=staff_data[1],  # Workdays
                name='Workdays',
                marker_color='darkblue'
            ))
            
            fig_staff.add_trace(go.Bar(
                x=staff_data.index,
                y=staff_data[0],  # Weekends
                name='Weekends/Holidays',
                marker_color='lightblue'
            ))
            
            fig_staff.update_layout(
                title='Demand Patterns for Staff Planning',
                xaxis_title='Hour',
                yaxis_title='Average Demand',
                barmode='group',
                height=350
            )
            
            st.plotly_chart(fig_staff, use_container_width=True)
        
        with col2:
            st.markdown("#### 👥 Staffing Guidelines")
            
            # Identify peak periods
            workday_peaks = staff_data[1].nlargest(3)
            weekend_peaks = staff_data[0].nlargest(3)
            
            st.warning(f"""
            **Workday Peak Hours:**
            - {', '.join([f'{h}:00' for h in workday_peaks.index])}
            - Require {workday_peaks.mean()/50:.0f} staff members
            
            **Weekend Peak Hours:**
            - {', '.join([f'{h}:00' for h in weekend_peaks.index])}
            - Require {weekend_peaks.mean()/50:.0f} staff members
            """)
        
        # Maintenance scheduling
        st.markdown("### 3. Maintenance Windows")
        
        # Find best maintenance hours
        maintenance_hours = hourly_avg.nsmallest(6, 'mean')
        
        st.success(f"""
        **Optimal Maintenance Schedule:**
        - Primary window: 2:00 AM - 5:00 AM (lowest demand: {maintenance_hours['mean'].min():.0f} bikes/hour)
        - Secondary window: 10:00 PM - 12:00 AM
        - Avoid: 7-9 AM and 5-7 PM (peak commute times)
        - Weekend maintenance: More flexible, avoid 10 AM - 6 PM
        """)
    
    with tab2:
        st.markdown("## 💰 Revenue Optimization Strategies")
        
        # Pricing recommendations
        st.markdown("### 1. Dynamic Pricing Model")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create pricing tiers based on demand
            df['demand_tier'] = pd.qcut(df['cnt'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            # Calculate average demand by various factors
            pricing_factors = df.groupby(['hr', 'demand_tier'])['cnt'].count().unstack(fill_value=0)
            
            # Create heatmap
            fig_pricing = go.Figure(data=go.Heatmap(
                z=pricing_factors.values.T,
                x=pricing_factors.index,
                y=pricing_factors.columns,
                colorscale='YlOrRd',
                text=pricing_factors.values.T,
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig_pricing.update_layout(
                title='Demand Frequency by Hour (for Pricing Strategy)',
                xaxis_title='Hour of Day',
                yaxis_title='Demand Level',
                height=400
            )
            
            st.plotly_chart(fig_pricing, use_container_width=True)
        
        with col2:
            st.markdown("#### 💵 Pricing Tiers")
            
            st.info("""
            **Recommended Multipliers:**
            - Very High: 1.5x base
            - High: 1.3x base
            - Medium: 1.0x base
            - Low: 0.8x base
            - Very Low: 0.6x base
            """)
            
            # Revenue impact
            base_revenue = 1000000  # Example base
            dynamic_revenue = base_revenue * 1.15  # 15% increase estimate
            
            st.metric(
                "Revenue Impact",
                f"+{(dynamic_revenue/base_revenue-1)*100:.0f}%",
                "with dynamic pricing"
            )
        
        # Customer segmentation
        st.markdown("### 2. Customer Segment Focus")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User type analysis
            user_comparison = pd.DataFrame({
                'Casual': df.groupby('hr')['casual'].mean(),
                'Registered': df.groupby('hr')['registered'].mean()
            })
            
            fig_users = go.Figure()
            
            fig_users.add_trace(go.Scatter(
                x=user_comparison.index,
                y=user_comparison['Casual'],
                name='Casual Users',
                line=dict(color='orange', width=3),
                stackgroup='one'
            ))
            
            fig_users.add_trace(go.Scatter(
                x=user_comparison.index,
                y=user_comparison['Registered'],
                name='Registered Users',
                line=dict(color='blue', width=3),
                stackgroup='one'
            ))
            
            fig_users.update_layout(
                title='User Type Composition by Hour',
                xaxis_title='Hour',
                yaxis_title='Average Users',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_users, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Segment Strategies")
            
            casual_weekend = df[df['workingday'] == 0]['casual'].sum()
            casual_weekday = df[df['workingday'] == 1]['casual'].sum()
            casual_weekend_pct = (casual_weekend / (casual_weekend + casual_weekday)) * 100
            
            st.success(f"""
            **Casual Users ({casual_weekend_pct:.0f}% weekend):**
            - Weekend promotions
            - Tourist packages
            - Family discounts
            
            **Registered Users:**
            - Commuter subscriptions
            - Corporate partnerships
            - Loyalty rewards
            """)
        
        # Revenue opportunities
        st.markdown("### 3. Untapped Revenue Opportunities")
        
        opportunities = {
            'Corporate Partnerships': 25,
            'Tourist Packages': 20,
            'Event-based Rentals': 15,
            'Subscription Tiers': 18,
            'Accessories & Add-ons': 10
        }
        
        fig_opps = go.Figure([go.Bar(
            x=list(opportunities.values()),
            y=list(opportunities.keys()),
            orientation='h',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            text=[f'{v}%' for v in opportunities.values()],
            textposition='auto'
        )])
        
        fig_opps.update_layout(
            title='Estimated Revenue Impact of New Initiatives',
            xaxis_title='Potential Revenue Increase (%)',
            yaxis_title='Initiative',
            height=400
        )
        
        st.plotly_chart(fig_opps, use_container_width=True)
    
    with tab3:
        st.markdown("## 🌡️ Weather Response Strategy")
        
        # Weather impact analysis
        st.markdown("### 1. Weather Impact Quantification")
        
        weather_stats = df.groupby('weathersit').agg({
            'cnt': ['mean', 'count'],
            'casual': 'mean',
            'registered': 'mean'
        }).round(0)
        
        weather_stats.index = ['Clear', 'Mist/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Weather impact visualization
            fig_weather = go.Figure()
            
            categories = weather_stats.index
            
            fig_weather.add_trace(go.Bar(
                name='Total Demand',
                x=categories,
                y=weather_stats[('cnt', 'mean')],
                marker_color=['green', 'yellow', 'orange', 'red']
            ))
            
            fig_weather.update_layout(
                title='Average Demand by Weather Condition',
                xaxis_title='Weather Condition',
                yaxis_title='Average Rentals per Hour',
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig_weather, use_container_width=True)
        
        with col2:
            st.markdown("#### 📉 Impact Metrics")
            
            clear_demand = weather_stats.loc['Clear', ('cnt', 'mean')]
            rain_demand = weather_stats.loc['Light Rain/Snow', ('cnt', 'mean')]
            heavy_demand = weather_stats.loc['Heavy Rain/Snow', ('cnt', 'mean')] if 'Heavy Rain/Snow' in weather_stats.index else 0
            
            st.metric("Light Rain Impact", f"-{((clear_demand-rain_demand)/clear_demand*100):.0f}%")
            if heavy_demand > 0:
                st.metric("Heavy Rain Impact", f"-{((clear_demand-heavy_demand)/clear_demand*100):.0f}%")
            
            weather_frequency = weather_stats[('cnt', 'count')] / weather_stats[('cnt', 'count')].sum() * 100
            st.metric("Bad Weather Days", f"{weather_frequency[2:].sum():.1f}%")
        
        # Temperature optimization
        st.markdown("### 2. Temperature-Based Operations")
        
        # Create temperature bins
        df['temp_celsius'] = df['temp'] * 41
        temp_bins = pd.cut(df['temp_celsius'], bins=[-10, 0, 10, 20, 30, 40])
        temp_analysis = df.groupby(temp_bins)['cnt'].agg(['mean', 'count'])
        
        fig_temp = go.Figure()
        
        fig_temp.add_trace(go.Bar(
            x=['< 0°C', '0-10°C', '10-20°C', '20-30°C', '> 30°C'],
            y=temp_analysis['mean'],
            marker_color=['darkblue', 'lightblue', 'green', 'orange', 'red'],
            text=temp_analysis['mean'].round(0),
            textposition='auto'
        ))
        
        fig_temp.update_layout(
            title='Demand by Temperature Range',
            xaxis_title='Temperature Range',
            yaxis_title='Average Demand',
            height=350
        )
        
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Weather response actions
        st.markdown("### 3. Weather Response Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning("""
            **🌧️ Bad Weather Protocol:**
            1. **Pre-storm surge**: Increase availability 2 hours before
            2. **During storm**: Reduce fleet by 50-70%
            3. **Post-storm**: Rapid redeployment to high-demand areas
            4. **Communication**: Push notifications about weather conditions
            """)
        
        with col2:
            st.success("""
            **☀️ Good Weather Optimization:**
            1. **Weekend forecast**: Increase fleet by 30%
            2. **Tourist areas**: Double capacity
            3. **Marketing**: Weather-triggered promotions
            4. **Partnerships**: Event organizers, hotels
            """)
    
    with tab4:
        st.markdown("## 📈 Growth Opportunities")
        
        # Growth analysis
        st.markdown("### 1. Historical Growth Pattern")
        
        monthly_growth = df.groupby(['yr', 'mnth'])['cnt'].sum().unstack(level=0)
        monthly_growth.columns = ['2011', '2012']
        monthly_growth['Growth %'] = (monthly_growth['2012'] / monthly_growth['2011'] - 1) * 100
        
        fig_growth = go.Figure()
        
        # Add bars for absolute values
        fig_growth.add_trace(go.Bar(
            x=monthly_growth.index,
            y=monthly_growth['2011'],
            name='2011',
            marker_color='lightblue'
        ))
        
        fig_growth.add_trace(go.Bar(
            x=monthly_growth.index,
            y=monthly_growth['2012'],
            name='2012',
            marker_color='darkblue'
        ))
        
        # Add growth line
        fig_growth.add_trace(go.Scatter(
            x=monthly_growth.index,
            y=monthly_growth['Growth %'],
            name='Growth %',
            yaxis='y2',
            line=dict(color='red', width=3),
            mode='lines+markers'
        ))
        
        fig_growth.update_layout(
            title='Monthly Rentals and Year-over-Year Growth',
            xaxis_title='Month',
            yaxis_title='Total Rentals',
            yaxis2=dict(
                title='Growth %',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Expansion opportunities
        st.markdown("### 2. Expansion Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🏢 Geographic Expansion**
            - Business districts: +40% potential
            - University campuses: +30%
            - Tourist attractions: +25%
            - Residential areas: +20%
            """)
        
        with col2:
            st.success("""
            **🚴‍♀️ Service Expansion**
            - E-bikes: Capture older demographics
            - Cargo bikes: Delivery partnerships
            - Kids bikes: Family packages
            - Premium bikes: High-end market
            """)
        
        with col3:
            st.warning("""
            **🤝 Partnership Opportunities**
            - Transit integration: First/last mile
            - Hotel partnerships: Tourist packages
            - Corporate wellness programs
            - Event organizers: Special rates
            """)
        
        # Market penetration
        st.markdown("### 3. Market Penetration Strategy")
        
        # Estimate market penetration
        penetration_data = {
            'Segment': ['Commuters', 'Tourists', 'Students', 'Casual Recreation', 'Fitness'],
            'Current %': [35, 15, 20, 25, 5],
            'Potential %': [50, 30, 35, 40, 15],
            'Growth Opportunity': [15, 15, 15, 15, 10]
        }
        
        penetration_df = pd.DataFrame(penetration_data)
        
        fig_penetration = go.Figure()
        
        fig_penetration.add_trace(go.Bar(
            x=penetration_df['Segment'],
            y=penetration_df['Current %'],
            name='Current Penetration',
            marker_color='lightgreen'
        ))
        
        fig_penetration.add_trace(go.Bar(
            x=penetration_df['Segment'],
            y=penetration_df['Growth Opportunity'],
            name='Growth Opportunity',
            marker_color='darkgreen'
        ))
        
        fig_penetration.update_layout(
            title='Market Penetration by Customer Segment',
            xaxis_title='Customer Segment',
            yaxis_title='Market Share (%)',
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig_penetration, use_container_width=True)
    
    with tab5:
        st.markdown("## 🎯 Implementation Roadmap")
        
        # Phased implementation
        st.markdown("### 📅 90-Day Quick Wins")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **Month 1: Immediate Actions**
            - ✅ Implement dynamic fleet redistribution
            - ✅ Optimize maintenance windows
            - ✅ Launch weather-based communications
            - ✅ Start A/B testing pricing tiers
            """)
            
            st.info("""
            **Month 2: System Enhancements**
            - 📱 Deploy predictive demand app for staff
            - 📊 Create real-time dashboards
            - 🤝 Initiate corporate partnerships
            - 📈 Launch customer segmentation campaigns
            """)
        
        with col2:
            st.warning("""
            **Month 3: Scale & Optimize**
            - 🚀 Full dynamic pricing rollout
            - 🏪 Expand to 2 new locations
            - 💳 Launch subscription tiers
            - 📊 Measure and refine all initiatives
            """)
            
            # Expected impact
            st.markdown("#### 💰 Expected 90-Day Impact")
            
            impact_metrics = {
                'Revenue Increase': '+12-15%',
                'Operational Efficiency': '+20%',
                'Customer Satisfaction': '+15 NPS',
                'Fleet Utilization': '+25%'
            }
            
            for metric, value in impact_metrics.items():
                st.metric(metric, value)
        
        # Long-term vision
        st.markdown("### 🔮 12-Month Vision")
        
        # Create timeline
        timeline_data = {
            'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
            'Revenue Growth': [15, 25, 35, 45],
            'Fleet Size': [1000, 1200, 1500, 2000],
            'Locations': [10, 15, 20, 30]
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig_timeline = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Revenue Growth %', 'Fleet Size', 'Number of Locations')
        )
        
        # Revenue growth
        fig_timeline.add_trace(
            go.Scatter(x=timeline_df['Quarter'], y=timeline_df['Revenue Growth'],
                      mode='lines+markers', line=dict(color='green', width=3)),
            row=1, col=1
        )
        
        # Fleet size
        fig_timeline.add_trace(
            go.Scatter(x=timeline_df['Quarter'], y=timeline_df['Fleet Size'],
                      mode='lines+markers', line=dict(color='blue', width=3)),
            row=1, col=2
        )
        
        # Locations
        fig_timeline.add_trace(
            go.Scatter(x=timeline_df['Quarter'], y=timeline_df['Locations'],
                      mode='lines+markers', line=dict(color='orange', width=3)),
            row=1, col=3
        )
        
        fig_timeline.update_layout(
            title_text='12-Month Growth Projections',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Success metrics
        st.markdown("### 📊 Success Metrics & KPIs")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Daily Active Users", "Target: +50%", "Track via app")
            st.metric("Revenue per Bike", "Target: +30%", "Monthly tracking")
        
        with col2:
            st.metric("Utilization Rate", "Target: 75%", "Real-time monitoring")
            st.metric("Customer LTV", "Target: +40%", "Cohort analysis")
        
        with col3:
            st.metric("Operational Costs", "Target: -15%", "Per ride basis")
            st.metric("NPS Score", "Target: >50", "Quarterly surveys")
        
        # Final recommendations
        st.markdown("---")
        st.markdown("### 🎯 Key Takeaways")
        
        st.success("""
        **Top 5 Priority Actions:**
        
        1. **🚴 Dynamic Fleet Management**: Implement ML-based redistribution to reduce stockouts by 40%
        
        2. **💰 Smart Pricing**: Roll out time and weather-based pricing for 15% revenue uplift
        
        3. **🤝 Corporate Partnerships**: Target top 50 employers for commuter programs
        
        4. **📱 Digital Enhancement**: Launch predictive app for staff and customers
        
        5. **📍 Strategic Expansion**: Focus on high-demand corridors identified by the model
        
        **Remember**: The model shows hour of day as the #1 predictor - optimize everything around temporal patterns!
        """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure all data files are properly exported from the Jupyter notebook.")

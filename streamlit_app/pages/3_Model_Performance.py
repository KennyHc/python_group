import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import joblib

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

# Title
st.title("Model Performance Analysis")
st.markdown("Compare different models and analyze prediction accuracy")

# Load data
@st.cache_data
def load_model_data():
    model_results = pd.read_csv('./data/model_results.csv', index_col=0)
    tuned_results = pd.read_csv('./data/tuned_model_results.csv', index_col=0)
    predictions = pd.read_csv('./data/predictions.csv')
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    # Try to load feature importance
    try:
        feature_importance = pd.read_csv('./data/feature_importance.csv')
    except:
        feature_importance = None
    
    return model_results, tuned_results, predictions, feature_importance

@st.cache_resource
def load_model():
    try:
        return joblib.load('./models/best_model.pkl')
    except:
        return None

try:
    model_results, tuned_results, predictions, feature_importance = load_model_data()
    best_model = load_model()
    
    # Model comparison section
    st.markdown("## Model Comparison")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Overview",
        "Detailed Metrics",
        "Prediction Analysis",
        "Feature Importance"
    ])
    
    with tab1:
        st.markdown("### Model Performance Overview")
        
        # Combine baseline and tuned results
        all_results = pd.concat([model_results, tuned_results])
        
        # Filter for display - only show key metrics
        if 'RMSE' in all_results.columns and 'R²' in all_results.columns:
            display_cols = ['RMSE', 'R²']
            if 'MAE' in all_results.columns:
                display_cols.append('MAE')
            if 'MAPE' in all_results.columns:
                display_cols.append('MAPE')
        else:
            display_cols = all_results.columns.tolist()
        
        # Model performance comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE comparison
            fig_rmse = go.Figure()
            
            # Separate baseline and tuned models
            baseline_models = [m for m in all_results.index if 'tuned' not in m]
            tuned_models = [m for m in all_results.index if 'tuned' in m]
            
            # Plot baseline models
            if baseline_models:
                baseline_data = all_results.loc[baseline_models]
                fig_rmse.add_trace(go.Bar(
                    x=baseline_models,
                    y=baseline_data['RMSE'],
                    name='Baseline',
                    marker_color='lightblue',
                    text=baseline_data['RMSE'].round(1),
                    textposition='auto'
                ))
            
            # Plot tuned models
            if tuned_models:
                tuned_data = all_results.loc[tuned_models]
                fig_rmse.add_trace(go.Bar(
                    x=tuned_models,
                    y=tuned_data['RMSE'],
                    name='Tuned',
                    marker_color='darkgreen',
                    text=tuned_data['RMSE'].round(1),
                    textposition='auto'
                ))
            
            fig_rmse.update_layout(
                title='Model RMSE Comparison (Lower is Better)',
                xaxis_title='Model',
                yaxis_title='RMSE',
                showlegend=True,
                xaxis_tickangle=-45,
                height=400
            )
            
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            # R² comparison
            fig_r2 = go.Figure()
            
            # Plot baseline models
            if baseline_models:
                baseline_data = all_results.loc[baseline_models]
                fig_r2.add_trace(go.Bar(
                    x=baseline_models,
                    y=baseline_data['R²'],
                    name='Baseline',
                    marker_color='lightcoral',
                    text=baseline_data['R²'].round(3),
                    textposition='auto'
                ))
            
            # Plot tuned models
            if tuned_models:
                tuned_data = all_results.loc[tuned_models]
                fig_r2.add_trace(go.Bar(
                    x=tuned_models,
                    y=tuned_data['R²'],
                    name='Tuned',
                    marker_color='darkred',
                    text=tuned_data['R²'].round(3),
                    textposition='auto'
                ))
            
            fig_r2.update_layout(
                title='Model R² Comparison (Higher is Better)',
                xaxis_title='Model',
                yaxis_title='R² Score',
                showlegend=True,
                xaxis_tickangle=-45,
                height=400
            )
            
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # Best model highlight
        best_model_name = tuned_results['RMSE'].idxmin()
        best_rmse = tuned_results.loc[best_model_name, 'RMSE']
        best_r2 = tuned_results.loc[best_model_name, 'R²']
        
        st.success(f"""
        🏆 **Best Model: {best_model_name}**
        - RMSE: {best_rmse:.2f} bikes
        - R² Score: {best_r2:.4f} ({best_r2*100:.2f}% variance explained)
        - Average prediction error: ±{best_rmse:.0f} bikes per hour
        """)
        
        # Model improvement analysis
        st.markdown("### 📈 Model Improvement Through Tuning")
        
        improvement_data = []
        for model_base in ['XGBoost', 'LightGBM', 'CatBoost']:
            if model_base in baseline_models:
                baseline_rmse = all_results.loc[model_base, 'RMSE']
                tuned_model = f"{model_base} (tuned)"
                if tuned_model in tuned_models:
                    tuned_rmse = all_results.loc[tuned_model, 'RMSE']
                    improvement = ((baseline_rmse - tuned_rmse) / baseline_rmse * 100)
                    improvement_data.append({
                        'Model': model_base,
                        'Baseline RMSE': baseline_rmse,
                        'Tuned RMSE': tuned_rmse,
                        'Improvement %': improvement
                    })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            
            fig_improvement = go.Figure()
            
            x = np.arange(len(improvement_df))
            width = 0.35
            
            fig_improvement.add_trace(go.Bar(
                x=x - width/2,
                y=improvement_df['Baseline RMSE'],
                name='Baseline',
                marker_color='lightcoral',
                text=improvement_df['Baseline RMSE'].round(1),
                textposition='auto'
            ))
            
            fig_improvement.add_trace(go.Bar(
                x=x + width/2,
                y=improvement_df['Tuned RMSE'],
                name='Tuned',
                marker_color='lightgreen',
                text=improvement_df['Tuned RMSE'].round(1),
                textposition='auto'
            ))
            
            fig_improvement.update_layout(
                title='Impact of Hyperparameter Tuning',
                xaxis=dict(tickmode='array', tickvals=x, ticktext=improvement_df['Model']),
                yaxis_title='RMSE',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_improvement, use_container_width=True)
            
            # Improvement summary
            col1, col2, col3 = st.columns(3)
            for i, row in improvement_df.iterrows():
                with [col1, col2, col3][i]:
                    color = "green" if row['Improvement %'] > 0 else "red"
                    st.metric(
                        row['Model'],
                        f"{row['Tuned RMSE']:.1f}",
                        f"{row['Improvement %']:+.1f}%",
                        delta_color="normal" if row['Improvement %'] > 0 else "inverse"
                    )
    
    with tab2:
        st.markdown("### 🎯 Detailed Performance Metrics")
        
        # Select models to compare
        selected_models = st.multiselect(
            "Select models to compare:",
            all_results.index.tolist(),
            default=[best_model_name] if best_model_name else all_results.index[:3].tolist()
        )
        
        if selected_models:
            comparison_df = all_results.loc[selected_models, display_cols].T
            
            # Create radar chart for multi-metric comparison
            metrics_for_radar = ['RMSE', 'MAE', 'R²']
            available_metrics = [m for m in metrics_for_radar if m in comparison_df.index]
            
            if len(available_metrics) >= 3:
                fig_radar = go.Figure()
                
                for model in comparison_df.columns:
                    # Normalize metrics for radar chart (0-1 scale)
                    values = []
                    for metric in available_metrics:
                        val = comparison_df.loc[metric, model]
                        if metric in ['RMSE', 'MAE']:  # Lower is better
                            normalized = 1 - (val - comparison_df.loc[metric].min()) / (comparison_df.loc[metric].max() - comparison_df.loc[metric].min())
                        else:  # Higher is better (R²)
                            normalized = (val - comparison_df.loc[metric].min()) / (comparison_df.loc[metric].max() - comparison_df.loc[metric].min())
                        values.append(normalized)
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=available_metrics,
                        fill='toself',
                        name=model
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Model Performance Radar Chart (Normalized)"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Detailed metrics table
            st.markdown("#### 📊 Metrics Comparison Table")
            
            # Format the dataframe for display
            formatted_df = comparison_df.copy()
            for col in formatted_df.columns:
                if 'RMSE' in formatted_df.index:
                    formatted_df.loc['RMSE', col] = f"{formatted_df.loc['RMSE', col]:.2f}"
                if 'MAE' in formatted_df.index:
                    formatted_df.loc['MAE', col] = f"{formatted_df.loc['MAE', col]:.2f}"
                if 'R²' in formatted_df.index:
                    formatted_df.loc['R²', col] = f"{formatted_df.loc['R²', col]:.4f}"
                if 'MAPE' in formatted_df.index:
                    formatted_df.loc['MAPE', col] = f"{formatted_df.loc['MAPE', col]:.1f}%"
            
            st.dataframe(
                formatted_df,
                use_container_width=True,
                column_config={
                    col: st.column_config.TextColumn(col, width="medium")
                    for col in formatted_df.columns
                }
            )
            
            # Metric explanations
            with st.expander("📖 Metric Explanations"):
                st.markdown("""
                - **RMSE (Root Mean Square Error)**: Average prediction error in bikes/hour. Lower is better.
                - **MAE (Mean Absolute Error)**: Average absolute prediction error. Less sensitive to outliers than RMSE.
                - **R² (R-squared)**: Proportion of variance explained by the model. 1.0 = perfect, 0 = baseline.
                - **MAPE (Mean Absolute Percentage Error)**: Average percentage error. Shows relative accuracy.
                """)
        
        # Business context
        st.markdown("### 💼 Business Context")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_demand = predictions['actual'].mean()
            error_pct = (best_rmse / avg_demand) * 100
            
            st.info(f"""
            **Model Performance in Context:**
            - Average hourly demand: {avg_demand:.0f} bikes
            - Prediction error: ±{best_rmse:.0f} bikes
            - Error as % of average: {error_pct:.1f}%
            - 95% confidence interval: ±{best_rmse*1.96:.0f} bikes
            """)
        
        with col2:
            capacity_buffer = best_rmse * 2  # 2 standard deviations
            
            st.warning(f"""
            **Operational Recommendations:**
            - Maintain {capacity_buffer:.0f} extra bikes as buffer
            - Focus on peak hours (8-9 AM, 5-6 PM)
            - Monitor weather forecasts for demand spikes
            - Consider {error_pct:.0f}% margin in capacity planning
            """)
    
    with tab3:
        st.markdown("### 🔬 Prediction Analysis")
        
        # Actual vs Predicted
        st.markdown("#### Actual vs Predicted Values")
        
        # Add perfect prediction line
        fig_pred = go.Figure()
        
        # Perfect prediction line
        min_val = min(predictions['actual'].min(), predictions['predicted'].min())
        max_val = max(predictions['actual'].max(), predictions['predicted'].max())
        
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash'),
            showlegend=True
        ))
        
        # Actual predictions
        fig_pred.add_trace(go.Scatter(
            x=predictions['actual'],
            y=predictions['predicted'],
            mode='markers',
            name='Predictions',
            marker=dict(
                size=5,
                color=predictions['residual'].abs(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Absolute Error")
            ),
            text=[f"Actual: {a}<br>Predicted: {p:.0f}<br>Error: {r:.0f}" 
                  for a, p, r in zip(predictions['actual'], predictions['predicted'], predictions['residual'])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig_pred.update_layout(
            title='Actual vs Predicted Bike Rentals',
            xaxis_title='Actual Rentals',
            yaxis_title='Predicted Rentals',
            height=500
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Residual analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Residual distribution
            fig_residual_dist = go.Figure()
            
            fig_residual_dist.add_trace(go.Histogram(
                x=predictions['residual'],
                nbinsx=50,
                name='Residuals',
                marker_color='lightblue'
            ))
            
            # Add normal distribution overlay
            mean_residual = predictions['residual'].mean()
            std_residual = predictions['residual'].std()
            
            fig_residual_dist.update_layout(
                title='Residual Distribution',
                xaxis_title='Prediction Error (Actual - Predicted)',
                yaxis_title='Frequency',
                showlegend=False
            )
            
            st.plotly_chart(fig_residual_dist, use_container_width=True)
            
            # Residual stats
            st.caption(f"""
            **Residual Statistics:**
            - Mean: {mean_residual:.2f} (should be ~0)
            - Std Dev: {std_residual:.2f}
            - Skewness: {predictions['residual'].skew():.2f}
            """)
        
        with col2:
            # Residuals over time
            fig_residual_time = go.Figure()
            
            fig_residual_time.add_trace(go.Scatter(
                x=predictions['date'],
                y=predictions['residual'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=predictions['residual'],
                    colorscale='RdBu',
                    cmid=0,
                    showscale=True,
                    colorbar=dict(title="Residual")
                )
            ))
            
            # Add zero line
            fig_residual_time.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_residual_time.update_layout(
                title='Residuals Over Time',
                xaxis_title='Date',
                yaxis_title='Prediction Error',
                height=400
            )
            
            st.plotly_chart(fig_residual_time, use_container_width=True)
        
        # Error analysis by conditions
        st.markdown("#### 🎯 Error Analysis by Conditions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error by hour
            hourly_mae = predictions.groupby('hour')['residual'].apply(lambda x: x.abs().mean()).reset_index()
            hourly_mae.columns = ['hour', 'mae']
            
            fig_hour_error = go.Figure([go.Bar(
                x=hourly_mae['hour'],
                y=hourly_mae['mae'],
                marker_color=hourly_mae['mae'],
                marker_colorscale='Reds',
                text=hourly_mae['mae'].round(0),
                textposition='auto'
            )])
            
            fig_hour_error.update_layout(
                title='Average Prediction Error by Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Mean Absolute Error',
                showlegend=False
            )
            
            st.plotly_chart(fig_hour_error, use_container_width=True)
        
        with col2:
            # Error by weather
            weather_mae = predictions.groupby('weather')['residual'].apply(lambda x: x.abs().mean()).reset_index()
            weather_mae.columns = ['weather', 'mae']
            weather_mae['weather_name'] = weather_mae['weather'].map({
                1: 'Clear',
                2: 'Mist/Cloudy',
                3: 'Light Rain/Snow',
                4: 'Heavy Rain/Snow'
            })
            
            fig_weather_error = go.Figure([go.Bar(
                x=weather_mae['weather_name'],
                y=weather_mae['mae'],
                marker_color=['green', 'yellow', 'orange', 'red'][:len(weather_mae)],
                text=weather_mae['mae'].round(0),
                textposition='auto'
            )])
            
            fig_weather_error.update_layout(
                title='Average Prediction Error by Weather',
                xaxis_title='Weather Condition',
                yaxis_title='Mean Absolute Error',
                showlegend=False
            )
            
            st.plotly_chart(fig_weather_error, use_container_width=True)
        
        # Worst predictions analysis
        st.markdown("#### ⚠️ Largest Prediction Errors")
        
        # Find worst over and under predictions
        worst_over = predictions.nlargest(5, 'residual')[['date', 'actual', 'predicted', 'residual', 'weather', 'temp']]
        worst_under = predictions.nsmallest(5, 'residual')[['date', 'actual', 'predicted', 'residual', 'weather', 'temp']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Largest Over-predictions:**")
            st.dataframe(
                worst_over.round({'predicted': 0, 'residual': 0, 'temp': 2}),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.markdown("**Largest Under-predictions:**")
            st.dataframe(
                worst_under.round({'predicted': 0, 'residual': 0, 'temp': 2}),
                use_container_width=True,
                hide_index=True
            )
    
    with tab4:
        st.markdown("### 🌟 Feature Importance Analysis")
        
        if feature_importance is not None and len(feature_importance) > 0:
            # Top features bar chart
            top_n = st.slider("Number of top features to display", 10, 30, 20)
            
            top_features = feature_importance.head(top_n)
            
            # Create color coding for feature types
            def get_feature_color(feature_name):
                if 'temp' in feature_name.lower():
                    return 'darkgreen'
                elif feature_name in ['hr', 'mnth', 'season', 'weekday', 'yr']:
                    return 'darkblue'
                elif any(x in feature_name for x in ['is_', 'interaction', '_sin', '_cos']):
                    return 'darkorange'
                else:
                    return 'purple'
            
            colors = [get_feature_color(f) for f in top_features['Feature']]
            
            fig_importance = go.Figure([go.Bar(
                x=top_features['Importance'],
                y=top_features['Feature'],
                orientation='h',
                marker_color=colors,
                text=top_features['Importance'].round(3),
                textposition='auto'
            )])
            
            fig_importance.update_layout(
                title=f'Top {top_n} Most Important Features',
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                height=600,
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Feature importance by category
            st.markdown("#### 📊 Feature Importance by Category")
            
            # Categorize features
            categories = {
                'Temporal': ['hr', 'mnth', 'season', 'weekday', 'yr', 'holiday', 'workingday'],
                'Weather': ['temp', 'atemp', 'hum', 'windspeed', 'weathersit'],
                'Engineered': []
            }
            
            # Identify engineered features
            for feature in feature_importance['Feature']:
                if not any(feature in cat_list for cat_list in categories.values()):
                    categories['Engineered'].append(feature)
            
            # Calculate importance by category
            category_importance = {}
            for category, features in categories.items():
                cat_features = feature_importance[feature_importance['Feature'].isin(features)]
                if len(cat_features) > 0:
                    category_importance[category] = {
                        'Total Importance': cat_features['Importance'].sum(),
                        'Avg Importance': cat_features['Importance'].mean(),
                        'Feature Count': len(cat_features),
                        'Top Feature': cat_features.iloc[0]['Feature'] if len(cat_features) > 0 else 'N/A'
                    }
            
            # Display category analysis
            col1, col2, col3 = st.columns(3)
            
            for i, (category, stats) in enumerate(category_importance.items()):
                with [col1, col2, col3][i]:
                    st.metric(
                        category,
                        f"{stats['Total Importance']:.3f}",
                        f"{stats['Feature Count']} features"
                    )
                    st.caption(f"Top: {stats['Top Feature']}")
            
            # Feature importance insights
            st.markdown("#### 💡 Key Feature Insights")
            
            top_feature = feature_importance.iloc[0]['Feature']
            top_importance = feature_importance.iloc[0]['Importance']
            
            st.info(f"""
            **Top Insights:**
            - **Most Important**: `{top_feature}` (importance: {top_importance:.3f})
            - **Temporal features** dominate predictions, confirming strong time-based patterns
            - **Temperature** is the most important weather factor
            - **Engineered features** add significant value, validating feature engineering efforts
            """)
            
            # Feature correlation with target
            if st.checkbox("Show feature correlations with target"):
                # This would require the original dataframe, so we'll show a message
                st.info("""
                Feature correlations help understand linear relationships between features and the target.
                The feature importance from tree-based models captures both linear and non-linear relationships.
                """)
        else:
            st.warning("Feature importance data not available. Tree-based models typically provide this information.")
            
            st.info("""
            **Why Feature Importance Matters:**
            - Identifies which factors most influence bike rental demand
            - Helps focus business decisions on high-impact areas
            - Validates the feature engineering process
            - Guides future data collection efforts
            """)
    
    # Model card section
    st.markdown("---")
    st.markdown("## 📋 Model Card")
    
    with st.expander("View Model Documentation"):
        st.markdown(f"""
        ### Model Information
        - **Model Type**: {best_model_name}
        - **Training Period**: January 2011 - September 2012
        - **Test Period**: October 2012 - December 2012
        - **Target Variable**: Total bike rentals per hour (cnt)
        
        ### Performance Summary
        - **RMSE**: {best_rmse:.2f} bikes/hour
        - **R² Score**: {best_r2:.4f}
        - **MAE**: {tuned_results.loc[best_model_name, 'MAE']:.2f} if available
        - **Business Impact**: Prediction error represents ~{(best_rmse/predictions['actual'].mean()*100):.1f}% of average demand
        
        ### Use Cases
        - Hourly capacity planning
        - Staff scheduling optimization
        - Maintenance timing
        - Demand forecasting for new locations
        
        ### Limitations
        - Trained on 2011-2012 data (may need retraining for current patterns)
        - Weather dependency requires accurate weather forecasts
        - Does not account for special events or anomalies
        - Performance may degrade during extreme weather conditions
        
        ### Recommendations
        - Retrain quarterly with new data
        - Monitor prediction accuracy weekly
        - Maintain {best_rmse*2:.0f} bikes as safety buffer
        - Consider ensemble predictions during peak hours
        """)

except Exception as e:
    st.error(f"Error loading model data: {str(e)}")
    st.info("Please ensure the model files are properly exported from the Jupyter notebook.")
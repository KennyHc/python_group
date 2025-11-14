import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
from pathlib import Path

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

# Title
st.title("Real-Time Demand Predictions")
st.markdown("Make predictions for bike rental demand using the trained model")

script_dir = Path(__file__).parent.parent

@st.cache_resource
def load_model():
    """Loads the pickled model file."""
    try:
        model_path = script_dir / 'models/best_model.pkl'
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Error: The model file ('best_model.pkl') was not found.")
        st.info("Please check the 'models' folder in your repo.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

@st.cache_data
def load_feature_names():
    """Loads the feature names from a CSV."""
    try:
        features_path = script_dir / 'data/feature_names.csv'
        features = pd.read_csv(features_path)
        return features['features'].tolist()
    except FileNotFoundError:
        st.error("Error: The features file ('feature_names.csv') was not found.")
        st.info("Please check the 'data' folder in your repo.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading feature names: {e}")
        return None

@st.cache_data
def load_historical_data():
    """Loads the main processed dataset."""
    try:
        data_path = script_dir / 'data/bike_data_processed.csv'
        df = pd.read_csv(data_path)
        df['dteday'] = pd.to_datetime(df['dteday'])
        return df
    except FileNotFoundError:
        st.error("Error: The main data file ('bike_data_processed.csv') was not found.")
        st.info("Please check the 'data' folder in your repo.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading historical data: {e}")
        return None

# --- Feature Engineering Function (Refactored) ---
# This function can now be used by Tab 1 and Tab 3
def create_feature_dict(prediction_date, hour, temp_celsius, atemp_celsius, humidity_percent, windspeed_kmh, weather_condition_str, is_holiday_bool):
    """
    Creates the complete feature dictionary required by the model
    from raw user inputs.
    """
    
    # Temporal features
    weekday = prediction_date.weekday()
    month = prediction_date.month
    year = 1 if prediction_date.year > 2011 else 0  # Simplified year encoding
    
    # Season
    if month in [12, 1, 2]:
        season = 1  # Winter
    elif month in [3, 4, 5]:
        season = 2  # Spring
    elif month in [6, 7, 8]:
        season = 3  # Summer
    else:
        season = 4  # Fall
        
    # Holiday & Working Day
    holiday = 1 if is_holiday_bool else 0
    is_working_day = 1 if weekday < 5 and not is_holiday_bool else 0
    
    # Weather
    weather_map = {
        "Clear, Few clouds": 1,
        "Mist + Cloudy": 2,
        "Light Snow/Rain": 3,
        "Heavy Rain/Snow": 4
    }
    weathersit = weather_map.get(weather_condition_str, 1) # Default to 1
    
    # Normalized weather features
    temp_normalized = temp_celsius / 41.0
    atemp_normalized = atemp_celsius / 50.0
    humidity_normalized = humidity_percent / 100.0
    windspeed_normalized = windspeed_kmh / 67.0
    
    # --- Engineered Features ---
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Binary features
    is_rush_hour = 1 if hour in [8, 9, 17, 18] else 0
    is_good_weather = 1 if weathersit == 1 else 0
    is_bad_weather = 1 if weathersit >= 3 else 0
    is_comfortable_temp = 1 if 15 <= temp_celsius <= 25 else 0
    is_weekend = 1 if weekday in [5, 6] else 0
    
    # Interaction features
    temp_humidity_interaction = temp_normalized * humidity_normalized
    feels_like_diff = temp_normalized - atemp_normalized
    temp_weather_interaction = temp_normalized * (1 if weathersit <= 2 else 0)
    
    # Return complete feature dictionary
    features = {
        'season': season,
        'yr': year,
        'mnth': month,
        'hr': hour,
        'holiday': holiday,
        'weekday': weekday,
        'workingday': is_working_day,
        'weathersit': weathersit,
        'temp': temp_normalized,
        'atemp': atemp_normalized,
        'hum': humidity_normalized,
        'windspeed': windspeed_normalized,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_rush_hour': is_rush_hour,
        'is_good_weather': is_good_weather,
        'is_bad_weather': is_bad_weather,
        'is_comfortable_temp': is_comfortable_temp,
        'temp_humidity_interaction': temp_humidity_interaction,
        'feels_like_diff': feels_like_diff,
        'temp_weather_interaction': temp_weather_interaction,
        'is_weekend': is_weekend
    }
    
    return features

# --- Main App Logic ---
try:
    model = load_model()
    feature_names = load_feature_names()
    historical_data = load_historical_data()
    
    if model is None:
        st.error("⚠️ Model not found! Please ensure the model file is exported from the notebook.")
        st.stop()
    if feature_names is None:
        st.error("⚠️ 'feature_names.csv' not found! This file is required.")
        st.stop()
        
    # Prediction modes
    tab1, tab2, tab3 = st.tabs([
        " Single Prediction",
        " Batch Predictions",
        " What-If Scenarios"
    ])
    
    # --- TAB 1: SINGLE PREDICTION (FIXED) ---
    with tab1:
        st.markdown("### Make a Single Prediction")
        st.info("Adjust the parameters below to predict bike rental demand for a specific hour")
        
        # Create input columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Date & Time")
            prediction_date = st.date_input(
                "Select date",
                value=datetime.now().date(),
                min_value=datetime(2011, 1, 1).date(),
                max_value=datetime(2025, 12, 31).date(),
                key="tab1_date"
            )
            hour = st.slider("Hour of day", 0, 23, datetime.now().hour, key="tab1_hour")
            is_holiday = st.checkbox("Is it a holiday?", value=False, key="tab1_holiday")
        
        with col2:
            st.markdown("#### Weather Conditions")
            temp_celsius = st.slider(
                "Temperature (°C)", -10.0, 40.0, 20.0, 0.5, key="tab1_temp"
            )
            atemp_celsius = st.slider(
                "Feels-like temperature (°C)", -15.0, 50.0, temp_celsius, 0.5, key="tab1_atemp"
            )
            humidity = st.slider(
                "Humidity (%)", 0, 100, 50, key="tab1_hum"
            )
            windspeed_kmh = st.slider(
                "Wind speed (km/h)", 0.0, 60.0, 10.0, 1.0, key="tab1_wind"
            )
        
        with col3:
            st.markdown("#### Weather Type")
            weather_condition = st.radio(
                "Select weather condition",
                options=["Clear, Few clouds", "Mist + Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"],
                key="tab1_weather"
            )
        
        # Create feature vector
        st.markdown("---")
        
        # 1. Get feature dictionary from our refactored function
        features = create_feature_dict(
            prediction_date=prediction_date,
            hour=hour,
            temp_celsius=temp_celsius,
            atemp_celsius=atemp_celsius,
            humidity_percent=humidity,
            windspeed_kmh=windspeed_kmh,
            weather_condition_str=weather_condition,
            is_holiday_bool=is_holiday
        )
        
        # 2. Create feature vector in correct order
        try:
            feature_vector = np.array([features.get(f, 0) for f in feature_names]).reshape(1, -1)
        except Exception as e:
            st.error(f"Error creating feature vector: {e}. Check 'feature_names.csv'.")
            st.stop()
            
        # 3. Make prediction
        prediction = model.predict(feature_vector)[0]
        prediction = max(0, int(prediction)) # Ensure non-negative
        
        # 4. Display prediction (This now updates live)
        col1_pred, col2_pred, col3_pred = st.columns([1, 2, 1])
        
        with col2_pred:
            st.success("### Prediction Result")
            st.metric(
                "Predicted Bike Demand",
                f"{prediction} bikes",
                f"for {prediction_date} at {hour}:00"
            )
            
            if prediction < 50: level, color = "Very Low", "🟢"
            elif prediction < 150: level, color = "Low", "🟡"
            elif prediction < 300: level, color = "Moderate", "🟠"
            elif prediction < 500: level, color = "High", "🔴"
            else: level, color = "Very High", "🟣"
            
            st.info(f"{color} Demand Level: **{level}**")
        
        # 5. Show similar historical patterns
        st.markdown("---")
        st.markdown("### 📊 Similar Historical Patterns")
        
        if historical_data is not None:
            # Find similar conditions (using feature dict values)
            similar_conditions = historical_data[
                (historical_data['hr'] == features['hr']) &
                (historical_data['weathersit'] == features['weathersit']) &
                (historical_data['workingday'] == features['workingday']) &
                (historical_data['season'] == features['season'])
            ]
            
            if len(similar_conditions) > 0:
                avg_demand = similar_conditions['cnt'].mean()
                std_demand = similar_conditions['cnt'].std()
                
                col1_hist, col2_hist, col3_hist = st.columns(3)
                col1_hist.metric("Historical Average", f"{avg_demand:.0f} bikes", f"±{std_demand:.0f}")
                percentile = (prediction / avg_demand * 100) if avg_demand > 0 else 100
                col2_hist.metric("Prediction vs History", f"{percentile:.0f}%", "of historical average")
                col3_hist.metric("Similar Days Found", f"{len(similar_conditions)}", "in dataset")
                
                # Show distribution
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=similar_conditions['cnt'], nbinsx=30, name='Historical Distribution', marker_color='lightblue'
                ))
                fig_hist.add_vline(
                    x=prediction, line_dash="dash", line_color="red", annotation_text="Your Prediction"
                )
                fig_hist.update_layout(
                    title=f'Historical Demand Distribution for Similar Conditions',
                    xaxis_title='Number of Rentals', yaxis_title='Frequency'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("No similar historical data found for these specific conditions.")

    # --- TAB 2: BATCH PREDICTIONS (Unchanged) ---
    with tab2:
        st.markdown("### 📊 Batch Predictions")
        st.info("Upload a CSV file with multiple scenarios to get batch predictions")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="File should contain columns matching the template."
        )
        
        if uploaded_file is not None:
            # This tab would need its own full implementation
            # to process the CSV, which is not trivial.
            st.warning("Batch processing logic is not fully implemented in this demo.")
            # ... (rest of the placeholder logic) ...
        
        # Template download
        st.markdown("#### Need a template?")
        template_data = {
            'date': ['2024-01-01', '2024-01-01'], 'hour': [8, 12],
            'temp_celsius': [10, 15], 'atemp_celsius': [9.5, 15],
            'humidity_percent': [60, 50], 'windspeed_kmh': [10, 5],
            'weather_condition_str': ['Clear, Few clouds', 'Mist + Cloudy'],
            'is_holiday_bool': [False, False]
        }
        template_df = pd.DataFrame(template_data)
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            label=" Download Template CSV",
            data=template_csv,
            file_name="batch_prediction_template.csv",
            mime="text/csv"
        )
    
    # --- TAB 3: WHAT-IF SCENARIOS (FIXED) ---
    with tab3:
        st.markdown("### 📈 What-If Scenario Analysis")
        st.info("Explore how changing one factor affects demand, holding all others constant.")
        
        # --- Base Scenario Inputs ---
        st.markdown("#### 1. Set Your Base Scenario")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            base_date = st.date_input(
                "Select date",
                value=datetime(2022, 7, 1).date(), # A summer weekday
                key="tab3_date"
            )
            base_hour = st.slider("Hour of day", 0, 23, 17, key="tab3_hour")
            base_holiday = st.checkbox("Is it a holiday?", value=False, key="tab3_holiday")
        
        with col2:
            base_temp = st.slider(
                "Temperature (°C)", -10.0, 40.0, 25.0, 0.5, key="tab3_temp"
            )
            base_atemp = st.slider(
                "Feels-like temperature (°C)", -15.0, 50.0, base_temp, 0.5, key="tab3_atemp"
            )
            base_humidity = st.slider(
                "Humidity (%)", 0, 100, 60, key="tab3_hum"
            )
            base_windspeed = st.slider(
                "Wind speed (km/h)", 0.0, 60.0, 15.0, 1.0, key="tab3_wind"
            )
            
        with col3:
            base_weather = st.radio(
                "Select weather condition",
                options=["Clear, Few clouds", "Mist + Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"],
                key="tab3_weather"
            )
            
        st.markdown("---")
        
        # --- Analysis Type ---
        st.markdown("#### 2. Choose Analysis Type")
        scenario_type = st.radio(
            "Select which variable to analyze",
            ["Temperature Impact", "Time of Day Impact", "Weather Conditions"],
            horizontal=True
        )

        # --- Run Analysis & Plot ---
        # This logic is now LIVE and uses the REAL model
        
        if scenario_type == "Temperature Impact":
            variable_range = np.arange(-5, 41, 1) # Test temps from -5°C to 40°C
            predictions = []
            
            for temp in variable_range:
                # Get features, overriding temp and atemp
                features = create_feature_dict(
                    prediction_date=base_date, hour=base_hour,
                    temp_celsius=temp, atemp_celsius=temp, # Use loop temp
                    humidity_percent=base_humidity, windspeed_kmh=base_windspeed,
                    weather_condition_str=base_weather, is_holiday_bool=base_holiday
                )
                feature_vector = np.array([features.get(f, 0) for f in feature_names]).reshape(1, -1)
                pred = model.predict(feature_vector)[0]
                predictions.append(max(0, int(pred)))
            
            # Create visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=variable_range, y=predictions, mode='lines',
                line=dict(width=3, color='red')
            ))
            fig.add_vline(x=base_temp, line_dash="dash", line_color="grey", annotation_text="Base Temp")
            fig.update_layout(
                title='Demand vs. Temperature',
                xaxis_title='Temperature (°C)',
                yaxis_title='Predicted Bike Demand'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif scenario_type == "Time of Day Impact":
            variable_range = list(range(24)) # Test all 24 hours
            predictions = []
            
            for hour in variable_range:
                # Get features, overriding hour
                features = create_feature_dict(
                    prediction_date=base_date, hour=hour, # Use loop hour
                    temp_celsius=base_temp, atemp_celsius=base_atemp,
                    humidity_percent=base_humidity, windspeed_kmh=base_windspeed,
                    weather_condition_str=base_weather, is_holiday_bool=base_holiday
                )
                feature_vector = np.array([features.get(f, 0) for f in feature_names]).reshape(1, -1)
                pred = model.predict(feature_vector)[0]
                predictions.append(max(0, int(pred)))
            
            # Create visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(x=variable_range, y=predictions, marker_color='lightblue'))
            fig.add_vline(x=base_hour, line_dash="dash", line_color="grey", annotation_text="Base Hour")
            fig.update_layout(
                title='Demand vs. Hour of Day',
                xaxis_title='Hour of Day',
                yaxis_title='Predicted Bike Demand',
                xaxis=dict(tickmode='linear', tick0=0, dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif scenario_type == "Weather Conditions":
            variable_range = ["Clear, Few clouds", "Mist + Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"]
            predictions = []
            
            for weather in variable_range:
                # Get features, overriding weather
                features = create_feature_dict(
                    prediction_date=base_date, hour=base_hour,
                    temp_celsius=base_temp, atemp_celsius=base_atemp,
                    humidity_percent=base_humidity, windspeed_kmh=base_windspeed,
                    weather_condition_str=weather, is_holiday_bool=base_holiday # Use loop weather
                )
                feature_vector = np.array([features.get(f, 0) for f in feature_names]).reshape(1, -1)
                pred = model.predict(feature_vector)[0]
                predictions.append(max(0, int(pred)))
            
            # Create visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(x=variable_range, y=predictions, marker_color='dodgerblue'))
            fig.update_layout(
                title='Demand vs. Weather Condition',
                xaxis_title='Weather Condition',
                yaxis_title='Predicted Bike Demand'
            )
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.exception(e)
    st.info("Please ensure all required files (model, features, data) are available.")
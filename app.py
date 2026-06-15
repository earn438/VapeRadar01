import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
MODEL_PATH = 'rf_model_unified.joblib'
MODEL_NAME = "Aurafarm AI"
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ/pub?output=csv"

st.set_page_config(page_title=MODEL_NAME, page_icon="🚭", layout="wide")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning(f"⚠️ Model file not found at {MODEL_PATH}.")
        return None

my_model = load_model()

# --- 2. DATA LOADING ---
@st.cache_data(ttl=30)
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        column_mapping = {
            "Unnamed: 0": "Timestamp", "tvoc": "TVOC", "eco2": "eCO2",
            "temp": "Temp", "humidity": "Humidity", "ch0": "CH0",
            "ch3": "CH3", "mq135": "MQ135", "2.5": "PM2.5", "10": "PM10"
        }
        df = df.rename(columns=column_mapping)
        if "Unnamed: 1" in df.columns: df = df.drop(columns=["Unnamed: 1"])
            
        if 'Timestamp' in df.columns:
            df['Display_Time'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
            df['Sort_Time'] = df['Display_Time'] + pd.Timedelta(hours=7)
            df = df.dropna(subset=['Display_Time'])
            df = df.sort_values(by='Sort_Time', ascending=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_sensor_data()

# --- 3. DASHBOARD UI ---
st.title(f"🚭 {MODEL_NAME} Dashboard")
if df.empty:
    st.warning("No data found.")
    st.stop()

latest = df.iloc[0].to_frame().T

# --- 4. RUN MODEL PREDICTION ---
if my_model:
    # Update this dictionary to include ALL columns the model expects
    mapping_dict = {
        'TVOC': 'col_2',
        'eCO2': 'col_3',
        'Temp': 'col_4',
        'Humidity': 'col_5',
        'PM2.5': 'col_6',
        'CH0': 'col_7',   # Assuming these map based on your earlier table
        'CH3': 'col_8',
        'MQ135': 'col_9'
    }
    
    # Update this selection list to match the keys above
    features = latest[['TVOC', 'eCO2', 'Temp', 'Humidity', 'PM2.5', 'CH0', 'CH3', 'MQ135']].rename(columns=mapping_dict)
    
    try:
        prediction = my_model.predict(features)[0]
        if prediction == 1:
            st.error("🚨 VAPE DETECTED: AI Model indicates vape particles!")
        else:
            st.success("✅ AIR QUALITY: Clean.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Model expects these feature names:", my_model.feature_names_in_)
else:
    st.info("ℹ️ Prediction system offline.")
# --- 5. VAPE DETECTION HISTORY ---
if my_model:
    st.subheader("⚠️ Detection History")
    
    # Run prediction on full history
    hist_features = df[['TVOC', 'eCO2', 'Temp', 'Humidity', 'PM2.5', 'CH0', 'CH3', 'MQ135']].rename(columns=mapping_dict)
    df['is_vape'] = my_model.predict(hist_features)
    
    # Filter for only detected rows
    vape_rows = df[df['is_vape'] == 1].copy()
    
    if not vape_rows.empty:
        # Create blocks: A new block starts if the gap is > 5 minutes
        vape_rows = vape_rows.sort_values('Display_Time')
        vape_rows['block'] = (vape_rows['Display_Time'].diff() > pd.Timedelta(minutes=5)).cumsum()
        
        # Group and reverse to show latest detections at the top
        grouped_vape = vape_rows.groupby('block')
        for _, group in reversed(list(grouped_vape)):
            start_time = group['Display_Time'].min().strftime('%H:%M')
            end_time = group['Display_Time'].max().strftime('%H:%M')
            date_str = group['Display_Time'].min().strftime('%Y-%m-%d')
            
            # If the start and end are the same (single reading), just show the time
            if start_time == end_time:
                time_range = f"at {start_time}"
            else:
                time_range = f"from {start_time} to {end_time}"
            
            st.markdown(
                f"<div style='color: #ff4b4b; font-weight: bold; padding: 10px; border-left: 5px solid #ff4b4b; background-color: #fff0f0; margin-bottom: 10px; border-radius: 4px;'>"
                f"🚨 Vape Detected: {date_str} {time_range}"
                f"</div>", 
                unsafe_allow_html=True
            )
    else:
        st.info("No vape events detected in the available data.")
# --- METRICS & GRAPHS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temp", f"{latest['Temp'].values[0]} °C")
col2.metric("Humidity", f"{latest['Humidity'].values[0]} %")
col3.metric("TVOC", f"{latest['TVOC'].values[0]} ppb")
col4.metric("PM 2.5", f"{latest['PM2.5'].values[0]} μg/m³")

st.caption(f"Last updated (Sensor Time): {latest['Display_Time'].values[0]}")
st.divider()

st.subheader("📈 Past 24 Hours Trends")
chart_data = df.sort_values(by='Sort_Time', ascending=True)
cutoff = chart_data['Sort_Time'].max() - pd.Timedelta(days=1)
chart_data = chart_data[chart_data['Sort_Time'] >= cutoff]
chart_data = chart_data.set_index('Sort_Time')

numeric_cols = chart_data.select_dtypes(include='number').columns
chart_data = chart_data[numeric_cols].resample('1min').mean().interpolate(method='time')

tab1, tab2, tab3 = st.tabs(["🌫️ Particles", "🌬️ Air Quality", "🌡️ Climate"])
with tab1: st.line_chart(chart_data[['PM2.5', 'PM10', 'MQ135']])
with tab2: st.line_chart(chart_data[['TVOC', 'eCO2']])
with tab3: st.line_chart(chart_data[['Temp', 'Humidity']])

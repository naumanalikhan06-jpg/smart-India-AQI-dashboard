import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Indian Weather AI Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    csv_path = Path(__file__).parent / "indian_weather_data.csv"
    if not csv_path.exists():
        st.error("❌ CSV file not found. Make sure 'indian_weather_data.csv' is in the same folder as project.py")
        st.stop()
    return pd.read_csv(csv_path)

df = load_data()

# ----------------------------
# AQI CALCULATION FUNCTION (US-EPA PM2.5)
# ----------------------------
def calculate_aqi_pm25(pm25):
    """Returns AQI based on PM2.5 value using US-EPA breakpoints"""
    if pm25 <= 12:
        aqi = (50-0)/(12-0)*(pm25-0) + 0
    elif pm25 <= 35.4:
        aqi = (100-51)/(35.4-12.1)*(pm25-12.1) + 51
    elif pm25 <= 55.4:
        aqi = (150-101)/(55.4-35.5)*(pm25-35.5) + 101
    elif pm25 <= 150.4:
        aqi = (200-151)/(150.4-55.5)*(pm25-55.5) + 151
    elif pm25 <= 250.4:
        aqi = (300-201)/(250.4-150.5)*(pm25-150.5) + 201
    elif pm25 <= 350.4:
        aqi = (400-301)/(350.4-250.5)*(pm25-250.5) + 301
    elif pm25 <= 500.4:
        aqi = (500-401)/(500.4-350.5)*(pm25-350.5) + 401
    else:
        aqi = 500
    return round(aqi)

# Apply AQI calculation
df["aqi"] = df["pm2_5"].apply(calculate_aqi_pm25)

# ----------------------------
# APP START
# ----------------------------
st.title("🌦️ Indian Weather Interactive AI Dashboard")
st.markdown("A fully interactive data exploration, visualization & AI prediction dashboard.")

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("🔍 Filters")
cities = st.sidebar.multiselect(
    "Select City",
    sorted(df["city"].unique()),
    default=df["city"].unique()
)

df_filtered = df[df["city"].isin(cities)]

# ----------------------------
# KPI CARDS
# ----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Temperature 🌡️", f"{df_filtered['temperature'].mean():.2f} °C")
col2.metric("Avg Humidity 💧", f"{df_filtered['humidity'].mean():.2f} %")
col3.metric("Avg Pressure 🌬️", f"{df_filtered['pressure'].mean():.2f} hPa")
col4.metric("Avg Wind Speed 🍃", f"{df_filtered['wind_speed'].mean():.2f} km/h")

st.markdown("---")

# ----------------------------
# CHARTS SECTION
# ----------------------------
st.subheader("📊 Weather Visual Analytics")
tab1, tab2, tab3 = st.tabs(["Temperature", "Air Quality", "Wind & Clouds"])

with tab1:
    fig = px.bar(df_filtered, x="city", y="temperature", color="temperature",
                 title="City-wise Temperature", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(df_filtered, x="humidity", y="temperature",
                      color="city", size="humidity",
                      title="Humidity vs Temperature", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    # --- Fixed AQI chart ---
    fig3 = px.line(df_filtered, x="city", y="aqi",
                   markers=True, title="AQI Levels by City",
                   labels={"aqi": "AQI", "city": "City"})
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(df_filtered, y="pm2_5", x="city",
                  title="PM2.5 Distribution by City")
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    fig5 = px.bar(df_filtered, x="city", y="wind_speed",
                  title="Wind Speed by City", color="wind_speed")
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.area(df_filtered, x="city", y="cloudcover",
                   title="Cloud Cover Comparison")
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ----------------------------
# AI / FUTURE PREDICTION MODEL
# ----------------------------
st.subheader("🤖 AI Powered Weather Prediction")
st.write("This model predicts temperature using weather & pollution features.")

features = ["humidity", "pressure", "wind_speed", "co", "no2", "o3", "pm10", "pm2_5"]

X = df[features]
y = df["temperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

colA, colB, colC = st.columns(3)
colA.metric("Model Accuracy (R2)", f"{r2_score(y_test, pred):.2f}")
colB.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, pred):.2f}")
colC.metric("Test Samples", len(y_test))

st.markdown("### 🎯 Try Predicting Temperature Yourself")

c1, c2, c3, c4 = st.columns(4)
humidity = c1.slider("Humidity", 0, 100, int(df["humidity"].mean()))
pressure = c2.slider("Pressure", 900, 1100, int(df["pressure"].mean()))
wind_speed = c3.slider("Wind Speed", 0, 50, int(df["wind_speed"].mean()))
co = c4.slider("CO Level", 0, int(df["co"].max()), int(df["co"].mean()))

c5, c6, c7, c8 = st.columns(4)
no2 = c5.slider("NO2", 0, int(df["no2"].max()), int(df["no2"].mean()))
o3 = c6.slider("Ozone (O3)", 0, int(df["o3"].max()), int(df["o3"].mean()))
pm10 = c7.slider("PM10", 0, int(df["pm10"].max()), int(df["pm10"].mean()))
pm25 = c8.slider("PM2.5", 0, int(df["pm2_5"].max()), int(df["pm2_5"].mean()))

input_data = np.array([[humidity, pressure, wind_speed, co, no2, o3, pm10, pm25]])
temperature_prediction = model.predict(input_data)[0]

st.success(f"🌡️ **Predicted Temperature:** {temperature_prediction:.2f} °C")

st.markdown("---")
st.info("📌 Tip: Deploy this easily to Streamlit Cloud or HuggingFace Spaces!")

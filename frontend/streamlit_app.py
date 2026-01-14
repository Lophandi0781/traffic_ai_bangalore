import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime, date, time

st.set_page_config(page_title="PathPioneer.blr", layout="wide")

API = st.sidebar.text_input(
    "Backend API URL",
    "https://traffic-ai-bangalore.onrender.com"
)


st.title("🚦 Bangalore Traffic AI Predictor")

# Bangalore sample pins
PINS = {
    "Silk Board Junction": (12.9177, 77.6230),
    "Hebbal Flyover": (13.0458, 77.5917),
    "KR Puram": (13.0075, 77.6950),
    "Marathahalli": (12.9592, 77.6974),
    "Electronic City": (12.8452, 77.6602),
    "M G Road": (12.9758, 77.6067),
}

left, right = st.columns([1, 1])

with left:
    st.subheader("Inputs")
    location_id = st.selectbox("Road / Junction", list(PINS.keys()))
    d = st.date_input("Date", date.today())
    t = st.time_input("Time", datetime.now().time().replace(second=0, microsecond=0))
    horizon_minutes = st.slider("Horizon (minutes)", 15, 180, 30, 15)

    is_rain = st.toggle("Rain (simulate)", value=False)
    is_event = st.toggle("Event (simulate)", value=False)

    if st.button("Predict 🚀"):
        ts = datetime.combine(d, t).isoformat()
        payload = {
            "location_id": location_id,
            "timestamp": ts,
            "horizon_minutes": int(horizon_minutes),
            "is_rain": 1 if is_rain else 0,
            "is_event": 1 if is_event else 0,
        }

        try:
            r = requests.post(f"{API}/predict", json=payload, timeout=10)

            if r.status_code != 200:
                st.error(r.text)
            else:
                out = r.json()
                st.success("Prediction successful ✅")
                st.metric("Predicted Speed (km/h)", out["predicted_speed_kmph"])
                st.metric("Congestion", out["congestion_label"])
        except Exception as e:
            st.error(f"API call failed: {e}")

with right:
    st.subheader("Map (pins)")
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11)
    for name, (lat, lon) in PINS.items():
        folium.Marker([lat, lon], popup=name, tooltip=name).add_to(m)

    st_folium(m, width=700, height=520)

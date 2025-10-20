import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lasio
from io import StringIO

st.set_page_config(page_title="Time-Depth Chart Generator", layout="centered")

st.title("Time–Depth Chart from Sonic Log")
st.markdown("Generate a **Time–Depth (TWT–Depth)** chart from Sonic (DT) log data using well header inputs.")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("📂 Upload LAS or CSV file", type=["las", "csv"])

if uploaded_file:
    filename = uploaded_file.name.lower()

    # Read LAS or CSV
    if filename.endswith(".las"):
        las = lasio.read(uploaded_file)
        df = las.df()
        df.reset_index(inplace=True)
        if 'DEPT' in df.columns:
            df.rename(columns={'DEPT': 'Depth'}, inplace=True)
        st.success("LAS file loaded successfully.")
    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.success("CSV file loaded successfully.")

    # Validate columns
    if not {'Depth', 'DT'}.issubset(df.columns):
        st.error("The file must contain 'Depth' and 'DT' columns.")
        st.stop()

    # ---- WELL PARAMETERS ----
    st.subheader("Well Header Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        log_start = st.number_input("Log Start Depth (m)", value=1517.0, step=1.0)
    with col2:
        kb = st.number_input("Kelly Bushing Elevation (m)", value=15.0, step=0.5)
    with col3:
        repl_vel = st.number_input("Replacement Velocity (m/s)", value=2632.0, step=10.0)

    gap_int = log_start - kb
    log_start_time = 2.0 * gap_int / repl_vel
    st.write(f"**Computed Log Start Time Offset (TWT):** {log_start_time:.3f} s")

    # ---- UNIT SELECTION ----
    unit = st.radio("Select Sonic Log Unit:", ('us/ft', 'us/m'))

    # ---- DATA CLEANUP ----
    df['DT'] = df['DT'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Depth_diff'] = df['Depth'].diff().fillna(0)

    # ---- COMPUTE INTERVAL TIMES ----
    if unit == 'us/ft':
        df['dt_interval'] = np.nan_to_num(df['DT']) * df['Depth_diff'] / (1e6 * 0.3048)
    elif unit == 'us/m':
        df['dt_interval'] = np.nan_to_num(df['DT']) * df['Depth_diff'] / 1e6 

    # ---- COMPUTE TIME ----
    t_cum = np.cumsum(df['dt_interval'])
    df['TWT'] = t_cum * 2 + log_start_time

    # ---- DISPLAY RESULTS ----
    st.subheader("Time–Depth Chart")
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(df['TWT'], df['Depth'], color='blue', linewidth=1.5)
    ax.invert_yaxis()
    ax.set_xlabel("Two-Way Time (s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Time–Depth Chart from Sonic Log")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # ---- DOWNLOAD RESULTS ----
    csv_output = df[['Depth', 'TWT']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Time–Depth CSV",
        data=csv_output,
        file_name="time_depth_chart.csv",
        mime="text/csv"
    )

    # ---- OPTIONAL VIEW DATA ----
    with st.expander("View Processed Data"):
        st.dataframe(df.head(20))
else:
    st.info("Upload a LAS or CSV file to begin.")

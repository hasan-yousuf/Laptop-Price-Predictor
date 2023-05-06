import numpy as np
import pandas as pd
import streamlit as st
import pickle


# Importing Dataset and Pipe

with open("dataset.pkl", "rb") as data:
    df = pd.read_pickle(data)

pipe = pickle.load(open("pipe(RFR).pkl", "rb"))

st.title("Laptop Price Calculator")

# Brand Name
brand = st.selectbox("Brand", df["Company"].unique())

# Laptop Type
type = st.selectbox("Type", df["Type"].unique())

# RAM Size
# st.select_slider("Select the RAM size", options=[8,12,16,32])
ram = st.selectbox("RAM", [2, 4, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input("Weight")

# TouchScreen
touchscreen = st.selectbox("TouchScreen", ["No", "Yes"])

# IPS Panel
ips_panel = st.selectbox("IPS Panel", ["No", "Yes"])

# Screen SIze
screen_size = st.number_input("Screen Size")

# Resolution
resolution = st.selectbox("Screen Resolution",
                          ['1980x1080', '1366x768', '1600x900', '3840x2160', '2880x1800', '2560x1600', '2560x1440',
                           '2304x1440'])

# CPU
cpu = st.selectbox("CPU", df["CPU"].unique())

# Hard Disk
hdd = st.selectbox("Hard Disk Size", np.unique(df["HDD"].to_numpy()))

# Solid State Drive Size
ssd = st.selectbox("SSD Size", np.unique(df["SSD"].to_numpy()))

# GPU Type
gpu = st.selectbox("GPU", df["GPU"].unique())

# Operating System
os = st.selectbox("Operating System", df["OS"].unique())

if st.button("Predict Price"):

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips_panel == 'Yes':
        ips_panel = 1
    else:
        ips_panel = 0

    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])

    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    st.title("Please select a valid screen size!")
    input = np.array([brand, type, ram, gpu, weight, touchscreen, ips_panel, ppi, cpu, os, ssd, hdd])
    input = input.reshape(1, 12)
    price = round(np.exp(pipe.predict(input)[0]), 2)
    price = "{:,}".format(price)
    st.title(f"Predicted price of this configuration is: ₹ {price}")


import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# ==========================================
# LOAD MODEL AND PREPROCESSOR
# ==========================================

model = load_model("porter_delivery_model.keras")
preprocessor = joblib.load("porter_preprocessor.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Porter Delivery Time Predictor",
    page_icon="🚚"
)


# ==========================================
# TITLE
# ==========================================

st.title("🚚 Porter Delivery Time Predictor")

st.write(
    "Enter the order details to predict the estimated delivery time."
)


# ==========================================
# USER INPUTS
# ==========================================

market_id = st.number_input(
    "Market ID",
    min_value=0,
    value=1
)

order_protocol = st.number_input(
    "Order Protocol",
    min_value=0,
    value=1
)

total_items = st.number_input(
    "Total Items",
    min_value=1,
    value=2
)

subtotal = st.number_input(
    "Subtotal",
    min_value=0.0,
    value=500.0
)

num_distinct_items = st.number_input(
    "Number of Distinct Items",
    min_value=1,
    value=2
)

min_item_price = st.number_input(
    "Minimum Item Price",
    min_value=0.0,
    value=50.0
)

max_item_price = st.number_input(
    "Maximum Item Price",
    min_value=0.0,
    value=200.0
)

total_onshift_partners = st.number_input(
    "Total Onshift Partners",
    min_value=0,
    value=10
)

total_busy_partners = st.number_input(
    "Total Busy Partners",
    min_value=0,
    value=5
)

total_outstanding_orders = st.number_input(
    "Total Outstanding Orders",
    min_value=0,
    value=5
)

hour = st.number_input(
    "Order Hour",
    min_value=0,
    max_value=23,
    value=12
)

day_of_week = st.number_input(
    "Day of Week (0 = Monday, 6 = Sunday)",
    min_value=0,
    max_value=6,
    value=2
)

# Automatically determine weekend
is_weekend = 1 if day_of_week >= 5 else 0

store_primary_category = st.text_input(
    "Store Primary Category",
    value="American"
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Delivery Time"):

    # Create dataframe with EXACT
    # same features used during training

    input_data = pd.DataFrame({
        "market_id": [market_id],
        "order_protocol": [order_protocol],
        "total_items": [total_items],
        "subtotal": [subtotal],
        "num_distinct_items": [num_distinct_items],
        "min_item_price": [min_item_price],
        "max_item_price": [max_item_price],
        "total_onshift_partners": [total_onshift_partners],
        "total_busy_partners": [total_busy_partners],
        "total_outstanding_orders": [total_outstanding_orders],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend],
        "store_primary_category": [
            store_primary_category
        ]
    })

    # Preprocess input
    input_processed = preprocessor.transform(
        input_data
    )

    # Predict
    prediction = model.predict(
        input_processed,
        verbose=0
    )[0][0]

    # Display result
    st.success(
        f"Estimated Delivery Time: {prediction:.2f} minutes"
    )
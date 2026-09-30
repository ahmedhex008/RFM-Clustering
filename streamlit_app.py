import os

import requests
import streamlit as st

API_URL = os.getenv("RFM_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RFM Customer Segmentation", page_icon="📊", layout="wide")
st.title("RFM Customer Segmentation")
st.write(
    "Enter a customer's Recency, Frequency, and Monetary values to predict "
    "their KMeans segment."
)

with st.form("rfm_prediction"):
    recency_col, frequency_col, monetary_col = st.columns(3)
    recency = recency_col.number_input(
        "Recency (days since last purchase)", min_value=0.01, value=30.0, step=1.0
    )
    frequency = frequency_col.number_input(
        "Frequency (number of purchases)", min_value=0.01, value=5.0, step=1.0
    )
    monetary = monetary_col.number_input(
        "Monetary (customer value)", min_value=0.01, value=100.0, step=10.0
    )
    submitted = st.form_submit_button("Predict customer segment", type="primary")

if submitted:
    try:
        response = requests.post(
            f"{API_URL.rstrip('/')}/predict",
            json={
                "recency": recency,
                "frequency": frequency,
                "monetary": monetary,
            },
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(
            f"Could not reach the prediction API at {API_URL}. "
            "Start the API first. Details: "
            f"{exc}"
        )
    else:
        result = response.json()
        st.success(
            f"Predicted segment: {result['segment_name']} "
            f"(Cluster {result['cluster_id']})"
        )
        st.write(result["description"])
        st.subheader("Customer groups")
        st.caption(
            "Segment names describe each cluster's median RFM values compared "
            "with the overall customer median."
        )
        st.dataframe(
            [
                {
                    "Cluster": segment["cluster_id"],
                    "Segment": segment["segment_name"],
                    "Customers in training data": segment["customer_count"],
                    "Median recency": segment["median_rfm"]["recency"],
                    "Median frequency": segment["median_rfm"]["frequency"],
                    "Median monetary": segment["median_rfm"]["monetary"],
                    "Profile": segment["description"],
                }
                for segment in result["segments"]
            ],
            use_container_width=True,
            hide_index=True,
        )

st.caption("KMeans model · Recency, Frequency, Monetary")

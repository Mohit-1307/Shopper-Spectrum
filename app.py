"""
Shopper Spectrum - Streamlit Application
Customer Segmentation and Product Recommendations in E-Commerce

This app has two modules, exactly as specified in the project brief:

    1. Product Recommendation - takes a product name, returns 5 similar products
    2. Customer Segmentation  - takes Recency, Frequency, Monetary, returns the predicted segment

"""

import streamlit as st

import pandas as pd

import numpy as np

import joblib


# Page Configuration
st.set_page_config(

    page_title = "Shopper Spectrum",

    page_icon = "🛒",

    layout = "wide"

)


# Load Saved Models
@st.cache_resource
def load_models():
    """Load all saved model artifacts from the models/ directory."""

    kmeans = joblib.load('models/kmeans_model.pkl')

    scaler = joblib.load('models/rfm_scaler.pkl')

    cluster_label_map = joblib.load('models/cluster_label_map.pkl')

    cosine_sim_df = pd.read_pickle('models/cosine_sim_df.pkl')

    return kmeans, scaler, cluster_label_map, cosine_sim_df

kmeans, scaler, cluster_label_map, cosine_sim_df = load_models()


# Sidebar Navigation — matches the Home / Clustering / Recommendation
# layout shown in the project brief mockup
st.sidebar.title("🛒 Shopper Spectrum")

page = st.sidebar.radio(
    
    "Navigation",
    
    options = ["Home", "Clustering", "Recommendation"]
    
)


# Home Page
if page == "Home":

    st.title("Shopper Spectrum")

    st.subheader("Customer Segmentation and Product Recommendations in E-Commerce")

    st.markdown("""
    
    This application uses transaction data from an online retail business to:

    - **Segment customers** into High-Value, Regular, Occasional, and At-Risk groups based on
        their Recency, Frequency, and Monetary (RFM) purchase behaviour.

    - **Recommend products** using item-based collaborative filtering, so that entering a product
        name returns 5 similar products based on what customers tend to buy together.

    Use the sidebar to open the **Clustering** module to predict a customer segment, or the
    **Recommendation** module to get product suggestions.
    
    """)

    st.markdown("---")

    st.markdown("**Customer Segments Overview**")

    segment_data = {

        "Segment"   : ["High-Value", "Regular", "Occasional", "At-Risk"],

        "Recency"   : ["Low (very recent)", "Medium", "Medium-High", "High (long inactive)"],

        "Frequency" : ["High", "Medium", "Low", "Low"],

        "Monetary"  : ["High", "Medium", "Low", "Low"]

    }
    
    st.table(pd.DataFrame(segment_data))


# Clustering Page — Customer Segmentation Module
elif page == "Clustering":

    st.title("Customer Segmentation")

    st.markdown("Enter a customer's purchase behaviour to predict which segment they belong to.")

    st.markdown("---")

    # 3 number inputs as specified: Recency, Frequency, Monetary
    recency = st.number_input(

        "Recency (days since last purchase)",

        min_value = 0,

        max_value = 1000,

        value = 30,

        step = 1

    )

    frequency = st.number_input(

        "Frequency (number of purchases)",

        min_value = 1,

        max_value = 1000,

        value = 5,

        step = 1

    )

    monetary = st.number_input(
        
        "Monetary (total spend)",

        min_value = 0.0,

        max_value = 500000.0,

        value = 250.0,

        step = 10.0

    )

    if st.button("Predict Cluster"):

        # Arrange input as a DataFrame with the same column names the scaler was fitted on
        # Using named columns avoids the sklearn "missing feature names" warning
        input_rfm = pd.DataFrame([[recency, frequency, monetary]], columns = ['Recency', 'Frequency', 'Monetary'])

        # Scale the input using the same StandardScaler fitted during training
        input_scaled = scaler.transform(input_rfm)

        # Predict the cluster number using the saved KMeans model
        cluster_id = kmeans.predict(input_scaled)[0]

        # Map the cluster number to its business segment label
        segment_label = cluster_label_map.get(cluster_id, f"Cluster {cluster_id}")

        st.markdown("---")

        st.write(cluster_id)

        st.markdown(f"**This customer belongs to:** {segment_label}")


# Recommendation Page — Product Recommendation Module
elif page == "Recommendation":

    st.title("Product Recommender")

    product_input = st.text_input("Enter Product Name")

    if st.button("Recommend"):

        # Convert to uppercase to match the dataset's product description format
        product_upper = product_input.upper().strip()

        # Find products in the similarity matrix whose name contains the search term
        matched = [p for p in cosine_sim_df.index if product_upper in p]

        if not matched:

            st.error(f"No product found matching: '{product_input}'. Please check the spelling and try again.")

        else:

            selected_product = matched[0]

            # Get similarity scores for the selected product, sorted highest first
            sim_scores = cosine_sim_df[selected_product].sort_values(ascending = False)

            # Exclude the input product itself and take the top 5 recommendations
            recommendations = sim_scores[sim_scores.index != selected_product].head(5).index.tolist()

            st.markdown("**Recommended Products:**")

            for rec in recommendations:

                st.write(rec)

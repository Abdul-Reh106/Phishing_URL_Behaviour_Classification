import joblib
import pandas as pd
import streamlit as st
from features import extract_features
model = joblib.load("models/decision_tree_model.joblib")
st.title("Phishing URL Behaviour Classification")
st.write("Enter a full URL beginning with http:// or https://.")

url = st.text_input("URL", placeholder="https://example.com")
if st.button("Check URL"):
    cleaned_url = url.strip()

    if not cleaned_url:
        st.warning("Please enter a URL.")
    elif not cleaned_url.lower().startswith(("http://", "https://")):
        st.warning("Start the URL with http:// or https://.")
    else:
        try:
            features = extract_features(cleaned_url)
            st.write("Features extracted from the URL:")
            st.json(features)
            model_input = pd.DataFrame([features])
            prediction = 0 if model.predict_proba(model_input)[0, 0] >= 0.30 else 1

            if prediction == 0:
                st.warning("Prediction: Potential phishing")
            else:
                st.info("Prediction: Likely legitimate")

            st.caption(
                "Based on URL text only. The model can flag legitimate URLs "
                "or miss phishing URLs; it does not verify the website."
            )
        except ValueError:
            st.error("Could not extract a hostname. Please check the URL.")
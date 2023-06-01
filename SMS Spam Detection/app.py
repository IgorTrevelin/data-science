import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pickle
from PIL import Image

from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class SMSTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sentence_transformer = SentenceTransformer("all-mpnet-base-v2")

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.sentence_transformer.encode(X.ravel(), show_progress_bar=False)

st.set_page_config("SMS Spam Detector - by Igor Trevelin")

try:
    with open("spam_detector.pkl", "rb") as f:
        spam_detector = pickle.load(f)
except Exception as ex:
    st.error("Backend error. Please contact the developer through the address igor.trevelin.xavier@gmail.com")
    spam_detector = None

if spam_detector:
    col1, col2 = st.columns(2)

    probas = None
    detecting = False

    with col1:
        sms = st.text_area(label="Enter the SMS message text:")
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Check Spam!"):
                if sms:
                    probas = None
                    detecting = True
                    probas = spam_detector.predict_proba(np.array([[sms]]))
                else:
                    st.info("Please, fill the textarea before checking the message.")

        with btn2:
            if st.button("Reset"):
                probas = None
                detecting=False

    with col2:
        if probas is not None:
            df = pd.DataFrame(data=probas, columns=["Ham", "Spam"]).transpose().reset_index()
            df.columns = ["Classification", "Probability"]
            pie = px.pie(data_frame=df, values="Probability", names="Classification")
            pie.update_layout(title="Ham vs Spam Probabilities", title_font=dict(color="white", size=24), title_x=0.5)
            st.plotly_chart(pie)
        elif detecting:
            st.spinner("Wait for it...")
        else:
            st.header("SMS Spam Detector")
            st.text("Please follow the steps:\n1. Fill the textarea\n2. Click the check button")
            st.text("Developed by Igor Trevelin")
            

import streamlit as st
import tensorflow as tf

# LOAD MODEL

model = tf.keras.models.load_model("models/rnn_model.keras")

# UI

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Movie Review Sentiment Analysis")

st.write("RNN Based Sentiment Analysis")

review = st.text_area("Enter Movie Review")

# PREDICT BUTTON

if st.button("Predict"):
    st.info("Full NLP tokenization pipeline not added.")

    st.success("This demonstrates RNN deployment.")

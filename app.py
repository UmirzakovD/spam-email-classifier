"""Streamlit web UI for the Spam Email Classifier.

Run with:
    streamlit run app.py

Trains the model on first launch if no saved model is found (same
auto-download / auto-train behavior as `python main.py`), then lets you
classify messages from the browser instead of the terminal.
"""
import json

import streamlit as st

from src.data_loader import DatasetDownloadError
from src.predict import predict as run_predict
from src.train import METRICS_PATH, MODEL_PATH, VECTORIZER_PATH, train

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")

EXAMPLES = {
    "Spam example": "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now!",
    "Ham example": "Hey, are we still meeting for lunch tomorrow?",
}


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model + vectorizer, training them first if missing."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        train()
    import joblib

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


def main():
    st.title("📧 Spam Email Classifier")
    st.caption(
        "TF-IDF + Naive Bayes / Logistic Regression, trained on the UCI SMS Spam Collection dataset. "
        "[View on GitHub](https://github.com/UmirzakovD/spam-email-classifier)"
    )

    with st.spinner("Loading model (first run trains it, this can take a minute)..."):
        try:
            model, vectorizer = load_model()
        except DatasetDownloadError as exc:
            st.error(f"Could not download the training dataset: {exc}")
            st.stop()

    metrics = load_metrics()
    if metrics:
        best = metrics["best_model"]
        best_metrics = metrics["results"][best]
        with st.sidebar:
            st.subheader("Model info")
            st.write(f"**Active model:** {best.replace('_', ' ').title()}")
            st.metric("Accuracy", f"{best_metrics['accuracy']:.2%}")
            st.metric("F1-score", f"{best_metrics['f1_score']:.2%}")
            st.metric("Precision", f"{best_metrics['precision']:.2%}")
            st.metric("Recall", f"{best_metrics['recall']:.2%}")

    if "message_text" not in st.session_state:
        st.session_state.message_text = ""

    st.subheader("Try it")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state.message_text = text

    message = st.text_area(
        "Message text",
        key="message_text",
        height=140,
        placeholder="Paste an email or SMS message here...",
    )

    if st.button("Classify", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Type or paste a message first.")
        else:
            result = run_predict(message, model, vectorizer)
            label = result["label"]
            prob = result["spam_probability"]

            if label == "spam":
                st.error(f"🚫 **SPAM** (spam probability: {prob:.1%})")
            else:
                st.success(f"✅ **HAM** — legitimate message (spam probability: {prob:.1%})")

            if prob is not None:
                st.progress(prob, text="Spam probability")


if __name__ == "__main__":
    main()

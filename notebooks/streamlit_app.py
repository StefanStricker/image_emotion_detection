"""Customer-facing GUI for the emotion classifier.

A thin Streamlit frontend: it has no PyTorch/model logic of its own, it just
uploads an image to the FastAPI backend (serve.py) and renders the response.
Run the backend first, then this app — see README.md "Usage".
"""

import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

# Emoji per class, purely cosmetic — makes the headline result readable at a
# glance for a non-technical user.
EMOJI = {
    "angry": "😠", "disgusted": "🤢", "fearful": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprised": "😲",
}

# Single-hue "blue" family (see project's dataviz palette): the predicted
# class is the full-strength hue, the rest are a muted step of the same
# ramp — this is a magnitude/emphasis distinction, not seven competing
# category colors, since all seven bars are one series (class probability).
BAR_HIGHLIGHT = "#2a78d6"
BAR_MUTED = "#b7d3f6"

st.set_page_config(page_title="Emotion Detector", page_icon="🙂", layout="centered")

st.title("🙂 Emotion Detector")
st.write(
    "Upload a photo of a face and the model will tell you which of 7 emotions "
    "it's showing: angry, disgusted, fearful, happy, neutral, sad, surprised."
)

with st.sidebar:
    st.header("Options")
    use_tta = st.checkbox(
        "More accurate (slower)", value=False,
        help="Runs the image through the model twice (once mirrored) and averages "
             "the result. Slightly slower, slightly more accurate.",
    )
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", width=300)

    with st.spinner("Analyzing..."):
        try:
            response = requests.post(
                f"{backend_url}/predict",
                params={"tta": use_tta},
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "Can't reach the backend. Make sure it's running:\n\n"
                "`uvicorn notebooks.serve:app --app-dir . --reload`"
            )
            st.stop()
        except requests.exceptions.RequestException as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    emotion = result["emotion"]
    confidence = result["confidence"]
    scores = result["scores"]

    st.subheader(f"{EMOJI.get(emotion, '')} {emotion.capitalize()} ({confidence:.0%} confident)")

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k.capitalize() for k, _ in ordered]
    values = [v for _, v in ordered]
    colors = [BAR_HIGHLIGHT if k == emotion else BAR_MUTED for k, _ in ordered]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0%}" for v in values],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Confidence"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=40, t=10, b=40),
        height=320,
    )
    st.plotly_chart(fig, width="stretch")

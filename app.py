import json
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VocalSense AI",
    page_icon="🎙️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "model/emotion_recognition_model.keras"
CONFIG_PATH = "model/preprocessing_config.json"


# ============================================================
# LOAD MODEL AND CONFIG
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
config = load_config()

SAMPLE_RATE = config["sample_rate"]
N_MFCC = config["n_mfcc"]
MAX_PAD_LEN = config["max_pad_len"]
CLASS_NAMES = config["classes"]


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background-color: #0b0e14;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* HERO */

.hero-badge {
    text-align: center;
    color: #a78bfa;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 15px;
}

.hero-title {
    text-align: center;
    font-size: 55px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.hero-title span {
    color: #8b5cf6;
}

.hero-subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 18px;
    margin-bottom: 40px;
}


.upload-title {
    color: white;
    font-size: 23px;
    font-weight: 700;
    margin-bottom: 8px;
}

.upload-description {
    color: #9ca3af;
    font-size: 15px;
    line-height: 1.6;
}

/* RESULT CARDS */

.result-card {
    background: #151923;
    border: 1px solid #292e3a;
    border-radius: 20px;
    padding: 25px;
    min-height: 150px;
}

.result-label {
    color: #9ca3af;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-value {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin-top: 10px;
}

.result-sub {
    color: #a78bfa;
    font-size: 15px;
    margin-top: 5px;
}

/* MODEL CARDS */

.info-card {
    background: #151923;
    border: 1px solid #292e3a;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
}

.info-number {
    color: white;
    font-size: 28px;
    font-weight: 800;
}

.info-label {
    color: #8f98a8;
    font-size: 13px;
    margin-top: 5px;
}

/* BUTTON */

.stButton > button {
    height: 50px;
    border-radius: 12px;
    font-weight: 700;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 45px;
    padding-top: 25px;
    border-top: 1px solid #20242e;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero-badge">DEEP LEARNING · SPEECH INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Vocal<span>Sense AI</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">Understand emotions hidden in human speech using deep learning and acoustic features.</div>',
    unsafe_allow_html=True
)


# ============================================================
# AUDIO PROCESSING
# ============================================================

def extract_mfcc(audio_file):

    audio_file.seek(0)

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    if mfcc.shape[1] < MAX_PAD_LEN:

        pad_width = MAX_PAD_LEN - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :MAX_PAD_LEN]

    # Same normalization used during training
    mean = mfcc.mean()
    std = mfcc.std()

    mfcc = (mfcc - mean) / (std + 1e-8)

    # Add channel dimension
    mfcc = mfcc[..., np.newaxis]

    # Add batch dimension
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="upload-title">Analyze a Speech Recording</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-description">Upload a clear WAV voice recording and let VocalSense AI identify the most likely emotional state.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload WAV audio",
    type=["wav"]
)

# ============================================================
# ANALYZE
# ============================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    if st.button(
        "Analyze Speech",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing speech..."):

            try:

                features = extract_mfcc(uploaded_file)

                predictions = model.predict(
                    features,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_emotion = CLASS_NAMES[
                    predicted_index
                ]

                confidence = float(
                    predictions[predicted_index] * 100
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

                st.stop()


        # ====================================================
        # RESULT
        # ====================================================

        st.subheader("Analysis Result")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Detected Emotion</div>
            <div class="result-value">{predicted_emotion.title()}</div>
            <div class="result-sub">Most likely emotional state</div>
        </div>
        """.replace("\n", ""),
        unsafe_allow_html=True
    )

        with col2:

            st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Model Confidence</div>
            <div class="result-value">{confidence:.2f}%</div>
            <div class="result-sub">Prediction confidence</div>
        </div>
        """.replace("\n", ""),
        unsafe_allow_html=True
    )

        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        st.subheader("Emotion Probability")

        probability_data = {}

        for emotion, probability in zip(
            CLASS_NAMES,
            predictions
        ):

            probability_data[
                emotion.title()
            ] = float(probability * 100)

        st.bar_chart(
            probability_data,
            height=350
        )


        # ====================================================
        # DETAILED PROBABILITIES
        # ====================================================

        st.subheader("Detailed Analysis")

        for emotion, probability in zip(
            CLASS_NAMES,
            predictions
        ):

            percentage = float(
                probability * 100
            )

            st.progress(
                min(percentage / 100, 1.0),
                text=f"{emotion.title()} — {percentage:.2f}%"
            )


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.subheader("Model Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        '<div class="info-card"><div class="info-number">CNN</div><div class="info-label">Deep Learning Model</div></div>',
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        '<div class="info-card"><div class="info-number">40</div><div class="info-label">MFCC Features</div></div>',
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        '<div class="info-card"><div class="info-number">8</div><div class="info-label">Emotion Classes</div></div>',
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        '<div class="info-card"><div class="info-number">45.83%</div><div class="info-label">Unseen Speaker Test Accuracy</div></div>',
        unsafe_allow_html=True
    )


# ============================================================
# SUPPORTED EMOTIONS
# ============================================================

st.subheader("Supported Emotions")

st.info(
    "  •  ".join(
        emotion.title()
        for emotion in CLASS_NAMES
    )
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">VocalSense AI provides machine-learning based estimates from speech recordings and should not be treated as a clinical or psychological assessment.<br><br>Built with Python · TensorFlow · Librosa · Streamlit</div>',
    unsafe_allow_html=True
)
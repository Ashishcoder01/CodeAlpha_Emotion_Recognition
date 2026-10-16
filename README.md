# VocalSense AI — Speech Emotion Recognition

**Live Demo:** https://vocalsense-ai.streamlit.app/  
**Source Code:** https://github.com/Ashishcoder01/CodeAlpha_Emotion_Recognition

VocalSense AI is a machine-learning application that analyzes speech recordings and predicts one of eight emotion categories. It uses MFCC (Mel-Frequency Cepstral Coefficient) audio features and a trained Convolutional Neural Network (CNN), with an interactive interface built using Streamlit.

> **Note:** Emotion predictions are model estimates, not definitive interpretations of a person's feelings or psychological state.

## Live Demo

Try the deployed application: **[VocalSense AI](https://vocalsense-ai.streamlit.app/)**

Upload a WAV speech recording, select **Analyze Speech**, and review the predicted emotion, confidence score, and probability distribution.

## Features

- Upload and play WAV audio recordings.
- Predict one of eight supported emotion categories.
- Display the highest-probability prediction and its model confidence.
- Visualize probability scores across all emotion categories.
- Show detailed per-emotion scores.
- Use a saved CNN model and matching preprocessing configuration.
- Run locally or access the hosted Streamlit application.

## Supported Emotions

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | TensorFlow / Keras |
| Audio feature extraction | Librosa |
| Audio support | SoundFile |
| Numerical processing | NumPy |
| Web interface | Streamlit |
| Dataset | RAVDESS |

## Model and Evaluation

The model extracts **40 MFCC coefficients** and pads or truncates the features to **174 frames**. Features are normalized before inference.

To better measure performance on speakers the model did not train on, the RAVDESS recordings were split by actor:

- **Training:** Actors 1–16
- **Validation:** Actors 17–20
- **Testing:** Actors 21–24

| Evaluation metric | Result |
|---|---:|
| Validation accuracy | 54.17% |
| Unseen-speaker test accuracy | 45.83% |

The **45.83% test accuracy** is the final result on held-out speakers. Performance differs across emotion classes, and some categories remain challenging. The confidence score represents the model's output probability; it does not guarantee that the prediction is correct.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ashishcoder01/CodeAlpha_Emotion_Recognition.git
cd CodeAlpha_Emotion_Recognition
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the application

```powershell
python -m streamlit run app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`.

## How to Use

1. Open the [live demo](https://vocalsense-ai.streamlit.app/) or run the app locally.
2. Upload a WAV speech recording.
3. Play the recording if needed.
4. Select **Analyze Speech**.
5. Review the predicted emotion, confidence score, and emotion probability chart.

## Project Structure

```text
CodeAlpha_Emotion_Recognition/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   ├── emotion_recognition_model.keras
│   ├── label_encoder.pkl
│   └── preprocessing_config.json
└── utils/
    ├── __init__.py
    └── audio_processor.py
```

## Limitations

- Generalization to unseen speakers is limited.
- Predictions may be affected by recording quality, background noise, accent, and speaking style.
- Some emotion classes are harder for the current model to distinguish.
- The model is intended for educational and experimental use, not clinical, psychological, hiring, or other high-stakes decisions.

## Future Improvements

- Train and evaluate on additional speech-emotion datasets.
- Improve recognition for underperforming emotion categories.
- Test robustness across accents, microphones, and noisy environments.
- Compare CNN models with recurrent and transformer-based architectures.
- Add support for additional audio formats and more extensive evaluation.

## Dataset and Acknowledgements

This project was developed as part of the CodeAlpha Machine Learning Internship tasks.

The model was trained using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. Please review the dataset's official terms and attribution requirements before redistributing any dataset recordings.

## Links

- **Live application:** https://vocalsense-ai.streamlit.app/
- **GitHub repository:** https://github.com/Ashishcoder01/CodeAlpha_Emotion_Recognition

---

**Project:** VocalSense AI  
**Category:** Machine Learning / Speech Emotion Recognition  
**Interface:** Streamlit

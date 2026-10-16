# VocalSense AI — Emotion Recognition from Speech

VocalSense AI is a deep learning application that predicts emotional categories from speech recordings using Mel-Frequency Cepstral Coefficients (MFCCs) and a Convolutional Neural Network (CNN). The application provides an interactive interface built with Streamlit.

## Features

- Upload and play WAV audio recordings.
- Predict one of eight emotion categories.
- Display the model's prediction confidence.
- Visualize probabilities across all emotion categories.
- View detailed emotion scores.
- Use a trained CNN model with saved preprocessing configuration.
- Run locally using Python and Streamlit.

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

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Audio Processing:** Librosa, SoundFile
- **Numerical Computing:** NumPy
- **Interface:** Streamlit
- **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

## Model and Evaluation

The model uses 40 MFCC features with a fixed input length of 174 frames. Audio features are normalized before being passed to the trained CNN.

The dataset was split by actor to evaluate generalization to speakers not used during training.

| Evaluation | Result |
|---|---:|
| Validation accuracy | 54.17% |
| Unseen-speaker test accuracy | 45.83% |

The test accuracy was measured on held-out actors. Performance varies by emotion, so predictions should be treated as estimates rather than definitive interpretations of a speaker's feelings.

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
    └── audio_processor.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_Emotion_Recognition.git
cd CodeAlpha_Emotion_Recognition
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Usage

1. Start the Streamlit application.
2. Upload a WAV speech recording.
3. Play the recording if needed.
4. Select **Analyze Speech**.
5. Review the predicted emotion, confidence, and probability distribution.

## Limitations

- The model's performance on unseen speakers is limited.
- Results can vary with recording quality, speaking style, and background noise.
- Some emotions are more difficult for the current model to distinguish.
- Prediction confidence is not a guarantee that the predicted emotion is correct.
- The application is intended for educational and experimental use, not clinical or psychological assessment.

## Future Improvements

- Expand training with additional speech-emotion datasets.
- Improve performance on under-recognized emotion categories.
- Evaluate noise robustness and different recording conditions.
- Compare CNN, recurrent, and transformer-based architectures.
- Add more audio formats and deployment options.

## Acknowledgements

This project was developed as part of the CodeAlpha Machine Learning Internship tasks.

The RAVDESS dataset is used for speech-emotion recognition research and experimentation. Please consult the dataset's official terms and attribution requirements before redistributing dataset recordings.

---

**Project:** VocalSense AI  
**Category:** Machine Learning / Speech Emotion Recognition  
**Interface:** Streamlit
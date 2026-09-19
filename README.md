# Speech Emotion Recognition using a Hybrid CNN-LSTM Architecture

## Objective
Recognize human emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from speech audio files using deep learning.

## Dataset
- **Name:** RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song
- **Source:** Downloaded automatically via Zenodo (official source)
- **Size:** 1,440 audio files — 24 actors (12 male, 12 female), 8 emotions
- **Format:** .wav files, 24kHz stereo

## Emotion Classes
| Code | Emotion | Code | Emotion |
|------|---------|------|---------|
| 01 | Neutral | 05 | Angry |
| 02 | Calm | 06 | Fearful |
| 03 | Happy | 07 | Disgust |
| 04 | Sad | 08 | Surprised |

## Tools & Libraries
- Python 3.10
- librosa — audio processing & MFCC extraction
- TensorFlow / Keras — deep learning model
- NumPy, pandas — data handling
- matplotlib, seaborn — visualization
- scikit-learn — preprocessing & evaluation

## Approach

### Feature Extraction
- Loaded each `.wav` file using `librosa`
- Extracted **40 MFCC (Mel-Frequency Cepstral Coefficients)** per audio clip
- Audio clipped to **3 seconds**, padded/truncated to fixed length (174 frames)
- Final feature shape per file: **(174 timesteps × 40 MFCC coefficients)**

### Model Architecture — CNN + LSTM Hybrid
```
Input (174, 40)
    ↓
Conv1D (64 filters) → BatchNorm → MaxPool → Dropout
    ↓
Conv1D (128 filters) → BatchNorm → MaxPool → Dropout
    ↓
Conv1D (256 filters) → BatchNorm → MaxPool → Dropout
    ↓
LSTM (128 units)
    ↓
Dense (128) → BatchNorm → Dropout
    ↓
Dense (64) → Dropout
    ↓
Dense (8, softmax) → Emotion Prediction
```

### Training Strategy
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Cross-Entropy
- EarlyStopping (patience=10)
- ReduceLROnPlateau (patience=5)
- Train / Val / Test split: 70% / 15% / 15%

## What's Inside the Notebook
| Step | Description |
|------|-------------|
| 1 | Install libraries |
| 2 | Import libraries |
| 3 | Download RAVDESS dataset automatically |
| 4 | Understand file naming convention |
| 5 | EDA — emotion distribution, waveform, spectrogram, MFCC visualization |
| 6 | MFCC feature extraction for all files |
| 7 | Label encoding & train/val/test split |
| 8 | Build CNN + LSTM model |
| 9 | Train model with callbacks |
| 10 | Plot training history (accuracy & loss) |
| 11 | Evaluate — accuracy, classification report, confusion matrix, F1 per class |
| 12 | Save model (.h5) and label encoder (.pkl) |
| 13 | Predict emotion from a new audio file |
| 14 | Key insights & findings |

## Key Results & Findings
- CNN layers extract local temporal patterns from MFCCs
- LSTM captures long-range sequential dependencies in speech
- **Angry** and **happy** emotions tend to achieve higher F1-scores due to distinct acoustic features
- **Neutral** and **calm** are sometimes confused due to acoustic similarity
- Model saved as `emotion_recognition_model.h5` for reuse and deployment

## How to Run
```bash
pip install librosa tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook Task2_Emotion_Recognition.ipynb
```
Run all cells top to bottom. The dataset downloads automatically in Step 3 (~215 MB).

## Repository Structure
```
CodeAlpha_EmotionRecognition/
├── Task2_Emotion_Recognition.ipynb   ← Main notebook
├── README.md                          ← This file
├── emotion_recognition_model.h5       ← Saved model (generated after training)
├── label_encoder.pkl                  ← Saved label encoder (generated after training)
├── emotion_distribution.png           ← EDA plot
├── waveform_spectrogram.png           ← Waveform & spectrogram
├── mfcc_visualization.png             ← MFCC heatmap
├── training_history.png               ← Accuracy & loss curves
├── confusion_matrix.png               ← Confusion matrix
└── f1_per_emotion.png                 ← F1-score per emotion
```

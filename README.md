# SpectroGenre-machine-learning-pipeline-for-music-genre-recognition

# 🎵 SpectroGenre

### Machine Learning Pipeline for Music Genre Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Audio Processing](https://img.shields.io/badge/Audio-Librosa-purple)

**SpectroGenre** is a machine learning pipeline that classifies music genres from raw audio signals using audio signal processing and multiple machine learning algorithms.

The system extracts meaningful acoustic features such as **MFCC, spectral features, chroma features, and energy**, then evaluates multiple machine learning models to determine the best classifier for music genre recognition.

---

# 🎧 Project Overview

Music genre classification is a core task in **Music Information Retrieval (MIR)**.

This project converts **raw audio signals into numerical features** using signal processing techniques and trains multiple machine learning models to recognize musical patterns across genres.

The pipeline includes:

* Audio exploration and visualization
* Feature extraction using Librosa
* Feature dataset creation
* Machine learning model comparison
* Best model selection
* Model evaluation and visualization

---

# 📊 Machine Learning Pipeline

```
Raw Audio (.wav)
      │
      ▼
Audio Signal Processing
(librosa)
      │
      ▼
Feature Extraction
MFCC + Spectral + Chroma
      │
      ▼
Feature Dataset
(audio_features_improved.csv)
      │
      ▼
Train / Validation / Test Split
70% / 15% / 15%
      │
      ▼
Feature Scaling
(MinMaxScaler)
      │
      ▼
Model Training
Multiple ML Algorithms
      │
      ▼
Best Model Selection
      │
      ▼
Evaluation
Confusion Matrix
Classification Report
Feature Importance
```

---

# 🎼 Dataset

This project uses the **GTZAN Music Genre Dataset**.

Dataset source:

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Dataset characteristics:

| Attribute        | Value      |
| ---------------- | ---------- |
| Genres           | 10         |
| Tracks per Genre | 100        |
| Track Duration   | 30 seconds |
| Format           | WAV        |
| Total Tracks     | 1000       |

Genres included:

* Blues
* Classical
* Country
* Disco
* HipHop
* Jazz
* Metal
* Pop
* Reggae
* Rock

---

# 🎚 Audio Signal Visualization

The project analyzes audio signals through several visualizations.

### Sound Wave

Shows amplitude variation over time.

```
Audio Signal → Waveform Visualization
```

### Spectrogram

Displays frequency intensity across time.

```
Time vs Frequency vs Amplitude
```

### Mel Spectrogram

Represents frequency based on human auditory perception.

```
Mel Frequency Scale Visualization
```

These visualizations help interpret **acoustic patterns inside music signals**.

---

# 🎧 Audio Feature Extraction

Features are extracted using **Librosa**.

### Spectral Features

* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff

### Temporal Features

* Zero Crossing Rate
* Root Mean Square Energy (RMSE)

### Harmonic Features

* Chroma STFT

### Cepstral Features

* 20 MFCC coefficients

Each feature stores:

* Mean
* Variance

Total features generated per audio track include **spectral + cepstral statistics**.

---

# 🤖 Machine Learning Models Evaluated

Multiple machine learning models are trained and compared.

| Model                  | Library      |
| ---------------------- | ------------ |
| Naive Bayes            | scikit-learn |
| SGD Classifier         | scikit-learn |
| KNN                    | scikit-learn |
| Decision Tree          | scikit-learn |
| Random Forest          | scikit-learn |
| Support Vector Machine | scikit-learn |
| Logistic Regression    | scikit-learn |
| Neural Network (MLP)   | scikit-learn |
| XGBoost                | XGBoost      |
| XGBoost Random Forest  | XGBoost      |

The best model is automatically selected based on **highest test accuracy**.

---

# 📈 Model Evaluation

The best model is evaluated using several metrics.

### Confusion Matrix

Shows how well the model distinguishes each genre.

```
True Genre vs Predicted Genre
```

### Classification Report

Includes:

* Precision
* Recall
* F1 Score

### Feature Importance

Tree-based models such as Random Forest and XGBoost reveal the most influential audio features.

Example:

```
MFCC coefficients
Spectral centroid
Spectral bandwidth
```

These features represent key acoustic signatures used by the classifier.

---

# 🧠 Data Splitting Strategy

The dataset is split using **stratified sampling**.

| Dataset    | Percentage |
| ---------- | ---------- |
| Training   | 70%        |
| Validation | 15%        |
| Testing    | 15%        |

Feature scaling is applied using:

```
MinMaxScaler
```

Scaling is fitted only on the training data to **avoid data leakage**.

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/yourusername/spectrogenre.git
cd spectrogenre
```

Install dependencies:

```
pip install pandas numpy opendatasets seaborn matplotlib scikit-learn librosa xgboost tqdm ipython
```

---

# ▶️ Running the Project

Run the pipeline:

```
python spectrogenre_pipeline.py
```

The script will:

1. Download the dataset
2. Extract audio features
3. Train multiple machine learning models
4. Select the best model
5. Generate evaluation visualizations

---

# 📂 Project Structure

```
SpectroGenre
│
├── spectrogenre_pipeline.py
├── audio_features_improved.csv
│
├── notebooks
│   └── exploration.ipynb
│
├── README.md
└── requirements.txt
```

---

# 🚀 Future Improvements

Potential improvements for this project:

* Deep Learning models (CNN on spectrograms)
* Real-time genre prediction
* Web interface with Streamlit
* Model deployment
* Larger music datasets

---

# 👨‍💻 Author
Puteri Amelia Azli

Machine Learning project focused on **audio signal processing and music genre classification**.

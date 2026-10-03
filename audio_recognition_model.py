"""
Dual-Model Audio Intelligence System for Forest Acoustic Surveillance
========================================================================
Author: Melvin John

This repository contains a dual-model audio processing pipeline:
1. VGG16 Transfer Learning (CNN) — 3-class Animal Sound Classification (Lion, Donkey, Monkey)
2. Kernel SVM (RBF) — Binary Gunshot Detection (Gunshot vs. Non-Gunshot) using UrbanSound8K

Feature Extraction & Engineering:
- CNN Pipeline: Log-Mel Spectrograms (128 mel bands, shape: 128x174x3)
- SVM Pipeline: 40 MFCCs + Δ-MFCC + statistical aggregations (160-D feature vectors)

Data Augmentation:
- Noise Injection (Gaussian noise, factor=0.005)
- Time Stretching (rate=0.8-0.95)
- Fixed length padding/truncation (4 seconds)
"""

import os
import glob
import numpy as np
import librosa
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical

# ======================================================================
# 1. Audio Preprocessing & Augmentation Utilities
# ======================================================================

def add_noise(signal, noise_factor=0.005):
    """Adds random Gaussian noise to an audio waveform."""
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise

def stretch_time(signal, rate=0.85):
    """Time-stretches an audio signal using librosa."""
    try:
        return librosa.effects.time_stretch(y=signal, rate=rate)
    except Exception:
        return signal

def extract_mel_spectrogram(signal, sr, n_mels=128, n_fft=2048, hop_length=512, max_pad_len=174):
    """
    Computes Log-Mel Spectrogram padded/truncated to a fixed temporal dimension.
    Returns array of shape (n_mels, max_pad_len).
    """
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    if log_mel.shape[1] > max_pad_len:
        log_mel = log_mel[:, :max_pad_len]
    else:
        pad_width = max_pad_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        
    return log_mel

def extract_mfcc_160d(signal, sr, n_mfcc=40):
    """
    Engineers a 160-dimensional feature vector combining 40 MFCCs, 
    delta-MFCCs, and statistical aggregations (mean & std).
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)
    
    # Concatenate to 160-D vector (40 + 40 + 40 + 40)
    feat_160d = np.hstack([mfcc_mean, mfcc_std, delta_mean, delta_std])
    return feat_160d

# ======================================================================
# 2. Pipeline 1: Animal Sound Classification (VGG16 Transfer Learning)
# ======================================================================

def build_vgg16_animal_model(input_shape=(128, 174, 3), num_classes=3):
    """
    Constructs a transfer learning model leveraging frozen VGG16 base
    with custom classification head.
    """
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze pretrained features
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ======================================================================
# 3. Pipeline 2: Gunshot Detection (Kernel SVM RBF)
# ======================================================================

def build_svm_gunshot_detector(C=2.0, gamma='scale'):
    """
    Constructs a Support Vector Classifier with RBF Kernel for binary gunshot detection.
    """
    return SVC(
        kernel='rbf',
        C=C,
        gamma=gamma,
        probability=True,
        class_weight='balanced',
        random_state=42
    )

# ======================================================================
# 4. Consolidated Dual-Model Inference Pipeline
# ======================================================================

class ForestAcousticSurveillanceSystem:
    def __init__(self, cnn_model=None, svm_model=None, scaler=None):
        self.cnn_model = cnn_model
        self.svm_model = svm_model
        self.scaler = scaler
        self.animal_classes = {0: "Lion", 1: "Donkey", 2: "Monkey"}

    def predict(self, audio_file_path):
        """
        Runs dual-inference on an incoming audio file:
        1. Evaluates binary gunshot threat level via SVM.
        2. Classifies animal sound species via VGG16 CNN.
        """
        signal, sr = librosa.load(audio_file_path, sr=None)
        
        # 1. Gunshot Detection (SVM)
        mfcc_feat = extract_mfcc_160d(signal, sr).reshape(1, -1)
        if self.scaler is not None:
            mfcc_feat = self.scaler.transform(mfcc_feat)
            
        gunshot_prob = self.svm_model.predict_proba(mfcc_feat)[0][1] if self.svm_model else 0.0
        gunshot_detected = gunshot_prob >= 0.5
        
        # 2. Animal Species Classification (CNN)
        mel_spec = extract_mel_spectrogram(signal, sr)
        mel_rgb = np.stack([mel_spec, mel_spec, mel_spec], axis=-1)
        mel_rgb = np.expand_dims(mel_rgb, axis=0)
        
        if self.cnn_model:
            animal_preds = self.cnn_model.predict(mel_rgb, verbose=0)[0]
            animal_class_id = np.argmax(animal_preds)
            animal_name = self.animal_classes.get(animal_class_id, "Unknown")
            animal_conf = animal_preds[animal_class_id]
        else:
            animal_name = "N/A"
            animal_conf = 0.0
            
        return {
            "gunshot_threat": {
                "detected": gunshot_detected,
                "probability": float(gunshot_prob)
            },
            "animal_classification": {
                "species": animal_name,
                "confidence": float(animal_conf)
            }
        }

if __name__ == "__main__":
    print("Dual-Model Audio Intelligence Pipeline Initialized.")
    print("Run model training notebooks or import functions into your pipeline.")

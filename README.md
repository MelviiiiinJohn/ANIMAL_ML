# 🎧 Intelligent Audio Recognition — Gunshot & Forest Animal Sound Classification

> **Research Project · 2025** | Python · TensorFlow · Scikit-learn · CNN · SVM

A **dual-model audio intelligence system** that combines deep learning and classical machine learning for real-time acoustic surveillance in forest environments.

---

## 🏗️ Architecture Overview

```
Audio Input (.wav)
       │
       ├──► [CNN Pipeline]  Mel-spectrogram (128 mel bands)
       │         └──► VGG16 Transfer Learning (frozen base)
       │                   └──► 3-class Animal Classification
       │                         Lion | Donkey | Monkey
       │
       └──► [SVM Pipeline]  MFCC + Δ-MFCC (160-D vectors)
                 └──► Kernel SVM (RBF, C=2.0)
                           └──► Binary Gunshot Detection
                                 Gunshot | No Gunshot
```

---

## 📊 Results

| Model | Task | Accuracy | ROC-AUC |
|---|---|---|---|
| VGG16 CNN | Animal Classification (Lion / Donkey / Monkey) | **90.7%** | — |
| Kernel SVM (RBF) | Gunshot Detection (binary) | **97%** | **0.995** |

---

## 📁 Repository Structure

```
├── audio_recognition_model.py   # ← Main script (clean, consolidated)
├── Untitled9 (2).ipynb          # ← Full Colab notebook with outputs & plots
├── requirements.txt             # ← All dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Model Details

### Model 1 — Animal Sound Classifier (CNN)
| Parameter | Value |
|---|---|
| Base model | VGG16 (ImageNet pretrained, frozen) |
| Input features | Log-Mel spectrogram (128 mel bands) |
| Input shape | (128, 174, 3) |
| Head | Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(3, Softmax) |
| Classes | Lion, Donkey, Monkey |
| Data split | 70% train / 15% val / 15% test |
| Augmentation | Noise injection, time-stretching, resampling |
| Dataset | Custom animal sound dataset (GitHub: YashNita/Animal-Sound-Dataset) |

### Model 2 — Gunshot Detector (SVM)
| Parameter | Value |
|---|---|
| Model | Kernel SVM, RBF kernel (C=2.0, gamma='scale') |
| Input features | 40-coeff MFCC + Δ-MFCC → 160-D feature vector |
| Data split | UrbanSound8K 10-fold: folds 1–8 train, fold 9 val, fold 10 test |
| Class balancing | Negative downsampling (1:3 ratio) + `class_weight='balanced'` |
| Dataset | [UrbanSound8K](https://urbansounddataset.weebly.com/) (8,732 labelled clips) |
| Threshold tuning | Best F1 threshold on validation fold |

---

## ⚙️ Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare datasets

**Animal sounds:**
```bash
git clone https://github.com/YashNita/Animal-Sound-Dataset
```

**UrbanSound8K (gunshot detection):**  
Download from [Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) and place under `datasets/`.
```
datasets/
├── fold1/ ... fold10/
└── UrbanSound8K.csv
```

### 3. Run the full pipeline
```bash
python audio_recognition_model.py
```

### 4. Or open the Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13ZSl0G_h050afTPL92mFCr3cq_DPXgTA)

---

## 🔧 Data Augmentation

Applied to both pipelines to improve robustness under noisy real-world acoustic conditions:

- **Noise injection** — Gaussian noise added (factor = 0.005)
- **Time-stretching** — Rate = 0.8–0.9× speed variation
- **Resampling** — Fixed-length audio segments (4 seconds)

---

## 📦 Key Libraries

| Library | Purpose |
|---|---|
| `librosa` | Audio loading, MFCC, Mel-spectrogram, delta features |
| `TensorFlow / Keras` | VGG16 transfer learning, CNN training |
| `scikit-learn` | SVM, StandardScaler, metrics, train/test split |
| `numpy` | Feature arrays and matrix ops |
| `matplotlib / seaborn` | Training curves, confusion matrices |
| `joblib` | SVM model serialization |

---

## 🗂️ Dataset Credits

- **Animal Sound Dataset** — [YashNita/Animal-Sound-Dataset](https://github.com/YashNita/Animal-Sound-Dataset)
- **UrbanSound8K** — Salamon, J., Jacoby, C., & Bello, J. P. (2014). ACM Multimedia. [Link](https://urbansounddataset.weebly.com/)

---

## 👤 Author

**Melvin John**  
*Audio ML · Computer Vision · Deep Learning*

---

> ⚠️ **Note:** No API keys, credentials, or `kaggle.json` files are stored in this repository. See `.gitignore` for details.

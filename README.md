# 🧠 Diabetes Prediction using Deep Neural Networks (TensorFlow & PyTorch)

This project implements a binary classification system to predict the likelihood of diabetes using feedforward neural networks in both **TensorFlow** and **PyTorch**. The model is trained on a medical dataset with 8 normalized features and one binary outcome.

Thanks, Monica! I’ve thoroughly reviewed your `Assignment1.pdf`, which includes well-documented TensorFlow and PyTorch implementations of binary classification using a feedforward neural network on the diabetes dataset.

---

## 📊 Dataset Overview

* The dataset consists of **759 rows** and **9 columns** (8 features + 1 binary target).
* It includes anonymized, normalized patient data.

| Column     | Description                            |
| ---------- | -------------------------------------- |
| Feature1–8 | Numeric features                       |
| Feature9   | Target (0 = No Diabetes, 1 = Diabetes) |

---

## 🔎 Problem Statement

Predict whether a patient has diabetes based on health-related features using feedforward neural networks.

---

## ⚙️ Technologies Used

| Framework            | Purpose                             |
| -------------------- | ----------------------------------- |
| TensorFlow           | Neural network modeling (Keras API) |
| PyTorch              | Neural network modeling (nn.Module) |
| Pandas / NumPy       | Data manipulation                   |
| Seaborn / Matplotlib | Visualization                       |
| Scikit-learn         | Data splitting, metrics             |

---

## 🔬 Workflow Summary

### ✅ Data Preparation

* Normalized numeric features
* Checked for missing values
* Train-test split: 70% / 30%

### 📊 Exploratory Data Analysis

* Histograms of features
* Correlation heatmap
* Class distribution bar plot

---

## 🧠 Model Architectures

### 🔹 Base Model (both TF & PyTorch)

```
Input (8 features) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Sigmoid)
```

### 🔹 Hypothesis 1: Wider Network

```
Input → Dense(64) → Dense(32) → Output
```

### 🔹 Hypothesis 2: Deeper + Wider Network

```
Input → Dense(64) → Dense(32) → Dense(16) → Output
```

---

## 📈 Results Summary

| Model Type           | Accuracy | F1-Score (Class 1) |
| -------------------- | -------- | ------------------ |
| Original TF Model    | 0.7368   | 0.80               |
| Wider TF Model       | 0.7588   | 0.82               |
| Deeper TF Model      | 0.7588   | 0.82               |
| Original PyTorch     | 0.7368   | 0.80               |
| Wider PyTorch Model  | 0.7544   | 0.82               |
| Deeper PyTorch Model | 0.7675   | 0.81               |

> 🔍 Conclusion: Increased width/depth improves performance slightly. Deeper PyTorch model achieved the best results.

---

## 📊 Visualizations

* Accuracy and loss over epochs
* Model comparison plots
* Classification reports

---

## 📌 Key Learnings

* **Width** improves feature learning and helps shallow networks generalize better.
* **Depth** improves hierarchical pattern learning but may cause vanishing gradients if not managed properly.
* Both frameworks (TF and PyTorch) achieved similar performance with comparable architectures.

---



# 🌍 AQI Escalation Prediction using Symbolic Deep-Learning

![AQI](https://img.shields.io/badge/AQI-Health%20Risk-red)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Deep Learning](https://img.shields.io/badge/Model-LSTM-orange)
![ONNX](https://img.shields.io/badge/Deployment-ONNX-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)
![MongoDB](https://img.shields.io/badge/MongoDB-Symbolic_Data-brightgreen)
![Time Series](https://img.shields.io/badge/Time--Series-Temporal_Modeling-purple)

---

## 🔗 Live Demo

👉 **Gradio App:**  
https://21494652413a323986.gradio.live/

---


![AQI Escalation Prediction UI](Screenshot%202026-01-05%20201915.png)
## 📌 Project Overview

This project predicts **air-quality health risk escalation** using **symbolic time-series modeling** on AQI data.

Instead of using raw sensor values directly, the system:
1. Converts AQI data into **symbolic interval representations**
2. Learns **temporal escalation patterns** using an **LSTM**
3. Deploys the trained model as a **lightweight ONNX inference service**
4. Provides an interactive **Gradio UI** for real-time predictions

🎯 **Goal**: Predict whether air quality is likely to escalate into **Low / Medium / High health risk** in the coming week.


Air quality does not become dangerous suddenly — it **escalates over time**.

Traditional AQI models:
- React to **single-day AQI**
- Ignore **temporal escalation**
- Are noisy and unstable

### ❌ Limitations of Raw AQI Modeling
- High variance
- Poor interpretability
- Weak temporal understanding

### ✅ Our Solution
- Convert AQI into **symbolic interval features**
- Use **time-sequence modeling**
- Predict **risk escalation**, not just AQI value

---

## 🧠 Core Idea (Simple Explanation)

> Instead of asking  
> **“What is today’s AQI?”**  
>  
> We ask  
> **“How has AQI behaved over the last 4 weeks, and is the risk increasing?”**

---

## 🧩 Data Pipeline (Step-by-Step)

### 1️⃣ Raw AQI Data
- Source: India AQI
- Frequency: Daily AQI readings

---

### 2️⃣ Symbolic Feature Conversion

Each **week** is converted into symbolic features and stored in MongoDB:

| Feature | Meaning |
|------|--------|
| `aqi_min` | Minimum AQI in the week |
| `aqi_max` | Maximum AQI |
| `aqi_median` | Typical AQI |
| `aqi_std` | AQI variability |
| `very_poor_days` | Days AQI > 200 |
| `severe_days` | Days AQI > 300 |

✔ Reduces noise  
✔ Improves stability  
✔ Enhances interpretability  

---

### 3️⃣ Time-Series Construction

- Input = **4 consecutive weeks**
- Each week = **6 symbolic features**
- Final input shape:

#### Example Input (UI JSON)
```json
[
 [-0.08, -0.77, -0.46, -1.15, -0.43, -0.20],
 [-0.62, -0.54, -0.66, -0.35, -0.43, -0.20],
 [-0.59, -0.93, -0.79, -0.92, -0.43, -0.20],
 [-0.66, -0.81, -0.75, -0.66, -0.43, -0.20]
]
```


## 🗂 Dataset Pipeline

Raw AQI CSV
↓
Weekly Aggregation
↓
Symbolic Interval Encoding
↓
Risk Label Generation
↓
Time-Series Sequences


### Risk Labels
| Label | Meaning |
|------|--------|
| Low | Stable / safe AQI |
| Medium | Warning-level escalation |
| High | Severe health risk |

---

## 🧱 Architecture
            ┌──────────────────┐
            │   Raw AQI Data   │
            └─────────┬────────┘
                      ↓
            ┌──────────────────┐
            │ Symbolic Encoding │
            │ (min,max,median…)│
            └─────────┬────────┘
                      ↓
            ┌──────────────────┐
            │ MongoDB Storage  │
            │ (Hierarchical)   │
            └─────────┬────────┘
                      ↓
            ┌──────────────────┐
            │ LSTM Time Series │
            │ (4-week window) │
            └─────────┬────────┘
                      ↓
            ┌──────────────────┐
            │ Risk Prediction  │
            │ Low / Med / High │
            └─────────┬────────┘
                      ↓
            ┌──────────────────┐
            │ ONNX Deployment  │
            └─────────┬────────┘
                      ↓
            ┌──────────────────┐
            │ Gradio Web UI    │
            └──────────────────┘

---

## 🧪 Why Not CNN?

A CNN was initially tested on symbolic vectors, but:

❌ CNN ignores temporal order  
❌ Cannot model escalation dynamics  
❌ Treats weeks independently  

### Why LSTM?
✔ Captures **week-to-week progression**  
✔ Models **memory & escalation**  
✔ Ideal for **temporal health risk forecasting**

---

## ⏱ Time-Series Modeling

**Input**:  
4 consecutive weeks × 6 symbolic features  

---

## 🧪 Why Not CNN?

A CNN was initially tested on symbolic vectors, but:

❌ CNN ignores temporal order  
❌ Cannot model escalation dynamics  
❌ Treats weeks independently  

### Why LSTM?
✔ Captures **week-to-week progression**  
✔ Models **memory & escalation**  
✔ Ideal for **temporal health risk forecasting**

---

## ⏱ Time-Series Modeling

**Input**:  
4 consecutive weeks × 6 symbolic features  

[
[-0.08,-0.77,-0.46,-1.15,-0.43,-0.20],
[-0.62,-0.54,-0.66,-0.35,-0.43,-0.20],
[-0.59,-0.93,-0.79,-0.92,-0.43,-0.20],
[-0.66,-0.81,-0.75,-0.66,-0.43,-0.20]
]


**Output**:
Low: 96%
Medium: 2%
High: 2%
---

## 📊 Model Results & Insights

### Classification Performance
- Strong **Low-risk precision**
- Medium risk learned via **threshold tuning**
- High risk recall improved via **class weighting**

### Key Insight
> AQI escalation is not about a single spike, but **persistent upward trends across weeks**.

---

## 💾 Why MongoDB?

Symbolic AQI records are **self-contained objects**:

```json
{
  "area": "Delhi",
  "week": "2022-04-01",
  "aqi_interval": {
    "min": 110,
    "max": 290,
    "median": 180,
    "std": 42
  },
  "escalation": {
    "very_poor_days": 3,
    "severe_days": 1
  },
  "risk_level": "High"
}
```

## 🚀 Deployment Strategy

### Why ONNX?

The trained LSTM model is exported to **ONNX (Open Neural Network Exchange)** to enable efficient, production-ready deployment.

Key advantages:
- **No PyTorch dependency** at inference time
- **Lightweight model footprint**
- **Fast CPU-based inference**
- **Cloud and edge friendly**
- **Framework-agnostic execution**

ONNX ensures that the model can be deployed reliably across environments without requiring heavy deep learning frameworks.

---

### Deployment Stack

LSTM Model (PyTorch)
↓
ONNX Model Export
↓
ONNX Runtime (Inference Engine)
↓
Gradio Web Interface


---

### 🔗 Live Demo

👉 **https://21494652413a323986.gradio.live/**

The application is publicly accessible and allows real-time inference using symbolic AQI inputs.

---

## 🖥 User Interface

The system is deployed using **Gradio**, providing a clean and interactive web interface.

Users can:
- Paste **weekly symbolic AQI sequences** (JSON format)
- Receive **real-time escalation risk probabilities**
- Interpret results visually without technical knowledge

**Design goals achieved:**
- ✔ No machine learning background required  
- ✔ Suitable for health monitoring and policy analysis  
- ✔ Lightweight and responsive inference  

---

## 🧰 Tech Stack

### Core Technologies
- **Python**
- **NumPy**
- **Pandas**

### Modeling & Inference
- **PyTorch (LSTM)**
- **ONNX**
- **ONNX Runtime**

### Data Storage
- **MongoDB** (Symbolic & hierarchical AQI data)

### Deployment
- **Gradio**
- **Cloud-ready inference pipeline**

---

## 🧠 Concepts Used

- Symbolic Data Analysis
- Time-Series Modeling
- Long Short-Term Memory (LSTM) Networks
- Class Imbalance Handling
- Health Risk Escalation Modeling
- Model Compression & Optimization (ONNX)
- Explainable AI (XAI)

---

## 📚 References

- Hochreiter, S., & Schmidhuber, J. (1997).  
  *Long Short-Term Memory*. Neural Computation.  
  https://www.bioinf.jku.at/publications/older/2604.pdf

- Bock, H. H., & Diday, E. (2000).  
  *Analysis of Symbolic Data*. Springer.  
  https://link.springer.com/book/10.1007/978-3-642-57155-8

- World Health Organization (WHO).  
  *Ambient Air Pollution and Health*.  
  https://www.who.int/teams/environment-climate-change-and-health/air-quality

- ONNX Runtime Documentation.  
  https://onnxruntime.ai/docs/

- PyTorch Time Series Modeling Guide.  
  https://pytorch.org/tutorials/

---

## 🏁 Conclusion

This project demonstrates that **symbolic representations combined with temporal learning** can outperform raw AQI modeling for **real-world health risk prediction**.

The solution is:
- ✔ **Explainable**
- ✔ **Lightweight**
- ✔ **Deployable**
- ✔ **Scalable**

It bridges data science research and practical deployment for environmental health monitoring.

---

## 👩‍💻 Author

**Priyanka K**  
MSc Data Science  
Christ University  
📍 Bengaluru, India







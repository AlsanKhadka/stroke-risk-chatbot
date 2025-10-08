# 🧠 AI Stroke Risk Prediction Chatbot

A machine learning–based web chatbot that predicts stroke risk levels (Low / Medium / High) through interactive user input.  
Developed as part of the **CET140 Specialist Project** — *University of Sunderland (2025)*.

## 🚀 Features

- 🤖 Real-time stroke risk prediction  
- ⚙️ Built with **Python**, **Flask**, and **XGBoost**  
- ⚖️ Dataset balanced using **SMOTEENN**  
- 💬 Conversational interface with validation  
- 💡 Provides diet & exercise suggestions  
- 🛡️ Includes ethical disclaimer  

## ⚙️ How to Run
1. Open **`stroke_risk_prediction.ipynb`** in Google Colab.  
2. Run all cells — it trains the model and launches the chatbot via **ngrok**.

## 🧠 Methodology

- Dataset: Kaggle Stroke Prediction Dataset

- Preprocessing: Missing values, encoding, normalization, SMOTEENN balancing

- Models: Logistic Regression, Random Forest, XGBoost

- Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC

- Best Model: Calibrated XGBoost (Accuracy 0.964 / ROC-AUC 0.994)

- Integration: Deployed via Flask chatbot interface


## 🌟 Future Enhancements

- Voice-based interaction

- Stroke-risk visual meter

- Integration with wearable devices

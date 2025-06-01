## 🚨 Intrusion Detection System using Machine Learning
A multi-model intrusion detection system built on the **NSL-KDD dataset**. This project includes end-to-end data preprocessing, feature engineering, training multiple ML models (Random Forest, ANN, etc.), saving model pipelines, and deploying the final model using **Streamlit** for real-time inference.

---

## 📁 Project Structure

```
intrusion-detection-multimodel/
│
├── app.py                            # Streamlit web app
├── requirements.txt                  # Project dependencies
├── sample_input_original.csv        # Sample test input for app
│
├── models/
│   ├── final_rf_model.pkl           # Trained Random Forest model
│   ├── final_scaler.pkl             # StandardScaler object
│   ├── final_encoders.pkl           # LabelEncoders for categorical cols
│   └── final_label_encoder.pkl      # LabelEncoder for labels
│
├── data/
│       ├── X_train.csv
│       ├── X_val.csv
│       ├── y_train.csv
│       └── y_val.csv
│
└── notebooks/
    └── 08_Streamlit_Web_App_Preparation.ipynb
```

---

## 📊 Dataset

* **NSL-KDD Dataset**
* Cleaned and preprocessed from original version.
* Features include: protocol type, service, flag, count features, host-based metrics, etc.

---

## 🔧 Steps Followed

### ✅ 1. Data Preprocessing

* Label encoded: `protocol_type`, `service`, `flag`
* Normalized numeric features using `StandardScaler`
* Encoded label: `normal` → 1, `attack` → 0
* Split into train/test and saved to `data/processed/`

### ✅ 2. Model Training

* Trained multiple models including:

  * Random Forest ✅
  * ANN (Keras)
  * Logistic Regression, SVM (optional)
* Random Forest chosen for best balance between accuracy & simplicity
* Saved model and preprocessing artifacts using `joblib`

### ✅ 3. Model Saving

* `final_rf_model.pkl`
* `final_scaler.pkl`
* `final_encoders.pkl`
* `final_label_encoder.pkl`

### ✅ 4. Streamlit Web App

* Allows users to upload CSV files (no label column)
* Predicts whether network traffic is **normal** or an **intrusion**
* Displays result in tabular format

---

## 🚀 Run the Web App

### 🔹 Local Setup
1. Install dependencies:
pip install -r requirements.txt
2. Run app locally:
streamlit run app.py
3. Go to browser:
http://localhost:8501
```

### 🔹 Upload File Example

Use the `sample_input_original.csv` file included in the repo to test the app.

📌 **Note:** Ensure uploaded file has **same schema** as `X_train.csv` (after preprocessing but before encoding/scaling).

---

## 🛆 Deployment

Successfully deployed on **Streamlit Cloud**. Ensure:

* All `.pkl` and `.csv` files are committed to GitHub
* `requirements.txt` is included
* `app.py` is in root directory

---

## 📚 Requirements

Generated via `pip freeze`:

```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 🧐 Model Info

| Metric       | Value                             |
| ------------ | --------------------------------- |
| Accuracy     | \~99%                             |
| Model        | Random Forest                     |
| Dataset      | NSL-KDD                           |
| Input Format | Preprocessed CSV with 41 features |

---

## 👨‍💻 Author

* **Sanchay Chauhan**
* GitHub: [Sanchay987](https://github.com/Sanchay987)

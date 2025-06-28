# Hypohyroid-Prediction-Model
This project focuses on predicting hypothyroidism using a machine learning classification model. It involves comprehensive data preprocessing, model training with multiple algorithms, evaluation, and final model deployment for prediction.

📌 Project Highlights
📊 Exploratory Data Analysis on medical data related to thyroid function

🧹 Handling missing values and outliers

🧬 Feature encoding and transformation

⚖️ Balancing the dataset using SMOTE

🤖 Model training using multiple classifiers:

Logistic Regression

Random Forest

XGBoost

LightGBM

CatBoost

VotingClassifier ensemble

🏆 Best model selection based on F1 Score and ROC-AUC

💾 Final model saved using joblib

🧪 Tested on holdout test set

📂 Project Structure
bash
Copy
Edit
hypothyroid-prediction/
│
├── app.py                  # (Optional) Flask/Streamlit app for deployment
├── hypothyroid_final_model.pkl  # Saved best-performing model
├── Hypothyroid Prediction Model.html  # Original notebook as HTML
├── requirements.txt        # Python dependencies
└── README.md               # This file
🧠 Dataset
The dataset contains patient records with medical attributes such as:

Age

TSH (Thyroid-Stimulating Hormone)

TT4 (Total Thyroxine)

T3 (Triiodothyronine)

... and more.

The target variable is binaryClass, indicating whether the patient is Positive (P) or Negative (N) for hypothyroidism.

🚀 Model Building Steps
Data Cleaning:

Replacing missing values

Dropping irrelevant features

Feature Engineering:

One-hot encoding of categorical variables

Scaling of numerical features

Balancing:

Addressed class imbalance with SMOTE

Model Training & Evaluation:

Compared multiple classifiers using cross-validation

Chose best model based on F1 score

Saving the Model:

Final model saved as hypothyroid_final_model.pkl using joblib

🛠️ Installation & Usage
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/hypothyroid-prediction.git
cd hypothyroid-prediction
Create a Virtual Environment

bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install Requirements

bash
Copy
Edit
pip install -r requirements.txt
Run the Web App (Optional if using app.py)

bash
Copy
Edit
streamlit run app.py
🧪 Example Prediction
Once the model is loaded using joblib.load('hypothyroid_final_model.pkl'), you can predict as follows:

python
Copy
Edit
import joblib
model = joblib.load("hypothyroid_final_model.pkl")
pred = model.predict(X_new)
📦 Dependencies
scikit-learn

xgboost

lightgbm

catboost

imblearn

joblib

pandas, numpy, matplotlib, seaborn

All dependencies are listed in requirements.txt

📈 Results
Achieved high accuracy and F1 score on test data

ROC-AUC and confusion matrix used for final evaluation

Balanced class predictions using SMOTE

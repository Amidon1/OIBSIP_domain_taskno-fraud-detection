💳 Credit Card Fraud Detection Using Machine Learning


This project uses machine learning to detect fraudulent credit card transactions. The goal is to help financial institutions automatically flag suspicious activity and reduce financial loss.

📌 Objectives
Analyze transaction data to understand patterns of fraud.

Build a classification model that accurately identifies fraudulent transactions.

Evaluate model performance using metrics like precision, recall, and F1-score.

Create a simple user interface using Streamlit for testing the model in real-time.

🧠 What the Model Looks At
Since fraud data is highly imbalanced (few frauds vs. many normal transactions), the model is trained carefully using features that represent:

Transaction amounts

Anonymized features (V1, V2, ..., V28) from PCA for privacy

Time-based trends

Binary label (0 = normal, 1 = fraud)

⚙ Tools & Libraries
Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn (for EDA and visualizations)

Streamlit (for web-based app)

pickle (for saving the model)

📈 ML Models Used
We experimented with:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

After comparing performance, we selected the best-performing model based on F1-Score and ROC-AUC due to class imbalance.

🔎 How the App Works
The Streamlit app lets users:

Upload new transaction data.



📊 Evaluation Metrics
Due to the imbalance, we focused on:

Precision (How many predicted frauds were correct?)

Recall (How many actual frauds did we catch?)

F1-Score (Balance of precision and recall)

ROC-AUC (Overall ability to distinguish fraud from normal)

🙋🏽 Why This Project Matters
This project demonstrates how machine learning can help financial institutions prevent fraud. It shows practical skills in data preprocessing, model training, and interface deployment — all critical for real-world data science work.


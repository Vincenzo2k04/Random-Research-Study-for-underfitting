# 💘 Predicting Match Outcomes on Dating Apps — A Study on Underfitting & Model Selection

This project is a **research-focused initiative** aimed at understanding how various machine learning models behave when trained on behavioral data from a dating application. The key emphasis is on exploring **underfitting**, its causes, and ways to mitigate it using both traditional and advanced ML techniques.

---

## 📂 Dataset Overview

The dataset (`dating_app_behavior_dataset.csv`) contains anonymized behavioral features collected from a dating app. Each record includes metrics such as:

- Number of swipes, messages, profile interactions
- Categorical attributes (like location, interests)
- A target column: `match_outcome` (representing the success of a potential match)

---

## 📚 Libraries Used

To build, train, and evaluate the models, the following Python libraries were used:

```bash
pandas
numpy
scikit-learn
xgboost
tensorflow / keras
matplotlib (optional, for plotting)

🧠 Models Implemented
✅ Preprocessing Steps
Null value removal

Outlier detection using IQR

Feature normalization using MinMaxScaler

Categorical encoding with LabelEncoder

🧪 Models Explored

Model	Status	Observations
Logistic Regression	✅ Done	Showed signs of underfitting
SVM	✅ Done	Improved slightly; sensitive to data scaling
Neural Networks	✅ Done	Flexible, deeper layers helped reduce underfit
XGBoost	✅ Done (Tuned)	Most performant, robust to over/underfitting
❗ Focus: UNDERFITTING
Underfitting occurs when the model is too simple to capture underlying trends in the data.
Symptoms observed:

Low training & testing accuracy across simpler models

Failure to learn patterns in complex or noisy data

🛠️ How I Tried Fixing Underfitting
Using deeper networks in the neural model (3 hidden layers, dropout, batchnorm)

Increasing model complexity in XGBoost (tuning n_estimators, max_depth, etc.)

Adding feature normalization and encoding for better signal extraction

Avoiding overly simple models like Logistic Regression without polynomial features

🚀 Future Scope
To further improve model performance and study the underfitting-overfitting balance:

Feature Engineering:

Introduce domain-driven features (e.g., response time, emoji usage)

Data Augmentation:

Simulate new realistic user behavior combinations

Hyperparameter Tuning:

Use GridSearchCV for model-specific optimization

Ensemble Techniques:

Combine neural and XGBoost predictions for stability

Cross-Validation:

Use K-Fold CV to reduce variance and assess generalizability

🧾 Conclusion
This research project emphasizes that model performance is not solely about complexity but rather about balance. Logistic Regression might underfit while Neural Networks and XGBoost generalize better with appropriate data treatment and regularization.

Underfitting is avoidable, but only with:

Sufficient feature richness

Proper preprocessing

Tuning model architecture and hyperparameters
# 🚢 Titanic - Machine Learning from Disaster
The goal of this project is to predict the survival of passengers on the Titanic based on various features such as age, sex, class, number of siblings/spouses aboard, and more.
This is a binary classification task — the model predicts whether a passenger survived (1) or not (0).

## 📁 Project Structure
```
Titanic_predict/
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for EDA and model experiments
├── scripts/            # Python scripts for preprocessing, training, and prediction
├── models/             # Saved trained models (pickle/joblib)
├── submission/         # Generated Kaggle submission files
└── README.md           # Project description
```
## 🧰 Technologies Used

Python: Pandas, NumPy, Matplotlib, Seaborn

Machine Learning: scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)

Advanced models: XGBoost / LightGBM

Model tuning: GridSearchCV, cross-validation

Feature Engineering: missing value imputation, categorical encoding, new feature creation

Ensembling: Stacking / Averaging multiple models

## 🔍 Workflow
### 1️⃣ Exploratory Data Analysis (EDA)

Check and handle missing values

Visualize class distributions (e.g., by gender, class, age)

Analyze correlations and relationships between features and survival

### 2️⃣ Data Preprocessing

Impute missing values (median, mode, or domain-specific rules)

Encode categorical features (Sex, Embarked, etc.)

Scale numeric features where needed

Create new features (e.g., family size, title extraction from name)

### 3️⃣ Model Training

Train baseline models (Logistic Regression, RandomForest, GradientBoosting)

Evaluate with cross-validation (StratifiedKFold)

Tune hyperparameters with GridSearchCV

Combine multiple models for better performance (Stacking / Voting Ensemble)

### 4️⃣ Final Prediction and Submission

Predict survival on the test dataset

Generate a file submission.csv in Kaggle format

Submit and check the score on the leaderboard

## 📊 Results

. Best accuracy on public leaderboard: ~0.80 – 0.82

. Stable and interpretable predictions due to clean preprocessing and model blending

. Insights into key survival factors: Sex, Pclass, Age, FamilySize, Fare

```
git clone https://github.com/BorisBaghiyan/Titanic_predict.git
cd Titanic_predict/Titanik
```
## 🚀 Future Improvements

Add feature importance visualization (SHAP, permutation importance)

Try deep learning models (MLP, TabNet)

Build a web interface using Streamlit or Flask for live predictions

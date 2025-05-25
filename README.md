# 🚢 Titanic Survival Prediction – Machine Learning    
**Author:** Nitish Raj Vinnakota | [LinkedIn](https://linkedin.com/in/vnr-nitish)

---

## 🔍 Project Overview

This project tackles the classic **Titanic dataset challenge**, where the goal is to predict which passengers survived the Titanic shipwreck. By analyzing passenger attributes such as age, gender, fare, and class, we apply **supervised machine learning algorithms** to determine survival probabilities.

This end-to-end pipeline demonstrates practical skills in **EDA, data preprocessing, encoding, feature selection, model building, and evaluation** using industry-standard tools.

---

## 🎯 Objective

> To build a robust classification model that predicts whether a passenger survived or not, based on features like age, sex, fare, and socio-economic class.

---

## 📁 Dataset

- **Source:** [Kaggle – Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **Rows:** 891 passengers  
- **Features:** 12 columns including:
  - `Survived` (Target), `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

---

## 🧹 Data Preprocessing & Cleaning

- Dropped columns with high missing values: `Cabin`, `Name`, `Ticket`
- Filled missing values:
  - `Age`: mean
  - `Embarked`: mode
- Removed irrelevant columns: `PassengerId`, etc.
- Encoded categorical features (`Sex`, `Embarked`, `Pclass`) using:
  - `LabelEncoder`, `OneHotEncoder`, and `pd.get_dummies`
- Scaled numerical features where required using `StandardScaler`

---

## 📊 Exploratory Data Analysis (EDA)

- Plotted survival rates by gender, class, port of embarkation, and family members onboard
- Observed survival trends:
  - Females had higher survival rates than males
  - Higher-class passengers had a better chance of survival
  - Age group 20–40 had higher mortality
- Outlier detection via boxplots (notably in `Fare` and `Age`)
- Used correlation heatmaps to identify important relationships between features

---

## 🧠 Machine Learning Models Used

| Model                   | Accuracy | Notes                                 |
|------------------------|----------|---------------------------------------|
| Logistic Regression     | 79%      | Baseline model, balanced performance |
| K-Nearest Neighbors     | 65%      | Struggled with imbalanced data       |
| Support Vector Machine  | 81%      | Best overall performance              |
| Random Forest           | 57%      | Overfitting issues, high variance     |
| Decision Tree (tested)  | (N/A)    | Explored but not finalized            |

✅ Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
✅ Train-Test Split: 80:20  
✅ Used `Pipeline` for consistent preprocessing and modeling

---

## 📌 Key Skills Demonstrated

- `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- `scikit-learn`: Classifiers, Pipelines, Imputation
- Label and One-Hot Encoding
- Outlier Handling & Data Imputation
- Cross-model comparison & accuracy tuning

---

## 🏁 Results & Conclusion

- **SVC (Support Vector Classifier)** achieved the best result with **81% testing accuracy**
- Logistic Regression followed closely with 79% accuracy
- KNN and Random Forest showed overfitting or poor generalization
- The project proves how thoughtful preprocessing and model choice impact predictive power

---

## 🚀 Future Enhancements

- Add **hyperparameter tuning** using `GridSearchCV`
- Include **ensemble methods** like XGBoost or Gradient Boosting
- Address class imbalance using `SMOTE` or `class_weight`
- Build a **frontend using Streamlit** for user-friendly prediction interface

---

## 📫 Contact

**Email:** nvinnako2@gitam.in  
**LinkedIn:** [linkedin.com/in/vnr-nitish](https://linkedin.com/in/vnr-nitish)

---

> *"A data-driven story of survival and model optimization through real-world insights."*

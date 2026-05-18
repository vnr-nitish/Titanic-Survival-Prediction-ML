# Titanic Survival Classification using Machine Learning

## Project Overview

This project predicts the survival of Titanic passengers using Machine Learning classification techniques. The implementation includes data preprocessing, missing value handling, feature engineering, model training, and comparison of multiple classification algorithms to identify the best-performing model.

The project evaluates different models and analyzes their prediction performance using standard evaluation metrics.

---

## Features

✔ Data cleaning and preprocessing  
✔ Missing value handling using SimpleImputer  
✔ Feature transformation and encoding  
✔ Exploratory Data Analysis (EDA)  
✔ Multiple machine learning algorithms implementation  
✔ Model performance comparison  
✔ Visualization and evaluation metrics  

---

## Dataset Information

Dataset: Titanic Dataset

Features used include:

- Passenger ID
- Passenger Class (Pclass)
- Sex
- Age
- Siblings/Spouses aboard (SibSp)
- Parents/Children aboard (Parch)
- Fare
- Embarked

Target Variable:

**Survived**

- 0 → Did not survive
- 1 → Survived

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Data Preprocessing Steps

### Data Cleaning
Removed unnecessary columns:

- Name
- Ticket
- Cabin

### Missing Value Handling

- Mean imputation
- Median imputation
- Most frequent value handling
- SimpleImputer

### Feature Engineering

- One-hot encoding
- Dummy variable generation
- Feature scaling using StandardScaler

---

## Machine Learning Models Used

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Classifier (SVC)
4. Random Forest Classifier
5. Decision Tree Classifier
6. AdaBoost Classifier

---

## Workflow

1. Import required libraries
2. Load Titanic dataset
3. Perform data cleaning
4. Handle missing values
5. Encode categorical variables
6. Scale features
7. Split data into training and testing datasets
8. Train multiple ML models
9. Generate predictions
10. Compare performance metrics

---

## Evaluation Metrics

Models were evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix
- Training Score
- Testing Score

---

## Results

### Model Performance Comparison

| Model | Accuracy Score |
|---------|--------------|
| SVC | 81.01% |
| Logistic Regression | 79.33% |
| KNN | 64.80% |
| Decision Tree Classifier | 59.22% |
| AdaBoost Classifier | 59.22% |
| Random Forest Classifier | 56.98% |

### Best Performing Model

**Support Vector Classifier (SVC)** achieved the highest accuracy of **81.01%**, making it the best-performing model for this Titanic survival prediction task.

---

## Future Enhancements

- Hyperparameter tuning
- Cross-validation
- Deep learning implementation
- Deployment using Flask or Streamlit
- Further feature engineering

---

## Author

**Vinnakota Nitish Raj**

GitHub: https://github.com/vnr-nitish

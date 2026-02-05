# 🚢 Titanic Survival Prediction

A comprehensive end-to-end machine learning pipeline to solve the classic [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) challenge. This project focuses on building a robust classification system to predict passenger survival based on demographic and trip data.

## 📈 Project Overview

The mission is to predict which passengers survived the Titanic shipwreck. This implementation emphasizes **clean code**, **automated pipelines**, and **feature engineering**.

### Key Features:
- **Exploratory Data Analysis (EDA)**: Detailed visualizations using `Seaborn` to identify survival patterns across gender, class, and age.
- **Automated Pipelines**: Uses Scikit-Learn `Pipeline` and `ColumnTransformer` to handle preprocessing (imputing, scaling, and encoding) in a single workflow.
- **Feature Engineering**: Constructed new features like `family_size` by combining `SibSp` and `Parch` to capture social dynamics.
- **Model Comparison**: Implements and compares multiple classifiers.

## 🛠️ Technology Stack

- **Modeling**: Logistic Regression, Decision Tree Classifier.
- **Preprocessing**: `SimpleImputer` (Median/Most Frequent), `StandardScaler`, `OneHotEncoder`.
- **Data Handling**: `Pandas`, `NumPy`.
- **Visualization**: `Matplotlib`, `Seaborn`.

## 🏗️ Pipeline Architecture

The project uses a modular preprocessing strategy:
1. **Numerical Pipeline**: Handles missing `Age` and `Fare` values via median imputation and normalizes data using standard scaling.
2. **Categorical Pipeline**: Fills missing `Embarked` values and converts categorical text (`Sex`, `Embarked`, `Pclass`) into numerical vectors using One-Hot Encoding.
3. **Training & Validation**: Splits the 891-row training set (80/20) to validate model performance before final testing.

## 🚀 Usage

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

1. Open `titanic-survival-classification.ipynb`.
2. Ensure the Titanic dataset files are in the expected `/kaggle/input/titanic/` directory.
3. Run the notebook to see the EDA plots and train the models.
4. The system automatically creates a `submission.csv` using the best-performing model (Decision Tree).

---
*Classic ML starter project with best-practice implementation.*

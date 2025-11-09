
# Heart Attack Risk Predictor

Live: https://heartattackpredictiondsp.streamlit.app/

Simple Streamlit demo that predicts heart-attack risk using a saved sklearn pipeline.

Files:
- app.py : Streamlit app
- model_pipeline.pkl : Trained pipeline (preprocessing + model)
- requirements.txt : Python dependencies
- heart-attack-eda-simple_prediction.ipynb : Notebook (optional)

How to run locally:
1. pip install -r requirements.txt
2. streamlit run app.py


## **About the Dataset**

The dataset used in this project is the [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), which contains clinical and health-related attributes to predict the likelihood of a heart attack.  
The target variable (`output`) indicates the presence of heart disease:
- **1** → more chance of heart attack  
- **0** → less chance of heart attack  

### **Key Features**
1. **age** — Age of the patient  
2. **sex** — Gender of the patient  
   - 0: Female  
   - 1: Male  
3. **cp** — Chest pain type  
   - 0: Typical angina  
   - 1: Atypical angina  
   - 2: Non-anginal pain  
   - 3: Asymptomatic  
4. **trtbps** — Resting blood pressure (in mm Hg)  
5. **chol** — Serum cholesterol in mg/dl  
6. **fbs** — Fasting blood sugar > 120 mg/dl  
   - 1: True  
   - 0: False  
7. **restecg** — Resting electrocardiographic results  
   - 0: Normal  
   - 1: ST-T wave abnormality  
   - 2: Left ventricular hypertrophy  
8. **thalachh** — Maximum heart rate achieved  
9. **exng** — Exercise induced angina  
   - 1: Yes  
   - 0: No  
10. **oldpeak** — ST depression induced by exercise relative to rest  
11. **slp** — Slope of the peak exercise ST segment  
    - 0: Upsloping  
    - 1: Flat  
    - 2: Downsloping  
12. **caa** — Number of major vessels (0–4) colored by fluoroscopy  
13. **thall** — Thalium stress test result  
    - 0: Normal  
    - 1: Fixed defect  
    - 2: Reversible defect  

---

## **Project Overview**

### **1. Exploratory Data Analysis (EDA)**
- **Data Inspection:** Checked dataset shape, data types, duplicates, and missing values.  
- **Outlier Detection:** Identified and handled outliers in `trtbps`, `chol`, and `oldpeak`.  
- **Statistical Summary:** Reviewed distributions of key numerical features.  
- **Correlation Analysis:** Visualized correlations between features using a heatmap to identify relationships with the target (`output`).

### **2. Visualization**
- **Distribution Analysis:** Used histograms and violin plots for age, cholesterol, and resting blood pressure.  
- **Target Distribution:** Compared the count of patients with and without heart disease.  
- **Pairwise Relationships:** Explored relationships such as `thalachh` vs `oldpeak` and their effect on the target.

### **3. Feature Engineering**
- **Derived Feature:** Created a new categorical feature `Age_CAT` (Adult / Middle_Age_Adult / Senior_Adult) using age binning.  
- **Encoding:** Applied one-hot encoding on categorical columns such as `cp`, `restecg`, `slp`, `caa`, and `thall`.  
- **Scaling:** Used `StandardScaler` on numerical features (`age`, `trtbps`, `chol`, `thalachh`, `oldpeak`).  
- **Train-Test Split:** 80% training and 20% testing with a fixed random state for reproducibility.

### **4. Modeling**
Six machine learning models were compared using **10-fold cross-validation (ROC-AUC)**:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

**Best model:** Logistic Regression  
- **Mean ROC-AUC:** 0.9064  
- **Test Accuracy:** 0.9016  
- **Test ROC-AUC:** 0.9418  

The Logistic Regression model was selected for its high accuracy, balanced precision-recall, and simplicity.

### **5. Model Evaluation**
| Metric | Value |
|--------|--------|
| **Accuracy** | 0.9016 |
| **ROC-AUC** | 0.9418 |
| **Precision (1)** | 0.93 |
| **Recall (1)** | 0.88 |
| **F1-Score (1)** | 0.90 |

**Confusion Matrix:**
```
[[27 2]
[ 4 28]]
```

Interpretation:  
- 27 true negatives and 28 true positives were correctly identified.  
- Only 6 misclassifications (2 false positives, 4 false negatives).

### **6. Deployment**
- The trained model and preprocessing steps were combined into a single **Sklearn Pipeline** using `ColumnTransformer` and `StandardScaler`.  
- The final pipeline was saved as **`model_pipeline.pkl`**.  
- A user-friendly **Streamlit web app** was built to allow interactive predictions using the model.  
- The app can be run locally or deployed to **Streamlit Cloud** for online access.

---

## **Future Work**
- Implement **hyperparameter tuning** using GridSearchCV or RandomizedSearchCV.  
- Add **SHAP feature importance visualization** for interpretability.  
- Integrate a **Flask API** backend to support mobile or web clients.  
- Expand dataset and perform advanced feature selection for even better accuracy.

---

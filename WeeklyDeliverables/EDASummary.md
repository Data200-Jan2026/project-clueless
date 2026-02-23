### **Dataset Overview**

* **Size**: The dataset contains **920 entries** and **16 columns** initially.
* **Features**: It includes demographic data (age, sex), clinical measurements (blood pressure, cholesterol, heart rate), and diagnostic results (ECG, chest pain type).
* **Target Variable (`num`)**: Originally a multi-class variable (0–4) representing heart disease severity.

---

### **Key Insights & Findings**

#### **1. Data Quality and Missingness**

* Several columns had significant missing data: **`ca` (66.4%)**, **`thal` (52.8%)**, and **`slope` (33.6%)**.
* **Decision**: These three columns were dropped because high missingness could lead to biased imputation and negatively affect model performance.
* Other features like `trestbps`, `chol`, and `thalch` had minor missingness (approx. 3–6%) and were imputed using the **median** to avoid outlier influence.

#### **2. Demographic Distribution**

* **Gender Imbalance**: The dataset is heavily skewed toward male participants, with **726 males** compared to a much smaller female cohort.
* **Age**: The average age of participants is approximately **53.5 years**, with ages ranging from 28 to 77.

#### **3. Clinical Observations**

* **Chest Pain**: The most common chest pain type recorded is **asymptomatic** (496 cases).
* **Resting ECG**: A majority of participants (**551**) showed "normal" results, though "lv hypertrophy" and "st-t abnormality" were also present.
* **Cholesterol**: The mean cholesterol is **199.13 mg/dl**, but the minimum value of 0 suggests potential data entry issues or specific patient conditions in some datasets (like Switzerland).

#### **4. Target Transformation**

* The original target `num` was imbalanced, with 44.7% non-disease cases and the remaining 55.3% spread across four severity levels.
* **Action**: To simplify the analysis and improve predictive performance, the problem was converted into a **binary classification** task:
* **0**: Absence of heart disease.
* **1**: Presence of heart disease (combining original levels 1, 2, 3, and 4).



---

### **EDA Summary Table**

| Feature | Top/Mean Value | Note |
| --- | --- | --- |
| **Sex** | Male (78.9%) | Significant gender imbalance. |
| **Dataset** | Cleveland | Most samples come from the Cleveland dataset. |
| **CP Type** | Asymptomatic | Most frequent chest pain category. |
| **Imputation** | Median/Mode | Used for numerical/categorical missing values. |
| **Target** | Presence (55.3%) | Dataset is relatively balanced for binary classification. |
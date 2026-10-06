# Classification of Imbalanced Insurance Claims Data

This project focuses on handling and modeling highly imbalanced data in machine learning, specifically applied to auto insurance claim predictions. In real-world insurance datasets, positive claims are rare compared to non-claims. This repository demonstrates end-to-end Exploratory Data Analysis (EDA), data preprocessing, and strategies to address class imbalance to build robust predictive models.

---

## 📌 Project Overview

When dealing with imbalanced datasets (e.g., fraud detection, medical diagnosis, insurance claims), standard metrics like raw **Accuracy** can be misleading. A model predicting the majority class 100% of the time may achieve high accuracy while failing entirely to identify the minority class of interest.

**Key Objectives:**
1. **Analyze Imbalance:** Inspect class distributions and feature patterns.
2. **Business Priority Alignment:** Focus on identifying rare positive claim cases while controlling false positives.
3. **Resampling Techniques:** Address class imbalance using resampling strategies (e.g., oversampling).
4. **Robust Machine Learning Models:** Utilize algorithms naturally suited for imbalanced tabular data, such as Tree-based models, Random Forest, and Gradient Boosting.
5. **Proper Evaluation:** Evaluate models using Precision, Recall, F1-Score, and AUROC rather than raw accuracy.

---

## 📊 Dataset Description

The dataset contains auto insurance policy details, customer demographics, vehicle specifications, and safety features.

- **Total Records:** 58,592 rows
- **Total Features:** 41 columns
- **Target Variable:** `claim_status` (Binary: `0` = No Claim, `1` = Claim Filed)

### Feature Categories:
- **Policy & Customer Info:** `policy_id`, `subscription_length`, `vehicle_age`, `customer_age`, `region_code`, `region_density`
- **Vehicle Specifications:** `segment`, `model`, `fuel_type`, `max_torque`, `max_power`, `engine_type`, `displacement`, `cylinder`, `transmission_type`, `steering_type`, `turning_radius`, `length`, `width`, `gross_weight`, `ncap_rating`
- **Safety & Equipment Features (Binary Yes/No):** `airbags`, `is_esc`, `is_adjustable_steering`, `is_tpms`, `is_parking_sensors`, `is_parking_camera`, `rear_brakes_type`, `is_front_fog_lights`, `is_rear_window_wiper`, `is_rear_window_washer`, `is_rear_window_defogger`, `is_brake_assist`, `is_power_door_locks`, `is_central_locking`, `is_power_steering`, `is_driver_seat_height_adjustable`, `is_day_night_rear_view_mirror`, `is_ecw`, `is_speed_alert`

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Resampling:** `scikit-learn`, `imbalanced-learn` (SMOTE / Oversampling)

---

## 📈 Methodology & Workflow

1. **Exploratory Data Analysis (EDA):**
   - Data structure examination (`df.info()`, `df.shape`, missing values check).
   - Visualization of target variable `claim_status` distribution to measure skewness.
   - Analysis of vehicle safety features vs. claim rates.

2. **Data Preprocessing & Feature Engineering:**
   - Encoding categorical variables (One-Hot Encoding / Label Encoding).
   - Numerical feature scaling where applicable.
   - Handling missing values (if any).

3. **Handling Class Imbalance:**
   - Minority class oversampling / SMOTE (Synthetic Minority Over-sampling Technique).
   - Class weight adjustment within algorithms.

4. **Model Building & Training:**
   - Baseline Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting / XGBoost Classifier

5. **Evaluation Metrics:**
   - **Confusion Matrix:** Breakdown of TP, FP, TN, FN.
   - **Precision & Recall:** Trade-off analysis between catching claims and avoiding false alarms.
   - **F1-Score:** Harmonic mean of precision and recall.
   - **ROC-AUC Score:** Overall discriminative ability of the classifier across thresholds.

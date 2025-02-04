# Dynamic Pricing Project

## 1. Business Understanding

**Main Objective:**  
Develop a dynamic pricing prediction system that estimates the price of devices (e.g., laptops and smartphones) based on their features such as RAM, storage, processor type, etc.

**Business Questions:**
- How can we utilize available data to predict a fair selling price for devices?
- What features have the most significant influence on a device’s price?
- How can we ensure the pricing system reflects market trends and dynamics effectively?

## 2. Data Understanding

**Data Source:**  
The data is collected from two CSV files:
- **laptops.csv**: Contains information about laptops.
- **smartphones.csv**: Contains information about smartphones.

**Features Included:**
- **Laptops:** Brand, Model, RAM, Storage, Storage Type, CPU, Touchscreen.
- **Smartphones:** Brand, Model, RAM, Storage, Color.

**Data Exploration:**
- Inspect the distribution of key features like RAM, storage, and price.
- Identify missing values, outliers, and inconsistencies.
- Analyze feature correlations to understand data relationships.

## 3. Data Preparation

**Cleaning:**
- Handle missing values by imputing with suitable values (e.g., median) or removing incomplete records.
- Remove duplicates and irrelevant data to maintain data quality.

**Transformation:**
- Convert categorical features (e.g., Brand, Storage Type) to numerical values using One-Hot Encoding.
- Standardize or normalize numerical features if necessary.

**Data Splitting:**
- Split the data into training and testing sets (typically 80/20) to validate model performance.

## 4. Modeling

**Model Selection:**
- Use Random Forest Regressor for its robustness in handling various data types and reducing overfitting.

**Input and Output Preparation:**
- **Inputs:** Device features like RAM, storage, processor type, etc.
- **Output:** Predicted price (Final Price).

**Training the Model:**
- Train separate models for laptops and smartphones using the prepared datasets.
- Apply cross-validation to improve generalization.

**Model Optimization:**
- Fine-tune hyperparameters (e.g., number of trees, max depth).
- Experiment with other algorithms such as Decision Trees or Gradient Boosting for comparison.

## 5. Evaluation

**Model Testing:**
- Evaluate the model's performance using unseen test data.

**Metrics Used:**
- **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions.
- **R² Score:** Indicates how well the model explains the variance in the data.

**Results:**
- Analyze performance metrics to determine model accuracy.
- If performance is unsatisfactory, revisit data cleaning, feature engineering, or model tuning.

## 6. Deployment

**User Interface Development:**
- Build a simple web interface using Flask to interact with the pricing model.

**Workflow:**
1. Users select the device type (laptop or smartphone).
2. Users input device specifications (e.g., RAM, storage, CPU).
3. The system processes the input and predicts the price.
4. The predicted price is displayed along with performance metrics (MAE, R²).

**Final Outputs:**
- **Interactive Web Interface:** Allows users to input device features and receive a price prediction.
- **Dynamic Pricing Model:** Provides market-aligned prices based on device characteristics.

**Future Steps:**
- Enhance model accuracy with more data and advanced algorithms.
- Integrate the system into e-commerce platforms for real-time dynamic pricing.
- Incorporate real-time market data to adjust prices dynamically.

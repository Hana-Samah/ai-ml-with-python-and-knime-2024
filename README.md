# Dynamic Pricing Project

## 1. Business Understanding
### Main Objective:
Develop a pricing prediction system that estimates the price of devices (e.g., laptops and smartphones) based on their features (RAM, storage, processor type, etc.).

### Business Questions:
- How can we use available data to predict a fair selling price for devices?
- What features have the most influence on a device’s price?
- How can we ensure the system reflects market trends and dynamics?

---

## 2. Data Understanding
### Data Source:
Two CSV files (`laptops.csv` and `smartphones.csv`) containing device details such as:

#### Laptops:
- Brand
- Model
- RAM
- Storage
- Storage Type
- Processor
- Touchscreen

#### Smartphones:
- Brand
- Model
- RAM
- Storage
- Color

### Data Exploration:
- Inspect the distribution of features like RAM, storage, and price.
- Ensure the data quality by checking for missing values or any anomalies.

---

## 3. Data Preparation
### Cleaning:
- Handle missing values by either imputing them with suitable values (e.g., median) or removing incomplete records.

### Transformation:
- Convert categorical features (like processor type or storage type) into numerical format using techniques like One-Hot Encoding.

### Data Splitting:
- Split the data into training and testing sets for model validation.

---

## 4. Modeling
### Model Selection:
- Train a Random Forest Regressor model to predict device prices.

### Input and Output Preparation:
- **Inputs:** Device features like RAM, storage, processor type, etc.
- **Output:** Predicted price (Final Price).

### Training the Model:
- Train separate models for laptops and smartphones using the prepared datasets.

### Model Optimization:
- Fine-tune models, possibly experimenting with additional algorithms like Decision Trees or boosting methods.

---

## 5. Evaluation
### Model Testing:
- Assess the model's performance using the test dataset.

### Metrics Used:
- **Mean Absolute Error (MAE):** To evaluate the average difference between predicted and actual prices.
- **R² Score:** To measure the model’s ability to explain the variance in data.

### Results:
- If the model’s accuracy is satisfactory, it will be used for price prediction. If not, improvements will be made to the data quality or model performance.

---

## 6. Deployment
### User Interface Development:
Build a web interface using Flask:
- **Input Form:** Allows users to select device type and enter specifications.
- **Price Result:** Displays the predicted price based on the input features.

### Workflow:
1. Users select a device type (laptop or smartphone) and input its specifications.
2. The system sends the data to the trained model for prediction.
3. The model outputs the predicted price, which is displayed on the screen.

### Future Steps:
- Improve model accuracy by incorporating more data and advanced techniques.
- Extend the system to integrate into an e-commerce platform or dynamic pricing system.

---

## Final Outputs:
- **Interactive Web Interface:** Allows users to enter specifications and receive a predicted price.
- **Dynamic Pricing Model:** Delivers market-aligned prices based on device features.

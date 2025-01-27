# Dynamic Pricing Project

## 1. Business Understanding
### Main Objective:
Develop a Dynamic Pricing system that determines appropriate prices for products (e.g., laptops or smartphones) based on specific features (RAM, storage, processor type, etc.).

### Business Questions:
- How can we use existing data to predict a fair selling price?
- What features influence the product's price?
- How can the system reflect market dynamics?

---

## 2. Data Understanding
### Data Source:
Two CSV files (`laptops.csv` and `smartphones.csv`) containing details such as:

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
- Examine the distribution of features (RAM, storage, price).
- Ensure data quality (no missing or incorrect data).

---

## 3. Data Preparation
### Cleaning:
- Handle missing values using appropriate methods (e.g., imputing with median values or dropping incomplete records).

### Transformation:
- Convert textual features (e.g., processor or storage type) into numerical values using techniques like One-Hot Encoding.

### Data Splitting:
- Split data into training and testing sets.

---

## 4. Modeling
### Model Selection:
- Use Linear Regression to predict prices.

### Input and Output Preparation:
- **Inputs:** Features like RAM, storage, processor type, etc.
- **Output:** Expected price (Final Price).

### Training the Model:
- Train separate models for laptops and smartphones.

### Model Optimization:
- Experiment with more complex models if necessary (e.g., Decision Trees or Random Forest).

---

## 5. Evaluation
### Model Testing:
- Assess model accuracy using the test data.

### Metrics Used:
- **Mean Absolute Error (MAE):** To measure the average difference between predicted and actual prices.
- **R² Score:** To evaluate how well the model explains data variance.

### Results:
- Adopt the model if accuracy is acceptable. Improve data quality or model performance if results are poor.

---

## 6. Deployment
### User Interface Development:
Build a web interface using Flask:
- **Input Form:** Allows users to select the device type and enter specifications.
- **Price Result:** Displays the predicted price based on the input.

### Workflow:
1. Users select the device type and enter specifications.
2. Data is sent to the trained model.
3. The system displays the predicted price on the screen.

### Future Steps:
- Enhance the model with more data or advanced models.
- Integrate the project into a larger system, such as an e-commerce platform.

---

## Final Outputs:
- **Interactive Web Interface:** Allows users to input device specifications and get a predicted price.
- **Dynamic Pricing Model:** Processes data and provides realistic, market-aligned prices.


# Dynamic Pricing Project

This project demonstrates a **Dynamic Pricing System** for electronic devices using **Machine Learning** and a **Flask Web Application**. It uses historical data to predict an optimal price based on demand, competition prices, and product features.

---

## **CRISP-DM Methodology**

### **1. Business Understanding**

#### Objective:
To build a pricing model that predicts the price of electronic devices dynamically based on:
- **Demand**: Number of product purchases.
- **Competition Prices**: Average prices of similar products in the market.
- **Features**: Product brand, specifications, and type.

#### Key Question:
How can the system determine an optimal price reflecting market factors?

#### Expected Outcome:
A model that predicts the price dynamically based on user input.

---

### **2. Data Understanding**

#### Required Data:
- **Demand**: Sales count.
- **Competition Prices**: Prices of competing products.
- **Product Features**: Brand, storage capacity, type (e.g., smartphone, laptop).
- **Target Price**: The final price (used as the target variable).

#### Data Source:
Datasets such as [Amazon Electronics Data](https://www.kaggle.com/) can be used.

---

### **3. Data Preparation**

#### Steps:
1. **Data Cleaning**: Remove null values and inconsistent data.
2. **Feature Encoding**: Convert categorical data (e.g., brand) into numeric values using Label Encoding.
3. **Feature Scaling**: Standardize the input features for better model performance.
4. **Data Splitting**: Split the data into training and testing sets.

---

### **4. Modeling**

#### Model:
- **Linear Regression**: A regression algorithm suitable for predicting continuous values.

#### Steps:
1. Train the model using the training dataset.
2. Evaluate model performance using the testing dataset.

---

### **5. Evaluation**

#### Metrics:
- **Mean Absolute Error (MAE)**: To measure prediction accuracy.
- **R-squared**: To measure the proportion of variance explained by the model.

---

### **6. Deployment**

The final model is deployed as a web application using **Flask**, allowing users to input data and get real-time price predictions.

---

## **Project Setup**

### **1. Environment Setup**

#### Install Anaconda:
Download and install Anaconda from [here](https://www.anaconda.com/).

#### Install Required Libraries:
```bash
pip install flask scikit-learn pandas joblib
```

---

### **2. Data Preparation**

#### Sample Code to Prepare Data:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load data
data = pd.read_csv("electronics_pricing.csv")

# Clean data
data = data.dropna()

# Encode categorical features
label_encoder = LabelEncoder()
data['brand_encoded'] = label_encoder.fit_transform(data['brand'])

# Prepare features and target
X = data[['demand', 'competition_price', 'brand_encoded']]
y = data['price']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

---

### **3. Flask Application**

#### Flask Code (`app.py`):

```python
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("price_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    demand = float(request.form["demand"])
    competition_price = float(request.form["competition_price"])
    brand_encoded = int(request.form["brand_encoded"])

    # Prepare input
    input_data = scaler.transform([[demand, competition_price, brand_encoded]])
    price = model.predict(input_data)

    return render_template("result.html", price=round(price[0], 2))

if __name__ == "__main__":
    app.run(debug=True)
```

---

### **4. HTML Templates**

#### `templates/index.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Dynamic Pricing</title>
</head>
<body>
    <h1>Dynamic Pricing Prediction</h1>
    <form action="/predict" method="POST">
        <label>Demand:</label>
        <input type="number" name="demand" required><br>
        <label>Competition Price:</label>
        <input type="number" name="competition_price" required><br>
        <label>Brand:</label>
        <select name="brand_encoded">
            <option value="0">Brand A</option>
            <option value="1">Brand B</option>
        </select><br>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
```

#### `templates/result.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h1>Predicted Price: ${{ price }}</h1>
    <a href="/">Back</a>
</body>
</html>
```

---

### **5. Run the Application**

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

---

## **Resources**

1. **Python Basics**:
   - [W3Schools Python](https://www.w3schools.com/python/)

2. **Flask Documentation**:
   - [Flask Docs](https://flask.palletsprojects.com/)

3. **Machine Learning**:
   - [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)

Feel free to reach out if you need further assistance!

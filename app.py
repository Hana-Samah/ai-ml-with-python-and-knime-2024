from flask import Flask, render_template, request
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# تحميل البيانات
laptops = pd.read_csv("data/laptops.csv")
smartphones = pd.read_csv("data/smartphones.csv")

# معالجة البيانات
def preprocess_data(data, features):
    data = data.dropna()
    X = data[features]
    y = data["Final Price"]
    return X, y

# تدريب النموذج
def train_model(data, features):
    X, y = preprocess_data(data, features)
    X = pd.get_dummies(X)
    model = LinearRegression()
    model.fit(X, y)
    return model

# مميزات اللابتوبات والجوالات
laptop_features = ["Brand", "Model", "RAM", "Storage", "Storage type", "CPU", "Touch"]
smartphone_features = ["Brand", "Model", "RAM", "Storage", "Color"]

# تدريب النماذج
laptop_model = train_model(laptops, laptop_features)
smartphone_model = train_model(smartphones, smartphone_features)

# معدل تحويل العملة
conversion_rate =34.02  # مثال لمعدل التحويل إلى البات التايلاندي

# المسار الرئيسي
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/laptop", methods=["GET", "POST"])
def laptop():
    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        user_data = {
            "Brand": request.form["brand"],
            "Model": request.form["model"],
            "RAM": int(request.form["ram"]),
            "Storage": int(request.form["storage"]),
            "Storage type": request.form["storage_type"],
            "CPU": request.form["cpu"],
            "Touch": request.form["touch"],
        }
        user_df = pd.DataFrame([user_data])
        user_df = pd.get_dummies(user_df).reindex(columns=laptop_model.feature_names_in_, fill_value=0)
        predicted_price = laptop_model.predict(user_df)[0]
        price_thb = round(predicted_price * conversion_rate, 2)
        reason = "Based on brand, RAM, storage, and processor type."
        return render_template("result.html", 
                       price=round(predicted_price, 2), 
                       price_thb=round(price_thb, 2), 
                       reason=reason, 
                       image_url=file_path)
    return render_template("laptops_form.html")

@app.route("/smartphone", methods=["GET", "POST"])
def smartphone():
    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        user_data = {
            "Brand": request.form["brand"],
            "Model": request.form["model"],
            "RAM": int(request.form["ram"]),
            "Storage": int(request.form["storage"]),
            "Color": request.form["color"],
        }
        user_df = pd.DataFrame([user_data])
        user_df = pd.get_dummies(user_df).reindex(columns=smartphone_model.feature_names_in_, fill_value=0)
        predicted_price = smartphone_model.predict(user_df)[0]
        price_thb = round(predicted_price * conversion_rate, 2)
        reason = "Based on brand, RAM, storage, and color."
        return render_template("result.html", 
                       price=round(predicted_price, 2), 
                       price_thb=round(price_thb, 2), 
                       reason=reason, 
                       image_url=file_path)
    return render_template("smartphones_form.html")

if __name__ == "__main__":
    app.run(debug=True)

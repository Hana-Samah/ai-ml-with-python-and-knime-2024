from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# التأكد من وجود المجلدات لحفظ الملفات والصور
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# 1. فهم الأعمال (Business Understanding)
# الهدف: بناء نموذج لتوقع أسعار اللابتوبات والهواتف الذكية بناءً على مواصفاتهاbased on their specifications.

# 2. فهم البيانات (Data Understanding)
laptops = pd.read_csv("data/laptops.csv")
smartphones = pd.read_csv("data/smartphones.csv")

# 3. تحضير البيانات (Data Preparation)
def preprocess_data(data, features):
    data = data.dropna()
    X = pd.get_dummies(data[features])  # تحويل البيانات النصية إلى رقمية
    y = data["Final Price"]
    return X, y

# 4. نمذجة البيانات (Modeling)
def train_model(data, features, filename):
    X, y = preprocess_data(data, features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. تقييم النموذج (Evaluation)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plot_filename = generate_price_plot(y_test, y_pred, filename)

    return model, mae, r2, plot_filename

# توليد الرسم البياني وحفظه (لتحليل الأداء)
def generate_price_plot(actual_prices, predicted_prices, filename):
    plt.figure(figsize=(8, 5))
    plt.scatter(actual_prices, predicted_prices, color='blue', label="Predicted Prices")
    plt.plot(actual_prices, actual_prices, color='red', linestyle='--', label="Actual Prices (Ideal)")

    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.legend()

    plot_path = os.path.join("static/images", filename)
    plt.savefig(plot_path)
    plt.close()

    return filename

# مميزات اللابتوبات والجوالات
laptop_features = ["Brand", "Model", "RAM", "Storage", "Storage type", "CPU", "Touch"]
smartphone_features = ["Brand", "Model", "RAM", "Storage", "Color"]

# تدريب النماذج
laptop_model, laptop_mae, laptop_r2, laptop_plot = train_model(laptops, laptop_features, "laptop_prices.png")
smartphone_model, smartphone_mae, smartphone_r2, smartphone_plot = train_model(smartphones, smartphone_features, "smartphone_prices.png")

# 6. النشر (Deployment)
conversion_rate = 34.02  # معدل التحويل إلى البات التايلاندي

@app.route("/")
def home():
    return render_template("index.html", 
                           laptop_plot=laptop_plot, 
                           smartphone_plot=smartphone_plot)

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

        return render_template("result.html", 
                               price=round(predicted_price, 2), 
                               price_thb=round(price_thb, 2), 
                               mae=laptop_mae,
                               r2=laptop_r2,
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

        return render_template("result.html", 
                               price=round(predicted_price, 2), 
                               price_thb=round(price_thb, 2), 
                               mae=smartphone_mae,
                               r2=smartphone_r2,
                               image_url=file_path)

    return render_template("smartphones_form.html")

if __name__ == "__main__":
    app.run(debug=True)

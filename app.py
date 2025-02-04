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
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. فهم العمل: تحديد الهدف من المشروع (التنبؤ بالسعر بناءً على مميزات الأجهزة)
goal = "التنبؤ بأسعار الأجهزة (اللاب توب والهواتف الذكية) بناءً على مميزاتها."

# 2. فهم البيانات: تحميل البيانات و استكشافها
laptops = pd.read_csv("data/laptops.csv")
smartphones = pd.read_csv("data/smartphones.csv")

# استكشاف البيانات: عرض بعض التفاصيل
laptop_info = laptops.describe()
smartphone_info = smartphones.describe()

# 3. تحضير البيانات: معالجة البيانات وإزالة القيم المفقودة
def preprocess_data(data, features):
    data = data.dropna()
    X = pd.get_dummies(data[features])
    y = data["Final Price"]
    return X, y

# 4. النمذجة: تدريب النماذج
def train_model(data, features, filename):
    X, y = preprocess_data(data, features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plot_filename = generate_price_plot(y_test, y_pred, filename)

    return model, mae, r2, plot_filename

# توليد الرسم البياني لعرض التنبؤات مقارنة بالقيم الفعلية
def generate_price_plot(actual_prices, predicted_prices, filename):
    plt.figure(figsize=(8, 5))
    plt.scatter(actual_prices, predicted_prices, color='blue', label="Predicted Prices")
    plt.plot(actual_prices, actual_prices, color='red', linestyle='--', label="Actual Prices (Ideal)")

    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Actual vs Predicted Prices - {filename.split('.')[0]}")
    plt.legend()
    
    plot_path = os.path.join("static/images", filename)
    plt.savefig(plot_path)
    plt.close()
    return filename

# 5. التقييم: تحليل أداء النموذج باستخدام MAE و R2
laptop_features = ["Brand", "Model", "RAM", "Storage", "Storage type", "CPU", "Touch"]
smartphone_features = ["Brand", "Model", "RAM", "Storage", "Color"]

laptop_model, laptop_mae, laptop_r2, laptop_plot = train_model(laptops, laptop_features, "laptop_price_comparison.png")
smartphone_model, smartphone_mae, smartphone_r2, smartphone_plot = train_model(smartphones, smartphone_features, "smartphone_price_comparison.png")

# 6. النشر: إعداد التطبيق وعرض النتائج للمستخدم
@app.route("/")
def home():
    return render_template("index.html", 
                           laptop_plot=laptop_plot, 
                           smartphone_plot=smartphone_plot,
                           laptop_mae=laptop_mae, 
                           laptop_r2=laptop_r2,
                           smartphone_mae=smartphone_mae, 
                           smartphone_r2=smartphone_r2)

if __name__ == "__main__":
    app.run(debug=True)

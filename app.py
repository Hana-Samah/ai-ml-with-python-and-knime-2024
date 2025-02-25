from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# إنشاء تطبيق Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# التأكد من وجود المجلدات لحفظ الملفات والصور
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# تحميل البيانات - البيانات هي جزء أساسي لفهم المشكلة (Data Understanding)
laptops = pd.read_csv("data/laptops.csv")
smartphones = pd.read_csv("data/smartphones.csv")

# مرحلة تحضير البيانات (Data Preparation) 
def preprocess_data(data, features):
    data = data.copy()
    for col in features:
        if data[col].dtype == "object":  
            # تحويل الأعمدة النصية إلى صيغة صغيرة وإزالة الفراغات
            data[col] = data[col].astype(str).str.lower().str.strip().fillna("unknown")
    
    # تحويل الأعمدة الرقمية مثل RAM, Storage, Final Price إلى أرقام
    numeric_columns = ["RAM", "Storage", "Final Price"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col].fillna(data[col].median(), inplace=True)
    
    # حذف القيم المفقودة في "Final Price"
    data = data.dropna(subset=["Final Price"])
    
    # تحويل البيانات إلى تنسيق مناسب للنموذج
    X = pd.get_dummies(data[features], drop_first=True)
    y = data["Final Price"]
    return X, y

# مرحلة النموذج (Modeling)
def train_model(data, features, model_type, filename):
    X, y = preprocess_data(data, features)
    # تقسيم البيانات إلى تدريب واختبار (Training and Test sets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # اختيار نوع النموذج
    if model_type == "linear":
        model = LinearRegression()  # الانحدار الخطي
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # الغابات العشوائية
    
    # تدريب النموذج
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # تقييم النموذج
    mae = mean_absolute_error(y_test, y_pred)  # حساب الخطأ المطلق
    r2 = r2_score(y_test, y_pred)  # حساب R2
    
    # توليد الرسم البياني للأسعار الفعلية مقابل المتوقعة
    plot_filename = generate_price_plot(y_test, y_pred, filename)
    return model, mae, r2, plot_filename

# توليد رسم بياني للمقارنة بين الأسعار الفعلية والمتوقعة
def generate_price_plot(actual_prices, predicted_prices, filename):
    plt.figure(figsize=(8, 5))
    plt.scatter(actual_prices, predicted_prices, color='blue', label="Predicted Prices")
    plt.plot(actual_prices, actual_prices, color='red', linestyle='--', label="Actual Prices (Ideal)")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    
    plot_path = os.path.join("static/images", filename)
    plt.savefig(plot_path)  # حفظ الرسم البياني
    plt.close()
    return filename

# مميزات اللابتوبات والجوالات
laptop_features = ["Brand", "Model", "RAM", "Storage", "Storage type", "CPU", "Touch"]
smartphone_features = ["Brand", "Model", "RAM", "Storage", "Color"]

# تدريب النماذج باستخدام "Linear Regression" للابتوب و"Random Forest" للجوال
laptop_model, laptop_mae, laptop_r2, laptop_plot = train_model(laptops, laptop_features, "linear", "laptop_prices.png")
smartphone_model, smartphone_mae, smartphone_r2, smartphone_plot = train_model(smartphones, smartphone_features, "random_forest", "smartphone_prices.png")

# تحويل العملة (تقدير الأسعار بالبات التايلاندي)
conversion_rate = 34.02

# صفحة المنزل - تعرض الرسوم البيانية للنماذج
@app.route("/")
def home():
    return render_template("index.html", laptop_plot=laptop_plot, smartphone_plot=smartphone_plot)

# صفحة اللابتوبات - إدخال بيانات المستخدم للتنبؤ بالسعر
@app.route("/laptop", methods=["GET", "POST"])
def laptop():
    if request.method == "POST":
        # معالجة الصورة المحملة
        file = request.files["image"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # بيانات المستخدم المدخلة
        user_data = {
            "Brand": request.form["brand"],
            "Model": request.form["model"],
            "RAM": int(request.form["ram"]),
            "Storage": int(request.form["storage"]),
            "Storage type": request.form["storage_type"],
            "CPU": request.form["cpu"],
            "Touch": request.form["touch"],
        }

        # تحويل بيانات المستخدم إلى صيغة مناسبة للنموذج
        user_df = pd.DataFrame([user_data])
        user_df = pd.get_dummies(user_df).reindex(columns=laptop_model.feature_names_in_, fill_value=0)
        
        # التنبؤ بالسعر
        predicted_price = laptop_model.predict(user_df)[0]
        price_thb = round(predicted_price * conversion_rate, 2)

        # عرض النتيجة
        return render_template("result.html", price=round(predicted_price, 2), price_thb=price_thb, mae=laptop_mae, r2=laptop_r2, image_url=file_path)
    return render_template("laptops_form.html")

# صفحة الجوالات - إدخال بيانات المستخدم للتنبؤ بالسعر
@app.route("/smartphone", methods=["GET", "POST"])
def smartphone():
    if request.method == "POST":
        # معالجة الصورة المحملة
        file = request.files["image"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # بيانات المستخدم المدخلة
        user_data = {
            "Brand": request.form["brand"],
            "Model": request.form["model"],
            "RAM": int(request.form["ram"]),
            "Storage": int(request.form["storage"]),
            "Color": request.form["color"],
        }

        # تحويل بيانات المستخدم إلى صيغة مناسبة للنموذج
        user_df = pd.DataFrame([user_data])
        user_df = pd.get_dummies(user_df).reindex(columns=smartphone_model.feature_names_in_, fill_value=0)
        
        # التنبؤ بالسعر
        predicted_price = smartphone_model.predict(user_df)[0]
        price_thb = round(predicted_price * conversion_rate, 2)

        # عرض النتيجة
        return render_template("result.html", price=round(predicted_price, 2), price_thb=price_thb, mae=smartphone_mae, r2=smartphone_r2, image_url=file_path)
    return render_template("smartphones_form.html")

# تشغيل التطبيق
if __name__ == "__main__":
    app.run(debug=True)

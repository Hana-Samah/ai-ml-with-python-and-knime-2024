from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# تحميل البيانات
laptops = pd.read_csv("data/laptops.csv")
smartphones = pd.read_csv("data/smartphones.csv")

# معالجة البيانات
def preprocess_data(data, features):
    data = data.dropna()  # إزالة القيم المفقودة
    X = data[features]
    y = data["Final Price"]
    return X, y

# تدريب النموذج
def train_model(data, features):
    X, y = preprocess_data(data, features)
    X = pd.get_dummies(X)  # تحويل القيم النصية إلى أرقام
    model = LinearRegression()
    model.fit(X, y)
    return model

# مميزات اللابتوبات والجوالات
laptop_features = ["Brand", "Model", "RAM", "Storage", "Storage type", "CPU", "Touch"]
smartphone_features = ["Brand", "Model", "RAM", "Storage", "Color"]

# تدريب النماذج
laptop_model = train_model(laptops, laptop_features)
smartphone_model = train_model(smartphones, smartphone_features)

# المسار الرئيسي
@app.route("/")
def home():
    return render_template("index.html")

# صفحة اختيار اللابتوب
@app.route("/laptop", methods=["GET", "POST"])
def laptop():
    if request.method == "POST":
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
        return render_template("result.html", price=predicted_price)
    return render_template("laptops_form.html")

# صفحة اختيار الجوال
@app.route("/smartphone", methods=["GET", "POST"])
def smartphone():
    if request.method == "POST":
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
        return render_template("result.html", price=predicted_price)
    return render_template("smartphones_form.html")

if __name__ == "__main__":
    app.run(debug=True)

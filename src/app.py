import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))

model = joblib.load(os.path.join(basedir, "../models/LogisticRegression_0.1_l2_newton-cg_42.sav"))
scaler = joblib.load(os.path.join(basedir, "../models/scaler.save"))
class_dict = {
    "0": "no contrata deposito",
    "1": "contrata deposito"
}

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":

        val1 = float(request.form["poutcome"])
        val2 = float(request.form["emp.var.rate"])
        val3 = float(request.form["cons.price.idx"])
        val4 = float(request.form["euribor3m"])
        val5 = float(request.form["nr.employed"])

        data = [[val1, val2, val3, val4, val5]]
        scaled_data = scaler.transform(data)

        prediction = str(model.predict(scaled_data)[0])
        pred_class = class_dict[prediction]

    else:
        pred_class = None

    return render_template("index.html", prediction = pred_class)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

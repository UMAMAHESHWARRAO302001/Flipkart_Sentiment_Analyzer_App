from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

###########################
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/prediction", methods=["get","post"])
def prediction():
    input_text = str(request.form.get("review"))
    
    data_point = [input_text]

    model = joblib.load("best_rf_model.pkl")

    prediction = model.predict(data_point)

    return render_template("output.html", prediction=prediction[0])

#############################

if __name__ =="__main__":
    app.run(debug=True, host="0.0.0.0")
from flask import Flask, render_template, request
import pickle
import sklearn

import joblib

app = Flask(__name__)

model = pickle.load(open("placement.pkl", 'rb'))

ct = joblib.load('placement.pkl')


@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/form')
def form():
    return render_template("index1.html")


@app.route('/y_predict', methods=["POST"])
def y_predict():
    sen1 = int(request.form["sen1"])
    sen2 = int(request.form["sen2"])
    sen3 = int(request.form["sen3"])
    sen4 = int(request.form["sen4"])
    sen5 = int(request.form["sen5"])
    sen6 = int(request.form["sen6"])

    X_test = [[sen1, sen2, sen3, sen4, sen5, sen6]]
    prediction = model.predict(X_test)
    prediction = prediction[0]

    return render_template("secondpage.html", y=prediction)

@app.route('/')
def hellos():
    return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)

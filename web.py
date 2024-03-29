from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
import pickle

dct = pickle.load(open('dct.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        iris_array = [[sepal_length, sepal_width, petal_length,petal_width]]
        prediction = dct.predict(iris_array)
        return render_template("index.html", prediction = prediction[0].capitalize())
    else:
        return render_template("index.html")



if __name__ == '__main__':
    app.run()

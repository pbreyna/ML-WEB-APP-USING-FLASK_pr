from flask import Flask, request, render_template
from pickle import load
import numpy as np
import joblib as joblib
import os 

model=joblib.load('iris_model_LR.pkl')

app=Flask(__name__)
IMG_FOLDER=os.path.join('static','IMG')
app.config['UPLOAD_FOLDER']=IMG_FOLDER
@app.route('/')
def index():
    return render_template('index.html')
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica",
    "Iris-versicolor": "Iris versicolor",
    "Iris-virginica": "Iris virginica"  
}

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        val1 = float(request.form['val1'])
        val2 = float(request.form['val2'])
        val3 = float(request.form['val3'])
        val4 = float(request.form['val4'])
        data = [[val1, val2, val3, val4]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    return render_template('index.html', prediction=pred_class)
if __name__ == '__main__':
    app.run(debug=True)

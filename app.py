from flask import Flask,render_template,request
from gevent.pywsgi import WSGIServer
import pickle
import numpy as np
import sklearn

model=pickle.load(open('finall.pkl','rb'))

app=Flask(__name__)



@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    arr = np.array([[data1,data2,data3,data4,data5,data6]],dtype=np.int32)
    pred = model.predict(arr)
    return render_template('after.html', data=pred)



if __name__ == "__main__":
    app.run(debug=True)        
    
## Importing relevant libraries....
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index_main.html')

##----------------------------------------------------------------------------
                             ## STROKE ##
##----------------------------------------------------------------------------
@app.route("/stroke")
def stroke():
    return render_template("stroke.html")

@app.route('/predict_stroke',methods=['POST'])
def predict_stroke():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]    
    final_features = [np.array(int_features)]
    
    ## Un-pickling scaler file....
    scaler = pickle.load(open("scaler_stroke.pkl" , "rb"))
        
    ## Un-pickling scaler file....
    logreg = pickle.load(open("stroke.pkl" , "rb"))
    
    temp = scaler.transform(final_features)
    prediction = logreg.predict(temp)

    if prediction == 1:
            return render_template('stroke.html', prediction_text='Oops SORRY! You have stroke!')
    else:
        return render_template('stroke.html', prediction_text='Great News! You are healthy!')

##----------------------------------------------------------------------------
##----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)    
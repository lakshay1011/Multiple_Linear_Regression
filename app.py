import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)
model = pickle.load(open('multi_linearreg_house.pkl','rb')) 
#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    sqft = float(request.args.get('sqft'))
    bedrooms=float(request.args.get('bedrooms'))
    bathrooms=float(request.args.get('bathrooms'))
    offers=float(request.args.get('offers'))
    Brick=float(request.args.get('Brick'))
    Neighbourhood=float(request.args.get('Neighbourhood'))

    
    prediction = model.predict([[sqft,bedrooms,bathrooms,offers,Brick,Neighbourhood]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted price : {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
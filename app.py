from flask import Flask, request, jsonify
import joblib
import numpy as np
# Initialize Flask application
app = Flask(__name__)

rfc_model = joblib.load('group63_mlops_M3_optuna_model.joblib')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = rfc_model.predict(features)
  
    if int(prediction[0]) == 0 :
           prediction='Setosa'
    elif int(prediction[0]) == 1 :
           prediction='versicolor'
    elif int(prediction[0]) == 2 :
           prediction='virginica'
    else:
           prediction='na'
    
    response = {
        'prediction':  prediction
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
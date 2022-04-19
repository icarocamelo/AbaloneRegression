import json

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)

# Load model from file as soon as the class is loaded
model = joblib.load("./saved_model.sav")


@app.route('/predict', methods=['POST'])
def forecast():
    # Get payload and transform to dict
    payload = json.loads(request.data)
    
    # Assembling dict with payload
    data = {'length': payload['length'],
            'diameter': payload['diameter'],
            'height': payload['height'],
            'whole_weight': payload['whole_weight'],
            'shucked_weight': payload['shucked_weight'],
            'viscera_weight': payload['viscera_weight'],
            'shell_weight': payload['shell_weight'],
            'rings':payload['rings']}

    # Transforming dict to Panda DataFrame
    df = pd.DataFrame.from_dict(data)
    n_rings = model.predict(df)

    # Initialize [age] with invalid value
    age = -1

    # Check if prediction succeeded and is greater than 0
    # Otherwise, return -1
    if n_rings > 0:
        age = n_rings * 1.5
    
    return age


if __name__ == '__main__':
    app.run()
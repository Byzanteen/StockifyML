import numpy as np
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from keras.models import load_model
model = load_model('stockify.h5')
model._make_predict_function()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():
  
    test_json = request.get_json()

    dataset = pd.DataFrame(test_json)
    
    dataset = dataset.transpose()
    
    dataset.rename(columns={'1. open':'open','2. high':'high','3. low':'low','4. close':'close','5. volume':'volume'}, inplace=True)
    
    dataset['timestamp']=dataset.index
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'],format='%Y-%m-%d')
    dataset.index = dataset['timestamp']
    
    dataset=dataset.iloc[:60,:]

    data = dataset.sort_index(ascending=True, axis=0)

    new_data = pd.DataFrame(index=range(0,len(dataset)),columns=['timestamp', 'close'])

    for i in range(0,len(data)):
        new_data['timestamp'][i] = data['timestamp'][i]
        new_data['close'][i] = data['close'][i]
    
    new_data.index = new_data.timestamp

    new_data.drop('timestamp', axis=1, inplace=True)

    inputs = new_data.values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.fit_transform(inputs)
    print(inputs.shape)
    inputs = np.reshape(inputs, (1,inputs.shape[0],inputs.shape[1]))
    print(inputs.shape)
    close = model.predict(inputs)

    print(close)

    return "OK";

if __name__ == '__main__':
    app.run(port=5000, debug=True)
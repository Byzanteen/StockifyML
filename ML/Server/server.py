import numpy as np
import pandas as pd
import datetime as dt
from sklearn.externals import joblib
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
scaler = joblib.load("scaler.save") 
from keras.models import load_model
model = load_model('stockify.h5')
model._make_predict_function()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():
  
    test_json = request.get_json()
    
    data_points = 15

    dataset = pd.DataFrame(test_json)
    
    dataset = dataset.transpose()
    
    dataset.rename(columns={'1. open':'open','2. high':'high','3. low':'low','4. close':'close','5. volume':'volume'}, inplace=True)
    
    dataset['timestamp']=dataset.index
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'],format='%Y-%m-%d')
    dataset.index = dataset['timestamp']

    finalTime=dataset['timestamp'][0]
    day = dt.timedelta(days=1)

    dates = []
    for i in range(data_points):
        while True:
            finalTime = finalTime+day
            auxDay = finalTime.strftime('%w')
            if auxDay != '0' and auxDay != '6':
                dates=np.append(dates,finalTime.strftime('%Y-%m-%d'))
                break
    
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
    inputs  = scaler.transform(inputs)
    inputs = np.reshape(inputs, (1,inputs.shape[0],inputs.shape[1]))

    results = []
    for i in range(data_points):
        close = model.predict(inputs)
        results=np.append(results,close[0])
        auxArray=np.append(inputs[0],close[0])
        auxArray=np.delete(auxArray,0)
        auxArray=auxArray.reshape(-1,1)
        inputs[0]=auxArray

    results=results.reshape(-1,1)
    results=scaler.inverse_transform(results)
    results=results.reshape(-1)

    for i in range(data_points):
        results[i]=format(results[i], '.2f')

    dic = {}
    dic['values'] = results.tolist()
    dic['dates'] = dates.tolist()

    response = jsonify(dic)
    response.status_code = 200

    return (response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
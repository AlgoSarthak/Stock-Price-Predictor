from flask import Flask,request,render_template


from src.stockdata import StockData
from src.data_processing import DataProcessing
from src.models.gru import GRU_Model
from src.models.lstm import LSTM_Model
from src.models.simple_rnn import RNN_Model
from src.models.bidirectional_rnn import BidirectionalRNN_Model
from src.models.encoder import Encoder_Model
from src.get_graphs import get_graphs

from PIL import Image
import base64
import io

import numpy as np
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        stockName = request.form.get('stockName')
        print(stockName)
        numDays = int(request.form.get('numOfDays'))
        print(numDays)
        model_type = request.form.get('model_type')
        print(model_type)

        stockData = StockData()
        df = stockData.getStockData(tickerSymbol=stockName,time='10y')
        last_price=df["Close"][-1]
        graph_loc = stockData.getCloseGraph()

        im = Image.open(graph_loc)
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())

        dataProcessing = DataProcessing()
        df = dataProcessing.feature_selection(df)
        data = dataProcessing.transform(df)
        input_sequences, targets = dataProcessing.get_sequential_data(numDays)
        train_x,train_y,test_x,test_y = dataProcessing.get_train_test_data(input_sequences,targets)
        prediction_data = dataProcessing.get_prediction_data()


        if model_type == 'GRU':
            model = GRU_Model(numDays)
        elif model_type == 'LSTM':
            model = LSTM_Model(numDays)
        elif model_type == 'RNN':
            model = RNN_Model(numDays)
        elif model_type == 'Bidirectional-RNN':
            model = BidirectionalRNN_Model(numDays)
        elif model_type == 'Encoder-model':
            model = Encoder_Model(numDays)

        model.train(train_x, train_y)
        pred = model.Predict(test_x)

        pred_copy = pred.copy()
        test_y_copy = test_y.copy()
        for i in range(numDays):
            pred_copy[:,i,:]=dataProcessing.inv_transform(pred_copy[:,i,:])
            test_y_copy[:,i,:]=dataProcessing.inv_transform(test_y_copy[:,i,:])
        
        graphs= get_graphs(pred_copy,test_y_copy)
        print(graphs)

        l = []
        for i in graphs:
            im = Image.open(i)
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data2 = base64.b64encode(data.getvalue())
            l.append(encoded_img_data2.decode('utf-8'))

        result = np.array(model.prediction(input_sequences, targets, prediction_data))
        result = np.array([dataProcessing.inverse_transform(result,numDays)])
        
        print('result has shape {}'.format(result))
        answers = []
        for i in result[0]:
            answers.append(i[4])
        print('answers : {}'.format(answers))

        
        per_change = []
        for i in range(numDays):
            if(i==0):
                per_change.append(format(100*(answers[0]-last_price)/last_price,".2f"))
            else:
                per_change.append(format(100*(answers[i]-answers[i-1])/answers[i-1],".2f"))
        
        return render_template('home.html',last_price=last_price,img_data=encoded_img_data.decode('utf-8'), graphs_location=l, results=answers, iterations=numDays, change=per_change)
    
if __name__=="__main__":
    app.run(port=8000,debug=True)

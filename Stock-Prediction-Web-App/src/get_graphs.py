from matplotlib import pyplot as plt
import os
import sys
folder_path = os.getcwd()
sys.path.append(folder_path)
from src.logger import logging

def get_graphs(pred, test_y):
    l = []
    logging.info('pred has shape {}'.format(pred.shape))
    logging.info(pred[:5])
    logging.info(test_y[:5])
    graphs_loc = os.path.join(folder_path,'graphs')
    for i in range(pred.shape[1]):
        plt.figure(figsize=(14,4))
        x = pred[:,i,4]
        y = test_y[:,i,4]
        plt.plot(x, 'r', label = 'Prediction')
        plt.plot(y, 'b', label = 'Actual')
        s = 'stock' + str(i)+'.jpg'
        graph_loc = os.path.join(graphs_loc, s)
        plt.savefig(graph_loc)
        l.append(graph_loc)
    
    return l
import os
import sys
folder_path = os.getcwd()
sys.path.append(folder_path)

import yfinance as yf
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging

class StockData():
    def __init__(self) -> None:
        logging.info('Stock Data object initialised.')

    def getStockData(self,tickerSymbol,time):
        try :
            tickerData = yf.Ticker(tickerSymbol)
            self.df = tickerData.history(period=time,)
            logging.info('Stock data for {} collected.'.format(tickerSymbol))
            return self.df
        except Exception as e:
            raise CustomException(e,sys)
    
    def getCloseGraph(self):
        try :
            graphs_loc = os.path.join(folder_path,'graphs')
            fig = plt.figure(figsize=(10,4))

            plt.plot(self.df['Close'], label='Close Price')
            plt.ylabel('Close Price', fontsize=18)
            plt.legend(loc="upper left", fontsize=12)
            logging.info('Stock data graph generated.')

            graph_loc = os.path.join(graphs_loc,'stock.jpg')
            plt.savefig(graph_loc)
            logging.info('Plot saved.')
            return graph_loc
        
        except Exception as e:
            raise CustomException(e,sys)

# if __name__=="__main__":
#     try :
#         stockData = StockData()
#         print(stockData.getStockData(tickerSymbol='AMZN',startDate='2023-6-1',endDate='2023-6-15'))
#         print(stockData.getCloseGraph())
#     except Exception as e:
#         raise CustomException(e,sys)
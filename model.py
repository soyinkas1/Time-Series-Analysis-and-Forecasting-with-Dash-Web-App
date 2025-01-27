from utils import evaluate_fbp_model
from commodity_etl import CommodityData
from logger import logging
from exception import CustomException
import sys

class ModelTrainer:
    def __init__(self, commodity):
        self.commodity=commodity

    def load_raw_data(self, interval, start_date, end_date):
        try:
            cd = CommodityData()
            logging.info("Data loader initialised")
            # Upload data from API or File
            self.df = cd.etl_commodity_data(self.commodity, interval, start_date, end_date)
            logging.info("Data loaded")
        except Exception as e:
            raise CustomException(e,sys)

    def prep_train_data_prophet(self, percent:float):
        # lets rename the DF as required in `fbprophet`
        try:
            df_fbp = self.df.copy()
            df_fbp.reset_index(inplace=True)
            df_fbp.rename(columns={'value':'y', 'date': 'ds'}, inplace=True)
            df_fbp = df_fbp[['ds', 'y']]
            df_fbp.sort_values('ds',inplace=True)
            train_size = int(len(df_fbp) * percent)

            train_df = df_fbp.iloc[:train_size]
            test_df = df_fbp.iloc[train_size:]

            logging.info("data split into train and test set")
            
            return train_df , test_df
        except Exception as e:
            raise CustomException(e,sys)
    
def main():
    try:
        commodity = 'WTI'
        interval = 'daily'
        start_date = '2022-01-01'
        end_date = '2023-01-01'
        percent = 0.8

        logging.info("Model trainer initialised")
        model_trainer = ModelTrainer('WTI')
        model_trainer.load_raw_data(interval, start_date, end_date, percent)
        train, test = model_trainer.prep_train_data_prophet
        tuning_results = evaluate_fbp_model(train, test, cutoff='180 days', horizon='365 days', param='param.yaml')
        logging.info("Model trained and evaluated")
        return tuning_results
    except Exception as e:
            raise CustomException(e,sys)


if __name__ == '__main__':
    main()
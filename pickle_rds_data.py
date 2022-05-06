import pickle
import os
from sqlalchemy import create_engine, inspect
import pandas as pd

class SaveRDSData():
    def __init__(self, rds_params) -> None:
        self._rds_params = rds_params

    def save(self, tab_path, img_path):
        # connect to the RDS
        DATABASE_TYPE = rds_params['DATABASE_TYPE']
        DBAPI = rds_params['DBAPI']
        ENDPOINT = rds_params['ENDPOINT']
        USER = rds_params['USER']
        PASSWORD = rds_params['PASSWORD']
        PORT = rds_params['PORT']
        DATABASE = rds_params['DATABASE']

        rds_engine = create_engine(
            f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")
        rds_engine.connect()

        # save the products table
        products_raw_df = pd.read_sql_table('products', rds_engine)
        with open(tab_path, 'wb') as f:
            pickle.dump(products_raw_df, f)

        # save the image details table
        image_details_raw_df = pd.read_sql_table('images', rds_engine)
        with open(img_path, 'wb') as f:
            pickle.dump(image_details_raw_df, f)

if __name__=='__main__':

    rds_params = {}
    rds_params['DATABASE_TYPE'] = 'postgresql'
    rds_params['DBAPI'] = 'psycopg2'
    rds_params['ENDPOINT'] = "products.c8k7he1p0ynz.us-east-1.rds.amazonaws.com"
    rds_params['USER'] = 'postgres'
    rds_params['PASSWORD'] = 'aicore2022!'
    rds_params['PORT'] = 5432
    rds_params['DATABASE'] = 'postgres'

    # location for tabular data
    tab_path = os.getcwd() + '/data/tabular/' + 'tab_data_raw.pkl'
    img_path = os.getcwd() + '/data/images/' + 'img_details.pkl'

    saver = SaveRDSData(rds_params=rds_params)
    saver.save(tab_path=tab_path, img_path=img_path)

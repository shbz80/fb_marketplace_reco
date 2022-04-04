import pandas as pd
from sqlalchemy import create_engine, inspect
from train_test_split import TrainTestSplitFBMarketData
from clean_tabular import price_pipeline, basic_pipeline

# connect to the RDS
DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = "products.c8k7he1p0ynz.us-east-1.rds.amazonaws.com"
USER = 'postgres'
PASSWORD = 'aicore2022!'
PORT = 5432
DATABASE = 'postgres'
rds_engine = create_engine(
    f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")
rds_engine.connect()

# read the products tabular data into a dataframe
products_df = pd.read_sql_table('products', rds_engine)

# split data into train and test
data_splitter = TrainTestSplitFBMarketData(product_cat_level=1)
train_data, test_date = data_splitter.train_test_split(products_df, 0.2)

# apply a pipleline transform to clean training data
train_data_tr = price_pipeline.fit_transform(train_data)


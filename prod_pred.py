import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect
from prepare_image_data import PrepareImageData
from train_test_split import TrainTestSplitFBMarketData
from clean_tabular import price_pipeline, basic_pipeline

# location for tabular
path_tabular = os.getcwd() + '/data/tabular/' + 'tab_data_raw.pkl'

# location for images
path_images = os.getcwd() + '/data/images/' + 'img_details.pkl'

# LOAD SAVED TABULAR AND IMAGE DETAILS DATA
with open(path_tabular, 'rb') as f:
    products_raw_df = pickle.load(f)
with open(path_images, 'rb') as f:
    image_details_raw_df = pickle.load(f)

# split data into train and test
data_splitter = TrainTestSplitFBMarketData(product_cat_level=0)
train_data, test_data = data_splitter.train_test_split(products_raw_df, 0.2)

# apply a pipleline transform to clean the training data
# set a category level
# 0 for broader level; 1 for detailed level (only 2 levels)
cat_level = 1   # retain lower level category
loc_level = 0   # retain higher level location
basic_pipeline.set_params(cat_cleaner__cat_selected=cat_level)
basic_pipeline.set_params(loc_cleaner__cat_selected=loc_level)
train_data_tr = basic_pipeline.fit_transform(train_data)
if train_data_tr.isna().sum().sum():
    raise ValueError
test_data_tr = basic_pipeline.fit_transform(test_data)
if test_data_tr.isna().sum().sum():
    raise ValueError

images_dir = os.getcwd() + '/data/images/'
image_cleaner = PrepareImageData(
    products_raw_df, image_details_raw_df, images_dir)

# # visualize the train data
# train_image_stat_dict = image_cleaner.get_image_stat(train_data_tr['category'])
# train_image_stat = pd.DataFrame(train_image_stat_dict)
# train_image_stat.hist()
# train_image_stat['mode'].value_counts().plot.bar()
# train_image_stat['cat'].value_counts().plot.bar()
# print('cats', len(train_image_stat['cat'].value_counts()))
# plt.show()

# PREPARE TEST DATA
test_image_ids = image_cleaner.get_image_ids(test_data_tr.index)

from PIL import Image
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from train_test_split import TrainTestSplitFBMarketData
from clean_tabular import basic_pipeline


class PrepareImageData():
    def __init__(self, product_df, image_details_df, image_path):
        self.product_df = product_df
        self.image_details_df = image_details_df
        self.image_path = image_path

    def get_image_ids(self, product_ix):
        product_ids = self.product_df['id']
        product_id = product_ids.loc[product_ix]
        image_prod_ids = self.image_details_df['product_id']
        selected_image_mask = image_prod_ids.isin([product_id])
        selected_image_ids = self.image_details_df['id'][selected_image_mask]
        return selected_image_ids

    def get_image_stat(self, cat_labels_tr):
        product_ixs = cat_labels_tr.index
        stat_dict = {'width': [], 'height': [],
                     'aspect_ratio': [], 'mode': [], 'cat': []}
        for prod_ix in product_ixs:
            prod_cat = cat_labels_tr.loc[prod_ix]
            # prod_des = self.product_df.loc[prod_ix]['product_description']
            # print(prod_des, prod_cat)
            prod_image_ids = self.get_image_ids(prod_ix)
            for prod_image_id in prod_image_ids:
                file_name = prod_image_id + '.jpg'
                image_file_path = self.image_path + file_name
                im = Image.open(image_file_path)
                # im.show()
                width, height = im.size
                stat_dict['width'].append(width)
                stat_dict['height'].append(height)
                stat_dict['aspect_ratio'].append(width / height)
                stat_dict['mode'].append(im.mode)
                stat_dict['cat'].append(prod_cat)
        return stat_dict

    def prepare_dataset(self, train_data, test_data, pklname, size=None):
        if not size:
            size = (100, 150)

        train_dict = dict()
        test_dict = dict()
        
        data, label = self.prepare_data(train_data)
        train_dict['label'] = label
        train_dict['data'] = data

        data, label = self.prepare_data(test_data)
        test_dict['label'] = label
        test_dict['data'] = data

        train_pklname = os.getcwd() + '/data/images/' + pklname + '_train.pkl'
        test_pklname = os.getcwd() + '/data/images/' + pklname + '_test.pkl'

        joblib.dump(train_dict, train_pklname)
        joblib.dump(test_dict, test_pklname)

    def prepare_data(self, dataset, size=None):
        if not size:
            size = (100, 150)
        # the required image size
        w_req, h_req = size
        # the required aspect ratio
        a_r_req = w_req / h_req
        product_ixs = dataset.index
        label = []
        data = []
        for prod_ix in product_ixs:
            prod_cat = dataset.loc[prod_ix]['category']
            prod_des = dataset.loc[prod_ix]['product_description']
            prod_image_ids = self.get_image_ids(prod_ix)
            for prod_image_id in prod_image_ids:
                file_name = prod_image_id + '.jpg'
                image_file_path = self.image_path + file_name
                im = Image.open(image_file_path)
                # im.show()
                # convert image to RGB
                im = im.convert('RGB')
                # im.show()
                # flip image to maintian an aspect ratio <= 1
                w, h = im.size
                a_r = w / h
                if a_r > 1.0:
                    im = im.rotate(90)
                # im.show()
                w, h = im.size
                # resize image to required size maintaining aspect ratio
                h_new = int(w_req / a_r)
                if h_new <= h_req:
                    w_new = int(w_req)
                else:
                    w_new = int(h_req * a_r)
                    h_new = int(h_req)
                    if h_new > h_req:
                        raise Exception
                im = im.resize((w_new, h_new))
                # print(im.size)
                # im.show()
                # create a black image of the req size
                result = Image.new(im.mode, (w_req, h_req), (0, 0, 0))
                # result.show()
                if w_new < w_req:
                    w_margin = (w_req - w_new) / 2
                else:
                    w_margin = 0
                w_margin = int(w_margin)
                if h_new < h_req:
                    h_margin = (h_req - h_new) / 2
                else:
                    h_margin = 0
                h_margin = int(h_margin)
                # paste the image on to thw background
                result.paste(im, (w_margin, h_margin))
                # result.show()
                data.append(result)
                label.append(prod_cat)
        return data, label

if __name__ == '__main__':
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
    train_data, test_data = data_splitter.train_test_split(
        products_raw_df, 0.2)

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

    # visualize the train data
    # train_image_stat_dict = image_cleaner.get_image_stat(train_data_tr['category'])
    # train_image_stat = pd.DataFrame(train_image_stat_dict)
    # print(train_image_stat.describe())
    # train_image_stat.hist()
    # train_image_stat['mode'].value_counts().plot.bar()
    # train_image_stat['cat'].value_counts().plot.bar()
    # print('cats', len(train_image_stat['cat'].value_counts()))
    # plt.show()

    image_cleaner.prepare_dataset(train_data_tr, test_data_tr, 'img_prepared')
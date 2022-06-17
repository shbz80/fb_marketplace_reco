'''
Implements classes for preparing the image data for CNN classification
'''
from PIL import Image
import os
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from train_test_split import TrainTestSplitFBMarketData
from clean_tabular import basic_pipeline as tab_basic_pipeline


class PrepareImageData():
    """The main class for preparing the datasets for CNN classification"""
    def __init__(self, product_df, image_details_df, image_path):
        """
        Args:
            product_df (dataframe): the main tabular data
            image_details_df (dataframe): the image tabular data
            image_path (str): path to the raw images
        """
        self.product_df = product_df
        self.image_details_df = image_details_df
        self.image_path = image_path

    def get_image_ids(self, product_ix):
        """Returns all the image ids for the given product id
        Args:
            product_ix (index): product index

        Returns:
            str: the image uuids for the given product index
        """
        product_ids = self.product_df['id']
        product_id = product_ids.loc[product_ix]
        image_prod_ids = self.image_details_df['product_id']
        selected_image_mask = image_prod_ids.isin([product_id])
        selected_image_ids = self.image_details_df['id'][selected_image_mask]
        return selected_image_ids

    def get_image_stat(self, cat_labels_tr):
        """prepares and return a stat dict for images
        Args:
            cat_labels_tr (pd series): the product category column

        Returns:
            dict: dict that contains image stats
        """
        # get all product indices
        product_ixs = cat_labels_tr.index
        stat_dict = {'width': [], 'height': [],
                     'aspect_ratio': [], 'mode': [], 'cat': []}
        # loop though product indices
        for prod_ix in product_ixs:
            # get the product category for this index
            prod_cat = cat_labels_tr.loc[prod_ix]
            # get all image uuids (also file names) for this prod index
            prod_image_ids = self.get_image_ids(prod_ix)
            # loop through all the images of this prod index
            for prod_image_id in prod_image_ids:
                # get the image path
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

    def prepare_dataset(self, train_data, val_data, test_data,
                pklname='img_pkl', size=None, mode='RGB'):
        if not size:
            size = (100, 150)

        train_dict = dict()
        val_dict = dict()
        test_dict = dict()
        
        data, label, desc = self.prepare_data(train_data, size=size, mode=mode)
        train_dict['label'] = label
        train_dict['data'] = data
        train_dict['desc'] = desc

        data, label, desc = self.prepare_data(val_data, size=size, mode=mode)
        val_dict['label'] = label
        val_dict['data'] = data
        val_dict['desc'] = desc

        data, label, desc = self.prepare_data(test_data, size=size, mode=mode)
        test_dict['label'] = label
        test_dict['data'] = data
        test_dict['desc'] = desc

        train_pklname = os.getcwd() + '/data/images/' + pklname + '_train.pkl'
        val_pklname = os.getcwd() + '/data/images/' + pklname + '_val.pkl'
        test_pklname = os.getcwd() + '/data/images/' + pklname + '_test.pkl'

        joblib.dump(train_dict, train_pklname)
        joblib.dump(val_dict, val_pklname)
        joblib.dump(test_dict, test_pklname)

    def prepare_data(self, dataset, size=None, mode='RGB'):
        """Prepares a dataset with images, label and text descriptions
        Args:
            dataset (pd dataframe): the main tabular dataframe
            size (tuple, optional): image size. Defaults to None.
            mode (str, optional): image mode. Defaults to 'RGB'.

        Returns:
            tuple: a tuple of lists of image, label and description
        """
        if not size:
            size = (100, 150)
        # the required image size
        w_req, h_req = size
        # the required aspect ratio
        a_r_req = w_req / h_req
        product_ixs = dataset.index
        label = []
        data = []
        desc = []
        # loop through the product indices
        for prod_ix in product_ixs:
            # get some column values for this index
            prod_cat = dataset.loc[prod_ix]['category']
            prod_des = dataset.loc[prod_ix]['product_description']
            prod_name = dataset.loc[prod_ix]['product_name']
            # get all the image uuids (also file names) for this prod index
            prod_image_ids = self.get_image_ids(prod_ix)
            # loop though the image uuids
            for prod_image_id in prod_image_ids:
                # open the image
                file_name = prod_image_id + '.jpg'
                image_file_path = self.image_path + file_name
                im = Image.open(image_file_path)
                # im.show()
                # convert image to the given mode
                im = im.convert(mode)
                # flip image to maintian an aspect ratio <= 1
                w, h = im.size
                a_r = w / h
                if a_r > 1.0:
                    im = im.rotate(90)
                w, h = im.size
                a_r = w / h
                # resize image to required size maintaining aspect ratio
                w_new = int(w_req)
                h_new = int(w_req / a_r)
                if h_new > h_req:
                    h_new = h_req
                    w_new = int(h_req * a_r)
                im = im.resize((w_new, h_new))
                # create a black image of the req size
                result = Image.new(im.mode, (w_req, h_req), (0, 0, 0))
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
                # paste the image on to the background to pad
                result.paste(im, (w_margin, h_margin))
                data.append(result)
                label.append(prod_cat)
                if prod_name[-1] == '.':
                    spacer = ' '
                else:
                    spacer = '. '
                desc.append(prod_name + spacer + prod_des)
        return data, label, desc

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
    # use lower level category for stratified split
    cat_level = 1   
    data_splitter = TrainTestSplitFBMarketData(product_cat_level=cat_level)
    # split the data into train (70) val (15) and test (15)
    # tab_train, tab_val, tab_test = data_splitter.train_test_split(
    #     products_raw_df, (0.7, 0.15, 0.15))
    tab_train, tab_val, tab_test = data_splitter.train_test_split(
        products_raw_df, (0.6, 0.2, 0.2))

    # apply a pipleline transform to clean the training data
    # set a category level
    # 0 for broader level; 1 for detailed level (only 2 levels)
    cat_level = 0   # retain higher level category
    loc_level = 0   # retain higher level location
    tab_basic_pipeline.set_params(cat_cleaner__cat_selected=cat_level)
    tab_basic_pipeline.set_params(loc_cleaner__cat_selected=loc_level)

    # apply transform on tabular train data
    tab_train_tr = tab_basic_pipeline.fit_transform(tab_train)
    if tab_train_tr.isna().sum().sum():
        raise ValueError

    # apply transform on tabular test data
    tab_test_tr = tab_basic_pipeline.fit_transform(tab_test)
    if tab_test_tr.isna().sum().sum():
        raise ValueError

    # apply transform on tabular val data
    tab_val_tr = tab_basic_pipeline.fit_transform(tab_val)
    if tab_val_tr.isna().sum().sum():
        raise ValueError

    images_dir = os.getcwd() + '/data/images/'
    image_cleaner = PrepareImageData(
        products_raw_df, image_details_raw_df, images_dir)

    # # visualize the train data
    # train_image_stat_dict = image_cleaner.get_image_stat(tab_train_tr['category'])
    # train_image_stat = pd.DataFrame(train_image_stat_dict)
    # print(train_image_stat.describe())
    # train_image_stat.hist()
    # plt.show()
    # train_image_stat['mode'].value_counts().plot.bar()
    # plt.show()
    # train_image_stat['cat'].value_counts().plot.bar()
    # plt.show()
    # print('cats', len(train_image_stat['cat'].value_counts()))

    # prepare the train, val and test datasets
    # each data set contains images, labels and descriptions
    # image data is prepared for RESNET-50
    image_cleaner.prepare_dataset(
        tab_train_tr, tab_val_tr, tab_test_tr,
        pklname='img_prepared', size=(224, 224),
        mode='RGB')


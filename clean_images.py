import os
import random
import time
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class RGBToGrayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([np.array(im.convert('L')) for im in X])


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, orientations=9,
                 pixels_per_cell=(14, 14),
                 cells_per_block=(2, 2),
                 block_norm='L2-Hys'):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            result = hog(X,
                         orientations=self.orientations,
                         pixels_per_cell=self.pixels_per_cell,
                         cells_per_block=self.cells_per_block,
                         block_norm=self.block_norm)
            return result

        try:
            return np.array([local_hog(im) for im in X])
        except:
            raise Exception('HOG tranformation failed')


image_pipeline = Pipeline([
    ('grayify', RGBToGrayTransformer()),
    ('hogify', HogTransformer()),
    ('scalify', StandardScaler()),
])


def view_random_images(im_data_dict, num_images=10, delay=3):
    data_size = len(im_data_dict['data'])
    for i in range(num_images):
        ix = random.randint(0, data_size - 1)
        im = im_data_dict['data'][ix]
        label = im_data_dict['label'][ix]
        print(label)
        im.show()
        time.sleep(delay)


if __name__ == '__main__':
    train_pklname = os.getcwd() + '/data/images/' + 'img_prepared' + '_train.pkl'
    test_pklname = os.getcwd() + '/data/images/' + 'img_prepared' + '_test.pkl'

    # load the prepared data
    train_data = joblib.load(train_pklname)
    test_data = joblib.load(test_pklname)
    im = train_data['data'][0]

    # get some common image details
    image_size = im.size
    image_mode = im.mode

    # random images before transormations
    # view_random_images(train_data, num_images=10, delay=3)

    X_train = np.array(train_data['data'])
    y_train = np.array(train_data['label'])

    X_test = np.array(test_data['data'])
    y_test = np.array(test_data['label'])
    
    # apply the pipleline: grayify, hogify and scalify
    X_train_tr = image_pipeline.fit_transform(X_train)
    X_test_tr = image_pipeline.fit_transform(X_test)

    # fit to the default SGD classifier (linear SVM)
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train_tr, y_train)

    # training loss
    y_pred =sgd_clf.predict(X_train_tr)
    correct_perc = sum(y_pred == y_train) / len(y_train)
    print('Training loss: ', correct_perc)
    # Training loss:  0.38534114609196546

    # test loss
    y_pred =sgd_clf.predict(X_test_tr)
    correct_perc = sum(y_pred == y_test) / len(y_test)
    print('Test loss: ', correct_perc)
    # Test loss:  0.1467455621301775

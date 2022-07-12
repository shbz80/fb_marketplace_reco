import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from image_proc.clean_images import RGBToGrayTransformer, HogTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

image_pipeline = Pipeline([
    ('grayify', RGBToGrayTransformer()),
    ('hogify', HogTransformer()),
    ('scalify', StandardScaler()),
])

train_pklname = os.getcwd() + '/data/images/' + 'img_prepared' + '_train.pkl'
test_pklname = os.getcwd() + '/data/images/' + 'img_prepared' + '_test.pkl'

# load the prepared data
train_data = joblib.load(train_pklname)
test_data = joblib.load(test_pklname)

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
y_pred = sgd_clf.predict(X_train_tr)
correct_perc = sum(y_pred == y_train) / len(y_train)
print('Training loss: ', correct_perc)
# Training loss:  0.38534114609196546

# test loss
y_pred = sgd_clf.predict(X_test_tr)
correct_perc = sum(y_pred == y_test) / len(y_test)
print('Test loss: ', correct_perc)
# Test loss:  0.1467455621301775

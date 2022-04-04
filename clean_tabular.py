import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class BasicCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.replace({'N/A': None})
        X.drop(columns=['id', 'page_id', 'create_time'], inplace=True)
        return X


class NullCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.dropna(inplace=True)
        return X


class CatSplitterMixin():
    """ This mixin class provides methods for splitting heirarchical
    multilevel product categories and locations into individual columns"""

    @staticmethod
    def split_string(text, sep):
        if text is None:
            return None
        l = text.split(sep)
        return [i.strip() for i in l]

    def split_by_slash(self, text):
        return self.split_string(text, '/')

    def split_by_comma(self, text):
        return self.split_string(text, ',')

    @staticmethod
    def pop_top_word(word_l):
        if word_l:
            return word_l.pop(0)
        else:
            return None

    def split_expand_col(self, df, col_name, sep_func):
        df_cat_list = df[col_name].apply(sep_func)
        cat_num = max(df_cat_list.dropna().apply(len))
        for i in range(cat_num):
            df[f'{col_name}_{i}'] = df_cat_list.apply(self.pop_top_word)
        df.drop(columns=[col_name], inplace=True)
        return df, cat_num

    def select_cat(self, df, cat_num, cat_prefix, cat_ix):
        cat_drop = list(range(cat_num))
        if cat_ix in cat_drop:
            cat_drop.remove(cat_ix)
        else:
            raise Exception('Selected category cannot be found')
        cat_drop = [f'{cat_prefix}_{i}' for i in cat_drop]

        df.drop(columns=cat_drop, inplace=True)
        df = df.rename(columns={f'{cat_prefix}_{cat_ix}': f'{cat_prefix}'})
        return df


class ProductCatCleaner(BaseEstimator, TransformerMixin, CatSplitterMixin):
    ''' a dataset transformer that splits slash seperated categories in 
        categories column and expands the dataframe
    '''

    def __init__(self, cat_selected=1):
        # categories to drop after splitting
        self.cat_selected = cat_selected

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X, cat_num = self.split_expand_col(X, 'category', self.split_by_slash)

        X = self.select_cat(X, cat_num, 'category', self.cat_selected)

        return X


class LocationCatCleaner(BaseEstimator, TransformerMixin, CatSplitterMixin):
    ''' a dataset transformer that splits comma seperated categories in 
        location column and expands the dataframe
    '''

    def __init__(self, cat_selected=1):
        # categories to drop after splitting
        self.cat_selected = cat_selected

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X, cat_num = self.split_expand_col(X, 'location', self.split_by_comma)

        X = self.select_cat(X, cat_num, 'location', self.cat_selected)

        return X


class PriceDataCleaner(BaseEstimator, TransformerMixin):
    ''' a transformer for cleaning the price column '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['price'] = X['price'].str.replace('£', '')
        X['price'] = X['price'].str.replace(',', '')
        X['price'] = X['price'].astype('float')
        return X


class PriceRegressionTransformer(BaseEstimator, TransformerMixin):
    ''' a transformer for preping tabular data for price prediction
    regression problem. It drops all text fetaures'''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=['product_name', 'product_description'])
        cat_encoded = pd.get_dummies(X.category, drop_first=True)
        loc_encoded = pd.get_dummies(X.location, drop_first=True)
        X = pd.concat([X, cat_encoded, loc_encoded], axis=1)
        X = X.drop(columns=['category', 'location'])
        return X


basic_pipeline = Pipeline([
    ('basic_cleaner', BasicCleaner()),
    ('null_cleaner', NullCleaner()),
    ('cat_cleaner', ProductCatCleaner()),
    ('loc_cleaner', LocationCatCleaner()),
    ('price_cleaner', PriceDataCleaner()),
])


price_pipeline = Pipeline([
    ('common', basic_pipeline),
    ('price_reg', PriceRegressionTransformer()),
])

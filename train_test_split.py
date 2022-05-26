import pandas as pd
from pandas import value_counts
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from clean_tabular import BasicCleaner, ProductCatCleaner


class TrainTestSplitFBMarketData():
    """A class to handle spliting the FB marketplace tabular data using 
    stratified shuffle split. 
    """
    def __init__(self, product_cat_level=0) -> None:
        """The feature used to do the startified shuffle split is specified
        here as a hyperparameter """
        self.product_cat_level = product_cat_level

    def train_test_split(self, df, ratio):
        """ Splits and returns the two datasets """
        split_1 = StratifiedShuffleSplit(
            n_splits=1, test_size=ratio[2], random_state=42)
        
        # use this piple line to extract the product catagory level that 
        # is used for the stratified shuffle
        split_pipeline = Pipeline([
            ('basic', BasicCleaner()),
            ('cat_splitter', ProductCatCleaner(
                cat_selected=self.product_cat_level)),
        ])
        
        df_tr = split_pipeline.fit_transform(df)
        
        # get the extracted product category feature column
        cat_col = df_tr[f'category']
        
        # replace None with 'NA' category for stratified sfuffle
        cat_col.loc[cat_col.isna()] = 'NA'
        
        # assign 'NA' to all ctaegories with counts less than the value
        # required to produce at least one test sample
        cat_counts = cat_col.value_counts()
        insig_cats = cat_counts[cat_counts < int(1/ratio[2])]
        insig_cat_dict = {cat:'NA' for cat in insig_cats.index}
        cat_col = cat_col.replace(insig_cat_dict)

        # split the data into train and test
        for train_index, test_index in split_1.split(df.to_numpy(), cat_col.to_numpy()):
            strat_train_set = df.iloc[train_index]
            strat_test_set = df.iloc[test_index]
            strat_cat_col = cat_col.iloc[train_index]
        
        split_2 = StratifiedShuffleSplit(
            n_splits=1, test_size=ratio[1]/(ratio[0] + ratio[1]), random_state=42)

        # split the train data into train and val
        for train_index, val_index in split_2.split(strat_train_set.to_numpy(), strat_cat_col.to_numpy()):
            strat_val_set = strat_train_set.iloc[val_index]
            strat_train_set = strat_train_set.iloc[train_index]
        
        # test the validity of the split
        all_indices = list(strat_train_set.index) + \
            list(strat_test_set.index) + list(strat_val_set.index)
        all_indices.sort()
        test_indeces = list(range(len(all_indices)))
        assert(all_indices == test_indeces)

        return strat_train_set, strat_val_set, strat_test_set

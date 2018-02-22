""" The Pipeline object in sci-kit learn is very useful for constructing
    simple and complex modeling pipelines. However, out of the box it is
    cumbersome to build pipelines that involve heterogenous data. Most
    transformers assume that the entirety of the input datasets are of the
    same dtype. So, how do you scale your numeric columns, make dummy
    variables out of your categorical variables, run TF-IDF on your text
    columns, and fit all of this into one pipeline? A possible solution
    is:

    http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html

    But building these FeatureUnion objects in a hardcoded fashion can be
    cumbersome in and of itself. The DtypePipeline object here is a solution
    to this. You can pass in Pipeline objects for each data type, and it
    takes care of the rest.
"""


import bisect
import copy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class DtypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=None):
        """ Initialize with a specific dtype to select. If None,
            will select all columns.

            :param dtype: a string representing the dtype to select.
                Valid values are:
                    'numeric' - for numbers
                    'categorical' - for categorical data
                    'ordinal' - for ordered categorical data
                    'text' - for free text columns
                    'datetime' - for datetime columns """
        self.dtype = dtype

    def __select_cat(self, X, ordered=False):
        """ Selects categorical columns, either categorical or ordinal.

            :param X: the dataset to select from

            :param ordered: if True, will return the ordinal columns,
                otherwise will return the categorical columns

            :return: a list of columns representing the categorical or
                ordinal variables. """
        c_cols = X.select_dtypes(include='category').columns
        return X[[c for c in c_cols if X[c].dtype.ordered == ordered]]

    def fit(self, X, y=None):
        """ Fits the dtype selector.

            :param X: the input X matrix

            :param y: optional target labels

            :return: returns self. """
        return self

    def transform(self, X):
        """ Selects the given input by selecting the specified
            column types.

            :param X: the input X matrix

            :return: a subset of X, with the specified dtypes selected. """
        if self.dtype == 'numeric':
            return X.select_dtypes(include=[np.number])
        elif self.dtype == 'categorical':
            return self.__select_cat(X, ordered=False)
        elif self.dtype == 'ordinal':
            return self.__select_cat(X, ordered=True)
        elif self.dtype == 'text':
            return X.select_dtypes(include='object')
        elif self.dtype == 'datetime':
            return X.select_dtypes(include='datetime')
        else:
            return X


class ColSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        """ Initialize with a specific column to select. If None,
            will select all columns.

            :param col: a string representing the column to select. """
        self.column = column

    def fit(self, X, y=None):
        """ Fits the column selector.

            :param X: the input X matrix

            :param y: optional target labels

            :return: returns self. """
        return self

    def transform(self, X):
        """ Selects the given input by selecting the specified column.

            :param X: the input X matrix

            :return: a subset of X, with the specified column selected. """
        if not self.column:
            return X
        return X[self.column]


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        """ Initializes the categorical encoder transform. """
        self._lbl = LabelEncoder()
        self._onehot = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        """ Fits the categorical encoder to the given dataset.

            :param X: the input dataset

            :param y: optional target labels.

            :return: returns self. """
        self._lbl.fit(X)
        self._onehot.fit(self._lbl.transform(X).reshape(-1, 1))
        return self

    def transform(self, X):
        """ Transforms the input categorical variables into a
            one/zero categorical encoding.

            :param X: the input dataset

            :return: the transformed input dataset. """
        # The bisection is to ensure that LabelEncoder can handle unseen
        # categories in the test set
        if isinstance(self._lbl.classes_[0], str):
            catch_val = '<unknown>'
        else:
            catch_val = -1

        X = X.map(lambda x: catch_val if x not in self._lbl.classes_ else x)
        lbl_classes = self._lbl.classes_.tolist()
        bisect.insort_left(lbl_classes, catch_val)
        self._lbl.classes_ = np.array(lbl_classes)
        return self._onehot.transform(self._lbl.transform(X).reshape(-1, 1))


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='most_frequent'):
        """ Initializes the categorical imputer.

            :param strategy: either 'most_frequent' or 'distribution'.
                If 'most_frequent', will impute the missing values
                with the most frequent value. If 'distribution', will
                create a discrete distribution based on the frequencies
                and will sample from this distribution. """
        self.strategy = strategy

    def fit(self, X, y=None):
        """ Fits the categorical imputer.

            :param X: the input X matrix

            :param y: optionally the target labels """
        counts = X.value_counts()
        self.counts = counts / counts.sum()
        return self

    def transform(self, X):
        """ Fills the missing values with the most common value (if
            'most_frequent' is the strategy) or sampled from the
            distribution.

            :param X: the input dataset

            :return: the transformed dataset with the missing values
                filled in. """
        if self.strategy == 'most_frequent':
            X = X.fillna(self.counts.index[0])
        elif self.strategy == 'distribution':
            missing = np.random.choice(self.counts.index,
                                       X.shape[0], p=self.counts)
            missing = pd.DataFrame(missing, index=X.index, columns=X.columns)
            X = X.fillna(missing)
        
        return X


class DtypePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, numeric=None, categorical=None,
                 ordinal=None, text=None, datetime=None):
        """ Builds the model tree with the given parameters.

            :param numeric: a Pipeline object to fit and transform all
                numeric columns

            :param categorical: a Pipeline object to fit and transform
                to all categorical columns

            :param ordinal: a Pipeline object to fit and transform to
                all categorical columns

            :param text: a Pipeline object to fit and transform to all
                text columns

            :param datetime: a Pipeline object to fit and transform to
                all datetime columns """
        self.model = None
        # Prepend each pipeline with a step to select the dtype
        self._steps = {'numeric': self.__type_sel(numeric, 'numeric'),
                       'categorical': self.__type_sel(categorical, 'categorical'),
                       'ordinal': self.__type_sel(ordinal, 'ordinal'),
                       'text': self.__type_sel(text, 'text'),
                       'datetime': self.__type_sel(datetime, 'datetime')}

    def __type_sel(self, pipeline, dtype):
        """ Prepends a step in a pipeline to select the given dtype first.

            :param pipeline: the pipeline to prepend to

            :param dtype: the dtype to add to

            :return: returns a Pipeline that selects the dtype and then
                performs the input pipeline. """
        if not pipeline:
            return pipeline

        select_step = [(str(dtype), DtypeSelector(dtype=dtype))]
        return Pipeline(select_step + copy.deepcopy(pipeline.steps))

    def __col_sel(self, pipeline, col):
        """ Prepends a column selector transform to a pipeline.

            :param pipeline: the pipeline to prepend to

            :param col: the column to select first

            :return: the modified pipeline object. """
        if not pipeline:
            return pipeline

        # We have to rename the steps in the pipeline so that there aren't
        # any steps with the same name
        select_step = [('{}_select'.format(col), ColSelector(column=col))]
        pipeline_steps = [('{}_{}'.format(col, name), step)
                          for name, step in copy.deepcopy(pipeline.steps)]
        return Pipeline(select_step + pipeline_steps)

    def __per_col(self, X, dtype):
        """ Expands the pipeline to apply it to each column for a given dtype.

            :param X: the input dataset
            
            :param dtype: the dtype to loop through
            
            :return: a FeatureUnion object where each item is a pipeline
                that operates on a given column. """
        pipeline = self._steps[dtype]
        if not pipeline:
            return None

        return FeatureUnion(transformer_list=[
            (col, self.__col_sel(Pipeline(pipeline.steps[1:]), col))
            for col in DtypeSelector(dtype=dtype).fit_transform(X).columns
        ])

    def __str__(self):
        """ Pretty prints the model tree. """
        def str_helper(model, lvl=0):
            is_feat = isinstance(model, FeatureUnion)
            steps = model.transformer_list if is_feat else model.steps
            model_str = model.__class__.__name__ + '('
            for step, func in steps:
                tabs = ''.join(['\t'] * (lvl + 1))
                model_str += "\n{}('{}', ".format(tabs, step)
                if not isinstance(func, (FeatureUnion, Pipeline)):
                    model_str += '{})'.format(func.__class__.__name__)
                else:
                    rest = str_helper(func, lvl=lvl + 1)
                    model_str += '{})'.format(rest)
            return model_str + ')'

        return str_helper(self.model, lvl=0)

    def build_model_tree(self, X, y=None):
        """ Builds the model tree, given the input data. The model tree
            cannot be created until this point because we don't know what
            the columns are until this point. The model tree consists of
            FeatureUnions that transform each group of columns correctly.

            :param X: the input dataset

            :param y: optional labels, may be required for some transformations
                in your pipeline

            :return: the model tree object, as either a FeatureUnion or
                Pipeline object. """
        transform_list = [('numeric', self._steps['numeric']),
                          ('category', self.__per_col(X, 'categorical')),
                          ('ordinal', self.__per_col(X, 'ordinal')),
                          ('text', self.__per_col(X, 'text')),
                          ('datetime', self.__per_col(X, 'datetime'))]

        return FeatureUnion([step for step in transform_list if step[1]])

    def fit(self, X, y=None):
        """ Fits the pipeline to the given data.

            :param X: the input X matrix

            :param y: optional target labels, may be required for some
                transformations in your pipeline

            :return: the fitted DtypePipeline object. """
        self.model = self.build_model_tree(X, y=y)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        """ Transforms the pipeline to the given data.

            :param X: the input X matrix

            :return: the transformed X matrix. """
        return self.model.transform(X)


if __name__ == '__main__':
    import os

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import StandardScaler

    # Read the Titanic dataset in to illustrate this use
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    train = pd.read_csv('{}/../datasets/titanic_train.csv'.format(curr_dir))
    test = pd.read_csv('{}/../datasets/titanic_test.csv'.format(curr_dir))

    # Make sure each column is the correct dtype
    for col in ['Pclass', 'SibSp', 'Parch', 'Embarked', 'Sex']:
        train[col] = pd.Categorical(train[col], ordered=False)
        test[col] = pd.Categorical(test[col], ordered=False)

    train['Name'] = train['Name'].astype('object')
    test['Name'] = test['Name'].astype('object')

    dropcols = ['PassengerId', 'Ticket', 'Cabin']
    train = train.drop(dropcols, axis=1)
    test = test.drop(dropcols, axis=1)
    X, y = train.drop('Survived', axis=1), train['Survived']

    ###########################################################
    # Make the pipelines and combine them using DtypePipeline #
    ###########################################################
    num_pipeline = Pipeline([('impute', Imputer()),
                             ('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('impute', CategoricalImputer()),
                             ('dummy', CategoricalEncoder())])
    txt_pipeline = Pipeline([('tfidf', TfidfVectorizer())])
    preprocessor = DtypePipeline(numeric=num_pipeline,
                                 categorical=cat_pipeline,
                                 text=txt_pipeline)

    model = Pipeline([('preprocess', preprocessor),
                      ('logistic', LogisticRegression())])

    # Fit the model and get CV results
    cv = KFold(n_splits=3)
    for train, test in cv.split(X):
        model.fit(X.ix[train, :], y[train])
        yhat = model.predict(X.ix[test, :])
        print('RMSE: {:.3f}'.format(mean_squared_error(y[test], yhat)))

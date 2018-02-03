""" Functions to easily download census data for a given geography. """

import os
import pandas as pd
from census import Census
from itertools import chain
from functools import reduce
from dotenv import load_dotenv
from dotenv import find_dotenv


def census_key():
    """ Retrieve the Census API key.

        :return: the US Census API key as a string. """
    key = os.environ.get('CENSUS_API_KEY')
    if key is not None:
        return key

    load_dotenv(find_dotenv())
    return os.environ.get('CENSUS_API_KEY')


def columns_for_table(table, line_nums):
    """ For a given table and line numbers, returns a list of census
        columns.

        :param table: a string representing the table name

        :param line_nums: a list of integers representing the line
            numbers in the table to create columns for.

        :return: a list of strings that represent the census columns. """
    line_ends = ['{}E'.format(str(l).zfill(3)) for l in line_nums]
    return ['{}_{}'.format(table, l) for l in line_ends]


def pull_census_columns(cols, census_api=None):
    """ Pull a given list of census columns.

        :param cols: a list of columns to pull

        :param census_api: a Census API object

        :return: a DataFrame where the geography is the index and
            the columns are the columns. """
    if census_api is None:
        census_api = Census(census_key())

    def pull_chunk(col_chunk):
        # Census API only allows 50 columns per call
        print('Pulling columns {}'.format(col_chunk))
        df = pd.DataFrame(census_api.acs5.zipcode(['NAME'] + col_chunk, '*'))
        df['NAME'] = df['NAME'].map(lambda x: x[-5:])
        return (df.rename(columns={'NAME': 'zipcode'})
                  .set_index('zipcode')[col_chunk]
                  .apply(pd.to_numeric, errors='coerce'))

    chunk_fn = lambda c, n: (c[i:i+n] for i in range(0, len(c), n))
    return reduce(lambda x, y: x.join(y),
                  (pull_chunk(chunk) for chunk in chunk_fn(cols, 49)))


def income_variables():
    """ Return the list of income variables to pull from.

        :return: A DataFrame with the income variables as the columns
            and the geography as the index. """
    return columns_for_table('B19001', range(2, 18))


def educational_attainment():
    """ Return the list of educational attainment variables to pull from.

        :return: A DataFrame with the educational variables as the
            columns and the geography as the index. """
    return columns_for_table('B15003', range(16, 26))


def age_variables():
    """ Return the list of age variables to pull from.

        :return: A list of column names to pull. """
    return columns_for_table('B01001', range(2, 50))


def population_variables():
    """ Return the list of population variables to pull from.

        :return: A list of columns names to pull. """
    return columns_for_table('B00001', range(1, 2))


def census_pipeline(col_pipeline):
    """ Given a list of functions, calls them all and joins them to get
        a census dataset.

        :param col_pipeline: a list of functions that return the list of
            columns needed to be pulled.

        :return: a DataFrame with all the variables as columns and zipcode
            as the index. """
    cols = set(chain.from_iterable(f() for f in col_pipeline))
    return pull_census_columns(list(cols))


if __name__ == '__main__':
    # Example use: build a dataset by pulling income, education, age,
    # and population by zipcode and save it out.
    df = census_pipeline([income_variables,
                          educational_attainment,
                          age_variables,
                          population_variables])
    df.to_csv('census_dataset.csv')

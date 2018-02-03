""" Functions to easily download census data for a given geography. """

import os
import pandas as pd
from census import Census
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


def pull_census_columns(cols, census_api):
    """ Pull a given list of census columns.

        :param cols: a list of columns to pull

        :param census_api: a Census API object

        :return: a DataFrame where the geography is the index and
            the columns are the columns. """
    print('Pulling columns {}'.format(cols))
    df = pd.DataFrame(census_api.acs5.zipcode(['NAME'] + cols, '*'))
    df['NAME'] = df['NAME'].map(lambda x: x[-5:])

    return (df.rename(columns={'NAME': 'zipcode'})
              .set_index('zipcode')[cols]
              .apply(pd.to_numeric, errors='coerce'))


def income_variables(census=None):
    """ Pull various income variables for all zipcodes.

        :param census: a Census API object

        :return: A DataFrame with the income variables as the columns
            and the geography as the index. """
    if census is None:
        census_api = Census(census_key())

    cols = columns_for_table('B19001', range(2, 18))
    return pull_census_columns(cols, census_api)


def educational_attainment(census=None):
    """ Pull educational variables for all zipcodes.

        :param census: a Census API object

        :return: A DataFrame with the educational variables as the
            columns and the geography as the index. """
    if census is None:
        census_api = Census(census_key())

    cols = columns_for_table('B15003', range(16, 26))
    return pull_census_columns(cols, census_api)


def census_pipeline(pipeline):
    """ Given a list of functions, calls them all and joins them to get
        a census dataset.

        :param pipeline: a list of functions to combine

        :return: a DataFrame with all the variables as columns and zipcode
            as the index. """
    return reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
                  (func() for func in pipeline))

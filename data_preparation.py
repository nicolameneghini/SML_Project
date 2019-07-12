import pandas as pd
import numpy as np
from datetime import datetime
import re


def find_language(url):
    res = re.search('[a-z][a-z].wikipedia.org', url)
    if res:
        return res[0][0:2]
    return 'na'


def data_per_date(data):

    temp = data.Page.str.rsplit("_", expand=True, n=3)

    data['lang'] = data.Page.map(find_language)
    data['Page'] = temp[0]
    data['Type_of_traffic'] = temp[2]
    data['Agent'] = temp[3]

    data_melted = pd.melt(data, id_vars=['Page', 'Type_of_traffic', 'Agent', 'lang'],
                          var_name='Date', value_name='count')
    data_melted['Date'] = data_melted['Date'].astype('datetime64[ns]')

    return data_melted


def find_page(data_melted, page_name):

    my_page = data_melted[data_melted.Page == page_name]
    my_page.set_index('Date', inplace=True)
    my_page = my_page[['count']]
    return my_page


def divide_page_by_lang(data_melted, page_name):

    my_page = data_melted[data_melted.Page == page_name]

    languages = my_page.lang.unique()
    lang_sets = {}
    for key in languages:
        lang_sets[key] = my_page[my_page.lang ==
                                 key][['Date', 'Page', 'lang', 'count']]

    data = pd.DataFrame(index=lang_sets[languages[0]].Date.unique())
    for key in languages:
        data[key] = lang_sets[key].groupby('Date')['count'].sum().values

    return data

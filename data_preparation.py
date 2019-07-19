import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


def find_language(url):
    res = re.search('[a-z][a-z].wikipedia.org', url)
    if res:
        return res[0][0:2]
    return 'na'


def extract_series(data, n_of_series):
    pages = {}
    titles = {}
    index = pd.to_datetime(data.columns[1:])
    for i in range(n_of_series):
        temp = (data[i:i+1].values[0][1:]).astype(None)
        titles[i] = data[i:i+1].values[0][0]
        if (np.isfinite(np.mean(temp))):
            pages[i] = pd.DataFrame(temp)
            pages[i].fillna(int(pages[i].mean()), inplace=True)
            pages[i].set_index(index)
            pages[i].index.name = 'Date'
            pages[i].columns = ['count']

    return pages, titles


def data_per_date(data):

    data1 = data.copy()
    temp = data1.Page.str.rsplit("_", expand=True, n=3)

    data1['lang'] = data1.Page.map(find_language)
    data1['Page'] = temp[0]
    data1['Type_of_traffic'] = temp[2]
    data1['Agent'] = temp[3]

    data_melted = pd.melt(data1, id_vars=['Page', 'Type_of_traffic', 'Agent', 'lang'],
                          var_name='Date', value_name='count')
    data_melted['Date'] = data_melted['Date'].astype('datetime64[ns]')

    return data_melted


def find_page(data_melted, page_name):

    my_page = data_melted[data_melted.Page == page_name]
    my_page = my_page.groupby('Date')[['count']].sum()
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


def plot_random_series(data_melted, n_series=5):

    titles = list()
    pages = {}
    np.random.seed(1)

    for i in range(n_series):
        titles.append(data_melted['Page']
                      [np.random.randint(0, len(data_melted))])

    sns.set()
    plt.figure(figsize=(14, 7))
    for i in range(n_series):
        pages[i] = find_page(data_melted, titles[i])
        plt.plot(pages[i], linewidth=1.7, label=titles[i])

    plt.legend()
    plt.show()

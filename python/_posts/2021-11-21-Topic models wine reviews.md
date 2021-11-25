---
layout: post
title: Time series topic models with wine review data
image:
  path: /assets/img/blog/maksym-kaharlytskyi-3uJt73tr4hI-unsplash.jpg
description: >
  Using Scikit-learn and additional Python code to study how the prevalence of certain topics changes over time
sitemap: false
---

Each blogpost in this series provides a fully working script which: i) opens and prepares a dataset; ii) runs a model; and iii) retrieves new, useful information from the model's output.

To run the script, you need a working Python programming environment. For this I strongly recommend [Anaconda](https://www.anaconda.com/). The remainder of this post assumes that you have Anaconda installed. Anaconda does not comes with the _Plotly_ and _pyLDAvis_ packages pre-installed, so you need to install these first for this specific script to work. [This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/) will help you to get started with Python, Anaconda and their various packages.  

## The case: reviews

Studying reviews longitudinally can help, for instance, to analyze how understandings of a product, or a whole group of products (e.g. wines, music, or movies), can change over time. The wine review data and the code below are used to demonstrate how such an analytical process in terms of Python code can look like (the actual results do not matter in this case).

However, for a meaningful example of shifting understandings, and how we can use topic models to study these, I refer to my paper "How disqualification leads to legitimation: dance music as a societal threat, legitimate leisure activity and established art in a British web of fields, 1985-2005" (available upon request). Drawing on an analysis of newspaper articles, the study shows and explains how distinct understandings of the British dance field were present over a 21 year period.

<!-- In a similar way, analyzing historical review data, on, for instance, movies, food, or consumer electronics, can reveal how the meanings of reviewed items may change. The data and code below is used purely for illustrative purposes, but it could be extended to other settings. -->


## The code

### 1. Open and prepare the dataset

Download the [original dataset](https://www.kaggle.com/zynicide/wine-reviews) from Kaggle and save it locally on your computer. This step uses some data wrangling, for example, to extract a date from the title of the review.


```python
# Read the .CSV as a dataframe
import os
corpus_path = 'C:/Users/User/Downloads/winemag data' #change this to the location where the data are saved
os.chdir(corpus_path)
import pandas as pd
df = pd.read_csv("winemag-data-130k-v2.csv", encoding='UTF-8')
df.reset_index(level=0, inplace=True)

# Extract the production year of the wine from the "title" collumn
# \d{4} is a pattern that matches with four digit numbers (which is useful to extract years from text)
df['date'] = df['title'].str.extract('(\d{4})', expand=True)
# Convert this string to a datevariable
df['datetime']  = pd.to_datetime(df['date'], errors = 'coerce')
# Add a count (this will be useful later when making the graphs)
df['count'] = 1

# Limit the data to all reviews concerning wines from 1990
df = df[(df['datetime'] > '1989-12-31')]
# Keep first 2000 rows to speed up the topic modeling
df = df[:2000]
```

### Run the model

The code below uses an LDA topic model from Scikit-Learn. It creates a plot using pyLDAvis.

```python
# Import necessary packages and such
from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# import warnings
# warnings.filterwarnings('ignore')
# only use this when you know the script and want to suppress unnecessary warnings

# Apply a count vectorizer to the data
# The run time of this cell is rather quick
tf_vectorizer = CountVectorizer(lowercase = True,
                                         strip_accents = 'unicode',
                                         stop_words = 'english',
                                         token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                                         max_df = 0.5, # ignore words occuring in > 50 % of the corpus (i.e. corpus specific stop words)
                                         min_df = 10) # ignore words in <10 documents of the corpus
dtm_tf = tf_vectorizer.fit_transform(df['description'].values.astype('U')) # import articles from df 'content' as unicode string
print(dtm_tf.shape)

# run a LDA model with 10 topics
lda_tf = LatentDirichletAllocation(n_components=5, random_state=0)
lda_tf.fit(dtm_tf)

# Print the topics in a conventional way
n_top_words = 30

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda_tf, tf_feature_names, n_top_words)
```

Output:

* Topic #0: flavors wine apple finish citrus acidity fruit fresh palate aromas green lemon pear crisp pineapple notes nose clean peach lime light vanilla chardonnay white grapefruit texture melon dry sweet honey

* Topic #1: wine drink fruit acidity tannins ripe fruits rich character flavors fruity structure black wood ready years texture red soft juicy aging age firm dense fresh crisp currant dry spice structured

* Topic #2: palate flavors white wine aromas acidity fruit dry pinot finish offers bright fresh cherry ripe mineral peach red elegant drink notes savory soft spice light stone hint noir long easy

* Topic #3: black cherry flavors aromas tannins palate fruit wine finish oak berry plum notes spice chocolate dark drink blackberry red ripe offers licorice nose tobacco cassis vanilla firm dried coffee pepper

* Topic #4: flavors aromas cabernet blend palate fruit finish cherry red sauvignon nose black merlot syrah berry pepper plum shows wine notes spice dried herbal blackberry franc spicy earthy cranberry tart light

These topics appear to be pretty similar, which is not so surprising. Yet, topic #0 appears to relate more to a citrusy range of flavors (associated with white wines), whereas topic #2 covers more a spicy range of flavors (associated with red wines).

### 3. Retrieve information from the model

This is the critical step. Several loops are used to calculate new metrics from the so-called "doc-topic" matrix, which are then turned into a time series dataset. Finally, I used the Plotly package for graphing.

```python
# create a doc-topic matrix
path = 'C:/Users/User/Desktop' # Change this path to a preferred location on your computer
os.chdir(path)

import numpy as np

filenames = df['index'].values.astype('U')

dates = df['date'].values.astype('U') # its better to use the date string here

dtm_transformed = tf_vectorizer.fit_transform(df['description'].values.astype('U'))

doctopic = lda_tf.fit_transform(dtm_transformed)

doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

# Write doctopic to a csv file
os.chdir(path)  

# filenamesclean = [fn.split('\\')[-1] for fn in filenames]
i=0
with open('doctopic_wine.csv',mode='w') as fo:
    for rij in doctopic:
        fo.write('"'+filenames[i]+'"')
        fo.write(',')
        fo.write('"'+dates[i]+'"')
        fo.write(',')
        for kolom in rij:
            fo.write(str(kolom))
            fo.write(',')
        fo.write('\n')
        i+=1

dfm = pd.read_csv('C:/Users/User/Desktop/doctopic_wine.csv', header=None, index_col=False,
                  names = ["file", "date", "t_0", "t_1","t_2", "t_3", "t_4"]) # Again change the location of the file

# calculate mean, std, cutoff high, and cutoff low
dfm1 = dfm.describe().loc[['mean','std']]
dfm2 = dfm1.transpose()
dfm2['cutoff_low'] = dfm2['mean'] + dfm2['std']

# Drop first two rows
dfm2 = dfm2.iloc[2:]
dfm2.reset_index(level=0, inplace=True)

# get cutoff_low from dfm2
d = {}
for i, row in dfm2.iterrows():
    d['t_{}_cutoff_low'.format(i)] = dfm2.at[i,'cutoff_low']
print(d)

for column in dfm.columns[-5:]:
    dfm['{}_low'.format(column)]=dfm['{}'.format(column)].apply(lambda x: 1 if x> d['{}_cutoff_low'.format(column)] else 0)

dfm['datetime'] = pd.to_datetime(df['date'], errors = 'coerce')
dfm

# Create five topic model time series datasets
g_1 = dfm.set_index('datetime').resample('A-DEC')['t_0_low'].sum()
g_1 = g_1.reset_index()

g_2 = dfm.set_index('datetime').resample('A-DEC')['t_1_low'].sum()
g_2 = g_2.reset_index()

g_3 = dfm.set_index('datetime').resample('A-DEC')['t_2_low'].sum()
g_3 = g_3.reset_index()

g_4 = dfm.set_index('datetime').resample('A-DEC')['t_3_low'].sum()
g_4 = g_4.reset_index()

g_5 = dfm.set_index('datetime').resample('A-DEC')['t_4_low'].sum()
g_5 = g_5.reset_index()

# Merge
dfs = [g_1, g_2, g_3, g_4, g_5]
from functools import reduce
df_topic_year = reduce(lambda  left,right: pd.merge(left,right,on=['datetime'],
                                            how='left'), dfs)
# df_topic_year will be used for the plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

x = df_topic_year['datetime']

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_0_low'], name="Topic 0", opacity=0.7, line=dict(color='#3A405A', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_1_low'], name="Topic 1", opacity=0.7, line=dict(color='#99B2DD', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_2_low'], name="Topic 2", opacity=0.7, line=dict(color='#E9AFA3', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_3_low'], name="Topic 3", opacity=0.7, line=dict(color='#685044', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_4_low'], name="Topic 4", opacity=0.7, line=dict(color='#F9DEC9', width=2)),
    secondary_y=False)

fig.update_layout(showlegend=True,
    xaxis_rangeslider_visible=False,
    width=600,
    height=600)


fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(title_text="Year", showgrid=True, gridwidth=0.3, gridcolor='LightGrey')
fig.update_yaxes(title_text="# Wines reviewed", showgrid=True, gridwidth=0.3, gridcolor='LightGrey')
fig.show()
```

Output:

![wine](/assets/img/blog/wine.png)


Test output:

```html
<div>
    <a href="https://plotly.com/~renswilderom/305/" target="_blank" title="Disney" style="display: block; text-align: center;"><img src="https://plotly.com/~renswilderom/305.png" alt="Disney" style="max-width: 100%;width: 1200px;"  width="1200" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="renswilderom:305" src="https://plotly.com/embed.js" async></script>
</div>
~~~

## Sources

Photo by [Maksym Kaharlytskyi](https://unsplash.com/@qwitka).

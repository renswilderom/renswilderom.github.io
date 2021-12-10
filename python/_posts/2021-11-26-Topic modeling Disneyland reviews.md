---
layout: post
title: Producing time series data from topic models in Scikit-learn
image:
  path: /assets/img/blog/bastien-nvs-g3CR0UJ1CyM-unsplash.jpg
description: >
  Analyzing more than 12.000 Disneyland Paris Tripadvisor reviews to detect understandings of Disneyland as a "magical world" vs Disneyland as a "hassle."
sitemap: false
comments: false
---

Each blogpost in this series provides a fully working script which: i) opens and prepares a dataset; ii) runs a model; and iii) retrieves new, useful information from the model's output.

To run the script, you need a working Python programming environment. For this I strongly recommend [Anaconda](https://www.anaconda.com/). The remainder of this post assumes that you have Anaconda installed. Anaconda does not comes with the _Plotly_ package pre-installed (for interactive plots), so you need to install this one first for this specific script to work. [This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/) will help you to get started with Python, Anaconda and their various packages.  

## The case: a decade of Disneyland Paris Tripadvisor reviews

Studying reviews longitudinally can help, for instance, to analyze how understandings of particular product or leisure activity may change with time. This post works with more than 12.000 Tripadvisor reviews of Disneyland Paris, which can be downloaded from [Kaggle](https://www.kaggle.com/arushchillar/disneyland-reviews).

To my surprise, I actually found some amusing dynamics in themes that the reviews discussed (note that a "theme" corresponds in this case to a distinct "topic" which the LDA topic model produces), yet I will not go to too deep into interpreting the output of the topic model. The Disneyland Paris review data and the code below are used to demonstrate how such an analytical process in terms of Python code can look like.

For a theoretically meaningful example of shifting public understandings, and how we can use topic models to study these, I refer to a paper that I wrote with Giselinde Kuipers and Alex van Venrooij titled "How disqualification leads to legitimation: dance music as a societal threat, legitimate leisure activity and established art in a British web of fields, 1985-2005" (available upon request). Drawing on an analysis of _Guardian_ newspaper articles, the study demonstrates and explains how distinct frames of the British dance field were present over a 21 year period.


## The code

### 1. Open and prepare the dataset

Download the [original dataset](https://www.kaggle.com/arushchillar/disneyland-reviews) from Kaggle and save it locally on your computer. This step uses some data wrangling, for example, to extract the year of visit from a date collumn. After limiting the data to Disneyland Paris reviews and dropping observartions without a date, there are more than 12.000 reviews left to analyze. This is a great quantity for topic models.


```python
# Read the .CSV as a dataframe
import os
corpus_path = 'C:/Users/User/Downloads/DisneylandReviews.csv'  # Change this to the preferred or relevant location on your computer
os.chdir(corpus_path)
import pandas as pd
df = pd.read_csv("DisneylandReviews.csv", encoding='ISO-8859-1')
df.reset_index(level=0, inplace=True)

# Limit the data to Disneyland Paris
df1 = df[(df['Branch'] == 'Disneyland_Paris')]
# Drop rows if Year_Month is missing
df1 = df1[df1.Year_Month != 'missing']
df1.reset_index(level=0, inplace=True)

# Extract the year of the visit # \d{4} is a pattern that matches with four digit numbers (which is useful to extract years from text)
df1['year'] = df1['Year_Month'].str.extract('(\d{4})', expand=True)

# Convert this string to a datevariable
df1['datetime']  = pd.to_datetime(df1['year'], errors = 'coerce')

# Add a count (this will be useful later when making the graphs)
df1['count'] = 1
```

### 2. Run the model

The code below uses an LDA topic model from Scikit-Learn.

```python
# Import necessary packages and such
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# import warnings
# warnings.filterwarnings('ignore') # only use this when you know the script and want to supress unnecessary warnings

# Apply a count vectorizer to the data
# The run time of this cell is rather quick
tf_vectorizer = CountVectorizer(lowercase = True,
                                         strip_accents = 'unicode',
                                         stop_words = 'english',
                                         token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                                         max_df = 0.5, # ignore words occuring in > 50 % of the corpus (i.e. corpus specific stop words)
                                         min_df = 10) # ignore words in <10 documents of the corpus
dtm_tf = tf_vectorizer.fit_transform(df1['Review_Text'].values.astype('U')) # import articles from df 'content' as unicode string
print(dtm_tf.shape)

# run a LDA model with 10 topics
# This code snippet takes most time to run
lda_tf = LatentDirichletAllocation(n_components=10, random_state=0)
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
* Topic #0: mountain ride thunder space pirates closed big buzz star peter good pan caribbean really jones indiana lightyear went small coaster kids just tours world queues land great wars like roller
* Topic #1: time amazing great year magical loved place day old just kids visit disneyland really children went days fantastic parades fireworks christmas parade experience love family characters enjoyed fun worth night
* Topic #2: staff food good florida paris just experience disneyland service time parks magic poor closed really like place people visit french better expensive queues disappointed years times rude friendly quality toilets
* Topic #3: hotel food day just stayed time good breakfast meal got euros room parks went kids bus train did disneyland children village eat booked really expensive half station took days restaurant
* Topic #4: characters time meet parade queue princess queues day hours good mickey children character early worth princesses great long main hotel got just magic castle didn little did hour photos wait
* Topic #5: people just like disneyland time don children place smoking day paris really know money say make line didn think experience kids want smoke going parks thing staff things waiting way
* Topic #6: food day time long queues expensive attractions waiting kids wait fast really good lot minutes closed place visit days went shops disneyland great lines fun just pass people queue worth
* Topic #7: time fast food day pass water good days use great plan expensive times ride queues make kids early queue need parks drinks wait want passes don snacks drink eat best
* Topic #8: disneyland paris parks day visit time train ride like world studios just great florida fun orlando attractions walt california different castle better lines mountain trip experience good smaller really tickets
* Topic #9: ride queue time day pass people fast minutes staff got queues wait hour went just told did hours tickets children closed ticket times didn long way disabled open waiting queuing

For me these topics are really great. I expect that many people are familiar with the experience of leisure activities as absolute wonder (e.g. see topic #1) versus leisure activities as moments of pain, resulting in enduring traumas. Scanning over the top terms of topic #9, I can imagine some of the horrors that initially enthusiastic Disneyland visitors were going through. This study can be read as a warning!

<p align="center">
<img src="/assets/img/blog/bastien-nvs-a4UVioeQGGY-unsplash.jpg" alt="disney" width="400" style="padding-top: 15px;"/>
</p>

### 3. Retrieve information from the model

This is the critical step. Several loops and a lambda function are used to calculate new metrics from the so-called "doc-topic" matrix, which are then turned into a time series dataset. Finally, I used the Plotly package for graphing.

```python
# create a doc-topic matrix
path = 'C:/Users/User/Desktop' # Change this to the preferred or relevant location on your computer
os.chdir(path)

import numpy as np

filenames = df1['index'].values.astype('U')

dates = df1['year'].values.astype('U') # its better to use the date string here

dtm_transformed = tf_vectorizer.fit_transform(df1['Review_Text'].values.astype('U'))

doctopic = lda_tf.fit_transform(dtm_transformed)

doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

# Write doctopic to a csv file
# I need to thank Damian Trilling for helping me with the code to turn the "doctopic" into a readable .CSV
os.chdir(path)  

i=0
with open('disney.csv',mode='w') as fo:
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
print("finsihed with creating doctopic matrix")

dfm = pd.read_csv('C:/Users/User/Desktop/disney.csv', header=None, index_col=False,
                  names = ["file", "year", "t_0", "t_1","t_2", "t_3", "t_4", "t_5", "t_6", "t_7", "t_8", "t_9"])

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

for column in dfm.columns[-10:]:
    dfm['{}_low'.format(column)]=dfm['{}'.format(column)].apply(lambda x: 1 if x> d['{}_cutoff_low'.format(column)] else 0)

dfm['datetime'] = pd.to_datetime(df1['year'], errors = 'coerce')

# Create 10 topic model time series datasets
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

g_6 = dfm.set_index('datetime').resample('A-DEC')['t_5_low'].sum()
g_6 = g_6.reset_index()

g_7 = dfm.set_index('datetime').resample('A-DEC')['t_6_low'].sum()
g_7 = g_7.reset_index()

g_8 = dfm.set_index('datetime').resample('A-DEC')['t_7_low'].sum()
g_8 = g_8.reset_index()

g_9 = dfm.set_index('datetime').resample('A-DEC')['t_8_low'].sum()
g_9 = g_9.reset_index()

g_10 = dfm.set_index('datetime').resample('A-DEC')['t_9_low'].sum()
g_10 = g_10.reset_index()

# Merge
dfs = [g_1, g_2, g_3, g_4, g_5, g_6, g_7, g_8, g_9, g_10]
from functools import reduce
df_topic_year = reduce(lambda  left,right: pd.merge(left,right,on=['datetime'],
                                            how='left'), dfs)

# Plotly graph with topic year data
# The code of this graph allows the use of two y axis (dual axis), but we will not use that here
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

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_5_low'], name="Topic 5", opacity=0.7, line=dict(color='#CE4257', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_6_low'], name="Topic 6", opacity=0.7, line=dict(color='#45CB85', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_7_low'], name="Topic 7", opacity=0.7, line=dict(color='#6FFFE9', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_8_low'], name="Topic 8", opacity=0.7, line=dict(color='#698F3F', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_9_low'], name="Topic 9", opacity=0.7, line=dict(color='#EF626C', width=2)),
    secondary_y=False)


fig.update_layout(showlegend=True,
    xaxis_rangeslider_visible=False,
    width=800,
    height=400)    


fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(title_text="Year", showgrid=True, gridwidth=0.3, gridcolor='LightGrey')
fig.update_yaxes(title_text="# Reviews", showgrid=True, gridwidth=0.3, gridcolor='LightGrey')
# fig.update_yaxes(title_text="Something", showgrid=False, secondary_y=True)
# Uncomment the line above for a dual axis graph
fig.show()
```

This gives:

<iframe width="750" height="500" frameborder="0" scrolling="no" src="//plotly.com/~renswilderom/305.embed?link=false"></iframe>

Keeping in mind that the main purpose of this post is to demonstrate how discursive trends can be investigated with topic models, we may tentatively try to interpret what happens in this graph. It appears that the salience of the "Disney as a magical world" theme (topic #1) won terrain compared to the "Disneyland as a hassle" theme (topic #9). Topic #3, which we may interpret as a "practical issues" theme (given terms such as "food," "meal," "bus," and "train"), started as the most dominant review topic, yet, after 2015, it became less salient compared to the rest.

Such findings usually become meaningful when researchers have in-depth knowledge about the case and complement (or "triangulate") the topic models by other methods, such qualitative analyses of the associated documents, through interviews, or even participant observation. For instance, in the study of newspaper coverage of the British dance field, referred to above, we qualitatively analyzed the top-100 articles associated with each topic included in the analysis. In addition, we collected publically available policy documents, and had a good understandings of the main events that took place (e.g. which new laws were constituted). This allowed us to unveil an interplay between discursive dynamics, on the one hand, and people's actions, on the other hand.      


## Sources

Photos by [Bastien Nvs](https://unsplash.com/@bastien_nvs).

---
layout: post
title: BERTopic actor-style matrix
image:
  path: /assets/img/blog/jakob-owens-ntqaFfrDdEA-unsplash-scaled.jpg
description: >
  Linking film styles to 'core crew' members with BERTopic
sitemap: false
comments: false
---

This blogpost provides a fully working script to: i) open and prepare a dataset; ii) run a model; and iii) retrieve new, useful information from the model's output.

To start your own "actor-style" project, you need a working Python programming environment. For this I recommend [Anaconda](https://www.anaconda.com/){:target="_blank"}, which makes Python programming easier (it comes with many packages pre-installed, helps you to install packages, manage package dependencies, and it includes Jupyter Notebooks, among other useful programs). The remainder of this post assumes that you are working with Anaconda and Jupyter Notebooks. [This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/){:target="_blank"} will help you to get started with Anaconda, the installation of packages, and Jupyter Notebooks.

In addition, you need to install the following extra package(s) for the script below to work:
> bertopic


## The case: Who’s creating which film styles?

Imagine a director of horror films who develops the innovative idea to produce a "zombie" film. The title resonates with audiences and becomes a success, so more directors follow and more zombie movies are released. At one point, however, audiences would be bored with yet another film about brain-dead walkers. The market for this stylistic variation is saturated and the production of such films declines. Audiences are now looking for something new, and there is an opportunity for a new stylistic variation to emerge. 

In the light of such a "cultural endogneous" dynamic (Van Venrooij, 2015; Godart and Galunic, 2019; Sgourev, Aadland, and Formilan, 2023), it is interesting to ask who is associated with the production of which film styles? Insights into this question could help us to understand how, for instance, film makers' network positions is shaped by such recurring style dynamics. Ultimately, this can help us to better comprehend the market-shaping effects of style and the performative effects of culture more generally. 

## The code

### 1. Open the dataset 

Download a [sample dataset](https://drive.google.com/file/d/1rtlzCniBY5g-wCNmyZf0DrSaIrccPWND/view?usp=sharing){:target="_blank"} and save it locally on your computer. This dataset contains a sample of 3000 horror films released in the US. The variables included are:
- year
- title
- producers_list, directors_list, writers_list, editing_list, cinematography_list, production_design_list and music_departments_list, who together can be considered as the 'core crew' (see Cattani and Ferriani, 2009)
- keywords_list


```python
import pandas as pd
df = pd.read_csv('C:/Users/bwilder1/Downloads/film_keywords.csv') 
# change the file path to the location where the data is stored on your computer
print(df.shape)
df.head()

```

The code cell below uses some data wrangling to create a clean 'core crew' column. 

```python
# These will be the 'actors' in the actor-style matrix
df['core_crew'] = df['producers_list'] + df['directors_list'] + df['writers_list'] + df['editing_list'] + df['cinematography_list'] + df['production_design_list'] + df['music_departments_list']

# Various consecutive cleaning steps to clean up the string
df["core_crew"] = (
    df["core_crew"]
    .str.replace('"', '', regex=False)
    .str.replace("'][", ',', regex=False)
    .str.replace("[", '', regex=False)
    .str.replace("]", '', regex=False)
    .str.replace("'", '', regex=False)
    .str.rstrip(',')
)

# Remove rows with empty lists (strings of only two characters [])
import numpy as np
df['len'] = df['core_crew'].apply(len)
df = df.loc[df['len'] != 0]

# Turn again into a list
df['core_crew'] = df['core_crew'].str.split(',')

# Keep row with 2 or more core crew members
import numpy as np
df['len_core_crew'] = df['core_crew'].apply(len)
df = df.loc[df['len_core_crew'] >= 2]

```

### 2. Create a new BERTopic model and visualize it 

This section draws on the code from the [BERT documentation](https://maartengr.github.io/BERTopic/getting_started/visualization/visualize_hierarchy.html){:target="_blank"}. 


```python
# %%time

# Create a new model
# This model uses a CountVectorizer
# calculate_probabilities=False to speed up the process
# min_topic_size= can be decreased (e.g. to 6) for smaller datasets. By default it is 10. 

# Turn the column keywords_list into a list
doc_list = df['keywords_list'].tolist()

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english")
topic_model = BERTopic(vectorizer_model=vectorizer_model, calculate_probabilities=False, min_topic_size=6)

topics, probs = topic_model.fit_transform(doc_list)

# Optional: load an existing model
# from bertopic import BERTopic
# topic_model = BERTopic.load("path to and name of model") # adjust path and name 
```

```python
# Get the descriptives as a topic model table
topic_df = topic_model.get_topic_info()
n_topics = topic_df.shape[0] 
n_obs = df.shape[0] 
topic_df.head()
```

<p align="center">
<img src="/assets/img/blog/Screenshot 2024-08-08 135618.png" alt="hierarchical" width="800" style="padding-top: 15px;"/>
</p>


Now you can visualize the topics with an intertopic distance map and a hierarchical cluster graph

```python
# Intertopic distance map 
topic_model.visualize_topics()
```

<p align="center">
<img src="/assets/img/blog/newplot (12).png" alt="distance" width="800" style="padding-top: 15px;"/>
</p>

```python
# Hierarchichal clustering
hierarchical_topics = topic_model.hierarchical_topics(doc_list)
fig = topic_model.visualize_hierarchy()
fig
```

<p align="center">
<img src="/assets/img/blog/newplot (14).png" alt="hierarchical" width="800" style="padding-top: 15px;"/>
</p>


### 3. Unpacking the clusters

This section uses the data from the hierarchical cluster matrix generated by BERT. Film titles are linked to specific topics, which are then associated with broader style clusters (i.e., groups of related topics). The data is aggregated yearly to produce time series data for these style clusters.

We begin by using the hierarchical_topics dataframe as input, filtering out all topics/clusters with a cosine distance greater than or equal to 1. The cosine distances for each cluster is visible in the Hierarchical cluster figure above. Next, we iterate through the remaining clusters, starting with those having the highest distance values. Finally, we remove topics from the list to ensure that each topic is only included in one cluster. The distance variable represents the cosine similarity score between topic keywords.


```python
# Compute the hierachical topics
hierarchical_topics = hierarchical_topics[hierarchical_topics['Distance']<1]  # start from distance < 1
hierarchical_topics = hierarchical_topics.reset_index()  # make sure indexes pair with the number of rows
collected=[]
clusters=[]
for index, row in hierarchical_topics.iterrows():
    if (len([i for i in row['Topics'] if i in collected])==0):
        collected.extend(row['Topics'])
        clusters.append([row['Topics'], row['Parent_Name']])        
index = 0
for cluster in clusters:
    print(index, cluster)
    index += 1
```

The code below creates a doc-topic matrix. It shows the probability of each film title being associated with specific topics, indicating the strength of these associations. It is used to determine a cutoff point, identifying whether a film title is "strongly associated" with a particular topic. A "strong" association is defined as a topic probability that exceeds 2 standard deviations (SD) above the mean probability.

```python
# Get the doc-topic matrix
topic_distr, _ = topic_model.approximate_distribution(doc_list)
doctopic=pd.DataFrame(topic_distr)
doctopic
```

```python
# Calculate the cutoff points
dfm1 = doctopic.describe().loc[['mean','std']]
dfm2 = dfm1.transpose()
dfm2['cutoff_high'] = dfm2['mean'] + dfm2['std'] + dfm2['std']
# dfm2 = dfm2.iloc[:-1]
dfm2.reset_index(level=0, inplace=True)
num_topics=len(dfm2)
dfm2
```

```python
# Build a dictionary with cutoff points and the column name to make it easier to apply to the dataframe
d = {}
for i, row in dfm2.iterrows():
    d['{}_cutoff_high'.format(i)] = dfm2.at[i,'cutoff_high']
    
# Use a lambda function to apply the low cutof (the -1 is to exclude the year column)
import warnings
warnings.filterwarnings('ignore')
for column in doctopic.columns:
    doctopic['{}_high'.format(column)]=doctopic[column].apply(lambda x: 1 if x> d['{}_cutoff_high'.format(column)] else 0)
doctopic
```

```python
# Create a seperate dataframe for the 1/0 doc-topic matrix 
doctopic_norm=doctopic.iloc[:, num_topics:]

# Add the clusters
cluster_doctopic_norm = doctopic_norm.copy()

for i in range(len(clusters)):
    cols=[str(x)+"_high" for x in clusters[i][0]]
    cluster_col=cluster_doctopic_norm[cols].sum(axis=1) # sum topics for relevant cluster
    cluster_doctopic_norm['c_{}'.format(i)] = cluster_col

cluster_doctopic_norm = cluster_doctopic_norm.reset_index(drop=False)
cluster_doctopic_norm.head() 
```

```python
# Create a dataframe with clusters only (so omitting the topics)
temp = cluster_doctopic_norm.iloc[:, -len(clusters):]

# Here we turn all the values larger than 1 into 1. 
# Hence, a title is either associated with a cluster or not. Titles can can associated with more than 1 cluster.
temp = temp.where(temp == 0, 1)
temp.head()
```

```python
# Add datetime variable
year = df['year'].values
temp.insert(0, 'year', year)
datetime = pd.to_datetime(temp['year'], errors = 'coerce', format = "%Y")
temp.insert(0, 'datetime', datetime) 
temp = temp.drop('year', axis=1)
# Add title and keywords
title = df['title'].values
temp.insert(1, 'title', title)
keywords_list = df['keywords_list'].values
temp.insert(2, 'keywords_list', keywords_list)
temp.head()
```

```python
# Create time series data
# Group clusters by datetime/year (with the temp dataframe as input)
timeseries={}
for column in temp.columns[1:]:
    timeseries[column]=temp.set_index('datetime').resample('A-DEC')[column].sum()
    
topic_timeseries_df=pd.DataFrame.from_dict(timeseries)
topic_timeseries_df = topic_timeseries_df.reset_index()
topic_timeseries_df = topic_timeseries_df.drop(topic_timeseries_df.iloc[:, 1:3], axis=1) # drop title and keywords
topic_timeseries_df
```


### 4. The actor-style matrix

```python
# Add the "core_crew_list" column
actors_matrix=temp.copy()
core_crew_list=df['core_crew']
actors_matrix['core_crew_list'] = core_crew_list.reset_index().drop('index', axis=1)
actors_matrix.head(2)
```

```python
# Here, the core_crew list is "exploded" meaning that each member and placed on a seperate row 
# Duplicates do not have to be removed, since actors are grouped by name and date
exploded_actors_df = actors_matrix.explode('core_crew_list')
exploded_actors_df['core_crew_list'] = exploded_actors_df['core_crew_list'].str.strip() # Strip whitespace to avoid "unique actors" based on a whitespace
print(exploded_actors_df.shape)
exploded_actors_df.head(2)
```

```python
sum_df = exploded_actors_df.groupby(['core_crew_list', 'datetime']).sum().reset_index()
pivot_df = sum_df.pivot(index='core_crew_list', columns='datetime')
pivot_df = pivot_df.fillna(0)
pivot_df
```

<p align="center">
<img src="/assets/img/blog/Screenshot 2024-08-08 135157.png" alt="hierarchical" width="800" style="padding-top: 15px;"/>
</p>



## References

Cattani, G., & Ferianni, S. (2008). A Core/Periphery Perspective on Individual Creative Performance. _Organization Science_, 19(6), 824—844.

Godart, F.C., & Galunic, C. (2019). Explaining the Popularity of Cultural Elements: Networks, Culture, and the Structural Embeddedness of High Fashion Trends. _Organization Science_, 30(1), 151-168.

Sgourev, S., Aadland, E., & Formilan, G. (2023). Relations in Aesthetic Space: How Color Enables Market Positioning. _Administrative Science Quarterly_, 68(1), 146-185.

Van Venrooij, A.T. (2015). A Community Ecology of Genres. _Poetics_, 52, 104-123.



## Other sources

Photos by [Jakob Owens](https://unsplash.com/@jakobowens1){:target="_blank"}.

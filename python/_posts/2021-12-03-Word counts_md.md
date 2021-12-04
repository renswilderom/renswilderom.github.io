---
layout: notebook
title: Word counts
image:
  path: /assets/img/blog/bastien-nvs-g3CR0UJ1CyM-unsplash.jpg
description: >
  Using Scikit-learn and additional Python code to study how the prevalence of certain topics changes over time
sitemap: false
---

## 1. Open and prepare the dataset


```python
# Read the .CSV file as a dataframe
import os
corpus_path = 'C:/Users/User/Downloads/DisneylandReviews.csv'  # Change this path to the preferred/relevant location on your computer
os.chdir(corpus_path)

import warnings
warnings.filterwarnings('ignore') # only use this when you know the script and want to supress unnecessary warnings

import pandas as pd
df = pd.read_csv("DisneylandReviews.csv", encoding='ISO-8859-1')
df.reset_index(level=0, inplace=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Review_ID</th>
      <th>Rating</th>
      <th>Year_Month</th>
      <th>Reviewer_Location</th>
      <th>Review_Text</th>
      <th>Branch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>670772142</td>
      <td>4</td>
      <td>2019-4</td>
      <td>Australia</td>
      <td>If you've ever been to Disneyland anywhere you...</td>
      <td>Disneyland_HongKong</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>670682799</td>
      <td>4</td>
      <td>2019-5</td>
      <td>Philippines</td>
      <td>Its been a while since d last time we visit HK...</td>
      <td>Disneyland_HongKong</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>670623270</td>
      <td>4</td>
      <td>2019-4</td>
      <td>United Arab Emirates</td>
      <td>Thanks God it wasn   t too hot or too humid wh...</td>
      <td>Disneyland_HongKong</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>670607911</td>
      <td>4</td>
      <td>2019-4</td>
      <td>Australia</td>
      <td>HK Disneyland is a great compact park. Unfortu...</td>
      <td>Disneyland_HongKong</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>670607296</td>
      <td>4</td>
      <td>2019-4</td>
      <td>United Kingdom</td>
      <td>the location is not in the city, took around 1...</td>
      <td>Disneyland_HongKong</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>42651</th>
      <td>42651</td>
      <td>1765031</td>
      <td>5</td>
      <td>missing</td>
      <td>United Kingdom</td>
      <td>i went to disneyland paris in july 03 and thou...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>42652</th>
      <td>42652</td>
      <td>1659553</td>
      <td>5</td>
      <td>missing</td>
      <td>Canada</td>
      <td>2 adults and 1 child of 11 visited Disneyland ...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>42653</th>
      <td>42653</td>
      <td>1645894</td>
      <td>5</td>
      <td>missing</td>
      <td>South Africa</td>
      <td>My eleven year old daughter and myself went to...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>42654</th>
      <td>42654</td>
      <td>1618637</td>
      <td>4</td>
      <td>missing</td>
      <td>United States</td>
      <td>This hotel, part of the Disneyland Paris compl...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>42655</th>
      <td>42655</td>
      <td>1536786</td>
      <td>4</td>
      <td>missing</td>
      <td>United Kingdom</td>
      <td>I went to the Disneyparis resort, in 1996, wit...</td>
      <td>Disneyland_Paris</td>
    </tr>
  </tbody>
</table>
<p>42656 rows × 7 columns</p>
</div>




```python
# Limit the data to Disneyland Paris
df = df[(df['Branch'] == 'Disneyland_Paris')]
# Drop rows if Year_Month is missing
df = df[df.Year_Month != 'missing']
df.reset_index(level=0, inplace=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>level_0</th>
      <th>index</th>
      <th>Review_ID</th>
      <th>Rating</th>
      <th>Year_Month</th>
      <th>Reviewer_Location</th>
      <th>Review_Text</th>
      <th>Branch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29026</td>
      <td>29026</td>
      <td>670721950</td>
      <td>5</td>
      <td>2019-3</td>
      <td>United Arab Emirates</td>
      <td>We've been to Disneyland Hongkong and Tokyo, s...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29027</td>
      <td>29027</td>
      <td>670686565</td>
      <td>4</td>
      <td>2018-6</td>
      <td>United Kingdom</td>
      <td>I went to Disneyland Paris in April 2018 on Ea...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29028</td>
      <td>29028</td>
      <td>670606796</td>
      <td>5</td>
      <td>2019-4</td>
      <td>United Kingdom</td>
      <td>What a fantastic place, the queues were decent...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29029</td>
      <td>29029</td>
      <td>670586937</td>
      <td>4</td>
      <td>2019-4</td>
      <td>Australia</td>
      <td>We didn't realise it was school holidays when ...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29031</td>
      <td>29031</td>
      <td>670400930</td>
      <td>5</td>
      <td>2019-4</td>
      <td>United Kingdom</td>
      <td>Such a magical experience. I recommend making ...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12689</th>
      <td>42113</td>
      <td>42113</td>
      <td>92198076</td>
      <td>4</td>
      <td>2011-1</td>
      <td>United Kingdom</td>
      <td>Although our pick up was prompt the taxi drive...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>12690</th>
      <td>42114</td>
      <td>42114</td>
      <td>92061774</td>
      <td>4</td>
      <td>2011-1</td>
      <td>Germany</td>
      <td>Just returned from a 4 days family trip to Dis...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>12691</th>
      <td>42115</td>
      <td>42115</td>
      <td>91995748</td>
      <td>1</td>
      <td>2010-12</td>
      <td>United Kingdom</td>
      <td>We spent the 20 Dec 2010 in the Disney park an...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>12692</th>
      <td>42116</td>
      <td>42116</td>
      <td>91984642</td>
      <td>2</td>
      <td>2010-12</td>
      <td>United Kingdom</td>
      <td>Well I was really looking forward to this trip...</td>
      <td>Disneyland_Paris</td>
    </tr>
    <tr>
      <th>12693</th>
      <td>42117</td>
      <td>42117</td>
      <td>91827418</td>
      <td>5</td>
      <td>2010-9</td>
      <td>United Kingdom</td>
      <td>If staying at a Disney hotel make good use of ...</td>
      <td>Disneyland_Paris</td>
    </tr>
  </tbody>
</table>
<p>12694 rows × 8 columns</p>
</div>




```python
# Extract the year of the visit # \d{4} is a pattern that matches with four digit numbers (which is useful to extract years from text)
df['year'] = df['Year_Month'].str.extract('(\d{4})', expand=True)

# Convert this string to a datevariable
df['datetime']  = pd.to_datetime(df['year'], errors = 'coerce')

# Add a count (this will be useful later when making the graphs)
df['count'] = 1
df

# Keep the columns that we need
df = df[['Review_Text','datetime', 'count']]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review_Text</th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>We've been to Disneyland Hongkong and Tokyo, s...</td>
      <td>2019-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I went to Disneyland Paris in April 2018 on Ea...</td>
      <td>2018-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What a fantastic place, the queues were decent...</td>
      <td>2019-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>We didn't realise it was school holidays when ...</td>
      <td>2019-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Such a magical experience. I recommend making ...</td>
      <td>2019-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12689</th>
      <td>Although our pick up was prompt the taxi drive...</td>
      <td>2011-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12690</th>
      <td>Just returned from a 4 days family trip to Dis...</td>
      <td>2011-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12691</th>
      <td>We spent the 20 Dec 2010 in the Disney park an...</td>
      <td>2010-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12692</th>
      <td>Well I was really looking forward to this trip...</td>
      <td>2010-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12693</th>
      <td>If staying at a Disney hotel make good use of ...</td>
      <td>2010-01-01</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>12694 rows × 3 columns</p>
</div>



You can off course stip these steps if you already have a dataframe with a text, datetime, and count variable column.

## 2. Getting the words counts


```python
# Word counts for 'expensive' per year (you can resample by year, month, or day,i.e. 'A-DEC', 'M', or 'D')
df['term_of_interest'] = df['Review_Text'].str.count('expensive')
df_word = df.set_index('datetime').resample('A-DEC')['term_of_interest'].sum()
df_word = df_word.reset_index()
print(df_word.sum())
df_word
```

    term_of_interest    2889
    dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>term_of_interest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-12-31</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-12-31</td>
      <td>207</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-12-31</td>
      <td>314</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-31</td>
      <td>489</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-12-31</td>
      <td>422</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-12-31</td>
      <td>438</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-12-31</td>
      <td>370</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-12-31</td>
      <td>301</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-12-31</td>
      <td>289</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019-12-31</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>



2013 has most references to 'expensive,' yet we also know that the total number of reviews per year can differ quite a bit. We can address this issue by assesing the average count of a term per review.


```python
# Get the total number of reviews per year
df_review = df.set_index('datetime').resample('A-DEC')['count'].sum()
df_review = df_review.reset_index()
print(df_review.sum())
df_review
```

    count    12694
    dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-12-31</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-12-31</td>
      <td>609</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-12-31</td>
      <td>1316</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-31</td>
      <td>1506</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-12-31</td>
      <td>1634</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-12-31</td>
      <td>2164</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-12-31</td>
      <td>1954</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-12-31</td>
      <td>1736</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-12-31</td>
      <td>1479</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019-12-31</td>
      <td>256</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge the two dataframes
# This can be done in a more simple way, but the method below allows you to merge more than two dataframesdfs = [df_word, df_review]
from functools import reduce
dfs = [df_word, df_review]
df_merge = reduce(lambda  left,right: pd.merge(left,right,on=['datetime'],
                                            how='left'), dfs)
df_merge
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>term_of_interest</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-12-31</td>
      <td>17</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-12-31</td>
      <td>207</td>
      <td>609</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-12-31</td>
      <td>314</td>
      <td>1316</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-31</td>
      <td>489</td>
      <td>1506</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-12-31</td>
      <td>422</td>
      <td>1634</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-12-31</td>
      <td>438</td>
      <td>2164</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-12-31</td>
      <td>370</td>
      <td>1954</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-12-31</td>
      <td>301</td>
      <td>1736</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-12-31</td>
      <td>289</td>
      <td>1479</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019-12-31</td>
      <td>42</td>
      <td>256</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Finally, we devide the variable 'term_of_interest' by 'count' (the total number of reviews)
df_merge['term/document'] = df_merge['term_of_interest']/df_merge['count']
df_merge
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>term_of_interest</th>
      <th>count</th>
      <th>term/document</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-12-31</td>
      <td>17</td>
      <td>40</td>
      <td>0.425000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-12-31</td>
      <td>207</td>
      <td>609</td>
      <td>0.339901</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-12-31</td>
      <td>314</td>
      <td>1316</td>
      <td>0.238602</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-31</td>
      <td>489</td>
      <td>1506</td>
      <td>0.324701</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-12-31</td>
      <td>422</td>
      <td>1634</td>
      <td>0.258262</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-12-31</td>
      <td>438</td>
      <td>2164</td>
      <td>0.202403</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-12-31</td>
      <td>370</td>
      <td>1954</td>
      <td>0.189355</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-12-31</td>
      <td>301</td>
      <td>1736</td>
      <td>0.173387</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-12-31</td>
      <td>289</td>
      <td>1479</td>
      <td>0.195402</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019-12-31</td>
      <td>42</td>
      <td>256</td>
      <td>0.164062</td>
    </tr>
  </tbody>
</table>
</div>



This shows that the relative use of the term of interest decreases with time

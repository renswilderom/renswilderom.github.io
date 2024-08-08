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

To start your own actor-style project, you need a working Python programming environment. For this I strongly recommend [Anaconda](https://www.anaconda.com/){:target="_blank"}, which makes Python programming easier (it comes with many packages pre-installed, helps you to install packages, manage package dependencies, and it includes Jupyter Notebooks, among other useful programs). The remainder of this post assumes that you are working with Anaconda and Jupyter Notebooks. [This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/){:target="_blank"} will help you to get there.

In addition, you need to install the following extra package(s) for the script below to work:
> bertopic


[This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/){:target="_blank"} will help you to get started with Anaconda, the installation of packages, and Jupyter Notebooks.


## The case: Who’s creating which film styles?

Imagine a director of horror films who develops the innovative idea to produce a "zombie" film. The title resonates with audiences and becomes a success, so more directors follow and more zombie movies are released. At one point, however, audiences would be bored with yet another film about brain-dead walkers. The market for this stylistic variation is saturated and the production of such films declines. Audiences are now looking for something new, and there is an opportunity for a new stylistic variation to emerge. 

In the light of such a "cultural endogneous" dynamic (Van Venrooij, 2015; Godart and Galunic, 2019; Sgourev, Aadland, and Formilan, 2023), it is interesting to ask who is associated with the production of which film styles? Insights into this question could help us to understand how, for instance, film makers' network positions is shaped by such recurring style dynamics. Ultimately, this can help us to better comprehend the market-shaping effects of style and the performative effects of culture more generally. 

...something about BERTopic


## The code

### 1. Open the dataset 

Download a [sample dataset](https://drive.google.com/file/d/1rtlzCniBY5g-wCNmyZf0DrSaIrccPWND/view?usp=sharing){:target="_blank"} and save it locally on your computer. 

This dataset contains a sample of 3000 horror films released in the US. The variables included are:
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

```python
# Create a clean 'core crew' column
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

This gives:
|    |   Topic |   Count | Name                                          | Representation                                                                                                                 |
|---:|--------:|--------:|:----------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
|  0 |      -1 |    1525 | -1_female_nudity_woman_film                   | ['female', 'nudity', 'woman', 'film', 'murder', 'death', 'man', 'sex', 'horror', 'killer']                                     |
|  1 |       0 |      81 | 0_creature_shark_feature_giant                | ['creature', 'shark', 'feature', 'giant', 'monster', 'sea', 'underwater', 'animal', 'attack', 'insect']                        |
|  2 |       1 |      68 | 1_alien_space_outer_planet                    | ['alien', 'space', 'outer', 'planet', 'spaceship', 'sci', 'fi', 'invasion', 'creature', 'spacecraft']                          |
|  3 |       2 |      58 | 2_psychotronic_film_independent_vacation      | ['psychotronic', 'film', 'independent', 'vacation', 'time', 'brain', 'travel', 'gone', 'wrong', 'anthology']                   |
|  4 |       3 |      47 | 3_slasher_sorority_teen_college               | ['slasher', 'sorority', 'teen', 'college', 'student', 'school', 'killer', 'girl', 'female', 'teenager']                        |
|  5 |       4 |      47 | 4_devil_demon_satanic_satan                   | ['devil', 'demon', 'satanic', 'satan', 'satanism', 'priest', 'demonic', 'catholic', 'supernatural', 'eclipse']                 |
|  6 |       5 |      46 | 5_serial_killer_slasher_film                  | ['serial', 'killer', 'slasher', 'film', 'police', 'independent', 'officer', 'cop', 'chevrolet', 'murder']                      |
|  7 |       6 |      46 | 6_independent_film_experimental_jackalope     | ['independent', 'film', 'experimental', 'jackalope', 'improvisation', 'shakespearean', 'dunne', 'post', 'andronicus', 'titus'] |
|  8 |       7 |      39 | 7_voodoo_zombie_film_haiti                    | ['voodoo', 'zombie', 'film', 'haiti', 'island', 'opera', 'cult', 'calypso', 'psychotronic', 'graveyard']                       |
|  9 |       8 |      39 | 8_ghost_haunted_haunting_house                | ['ghost', 'haunted', 'haunting', 'house', 'suspense', 'story', 'hampshire', 'spirit', 'attraction', 'true']                    |
| 10 |       9 |      36 | 9_zombie_apocalypse_survival_flesh            | ['zombie', 'apocalypse', 'survival', 'flesh', 'eating', 'outbreak', 'horror', 'sequel', 'undead', 'violence']|

_Note:_ row 11-74 are removed to save space. 

Now you can visualize the topics with an intertopic distance map and a hierarchical cluster graph

```python
# Intertopic distance map 
topic_model.visualize_topics()

```

```python

```



<p align="center">
<img src="/assets/img/blog/bastien-nvs-a4UVioeQGGY-unsplash.jpg" alt="disney" width="400" style="padding-top: 15px;"/>
</p>

### 3. Unpacking the clusters

...

```python

```

```python

```

```python

```

```python

```

```python

```

This gives:


<iframe width="700" height="400" frameborder="0" scrolling="no" src="//plotly.com/~renswilderom/305.embed?link=false"></iframe>



### 4. The actor-style matrix

```python

```

```python

```

```python

```

```python

```

```python

```


## References

Cattani, G., & Ferianni, S. (2008). A Core/Periphery Perspective on Individual Creative Performance. _Organization Science_, 19(6), 824—844.

Godart, F.C., & Galunic, C. (2019). Explaining the Popularity of Cultural Elements: Networks, Culture, and the Structural Embeddedness of High Fashion Trends. _Organization Science_, 30(1), 151-168.

Sgourev, S., Aadland, E., & Formilan, G. (2023). Relations in Aesthetic Space: How Color Enables Market Positioning. _Administrative Science Quarterly_, 68(1), 146-185.

Van Venrooij, A.T. (2015). A Community Ecology of Genres. _Poetics_, 52, 104-123.



## Other sources

Photos by [Jakob Owens](https://unsplash.com/@jakobowens1){:target="_blank"}.

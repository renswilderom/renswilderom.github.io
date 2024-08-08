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
| 10 |       9 |      36 | 9_zombie_apocalypse_survival_flesh            | ['zombie', 'apocalypse', 'survival', 'flesh', 'eating', 'outbreak', 'horror', 'sequel', 'undead', 'violence']                  |
| 11 |      10 |      35 | 10_vampire_blood_hybrid_male                  | ['vampire', 'blood', 'hybrid', 'male', 'versus', 'hunter', 'female', 'superhero', 'nudity', 'child']                           |
| 12 |      11 |      31 | 11_teen_slasher_teenager_teenage              | ['teen', 'slasher', 'teenager', 'teenage', 'school', 'axe', 'high', 'girl', 'horror', 'killer']                                |
| 13 |      12 |      30 | 12_vampire_lesbian_nudity_female              | ['vampire', 'lesbian', 'nudity', 'female', 'lesbianism', 'frontal', 'independent', 'erotica', 'comedy', 'rear']                |
| 14 |      13 |      30 | 13_werewolf_vampire_dracula_character         | ['werewolf', 'vampire', 'dracula', 'character', 'helsing', 'bite', 'london', 'reference', 'monster', 'bat']                    |
| 15 |      14 |      28 | 14_wife_relationship_husband_marriage         | ['wife', 'relationship', 'husband', 'marriage', 'psychotronic', 'love', 'film', 'doctor', 'mansion', 'house']                  |
| 16 |      15 |      28 | 15_killer_psycho_sadistic_serial              | ['killer', 'psycho', 'sadistic', 'serial', 'maniac', 'spree', 'psychopath', 'murderer', 'insane', 'terror']                    |
| 17 |      16 |      28 | 16_vampire_independent_film_blood             | ['vampire', 'independent', 'film', 'blood', 'trash', 'arizona', 'psychotronic', 'trailer', 'arts', 'campy']                    |
| 18 |      17 |      28 | 17_nudity_frontal_female_rear                 | ['nudity', 'frontal', 'female', 'rear', 'male', 'downward', 'independent', 'spiral', 'snuff', 'film']                          |
| 19 |      18 |      27 | 18_dog_animal_snake_rat                       | ['dog', 'animal', 'snake', 'rat', 'literature', 'screen', 'title', 'nature', 'attack', 'killed']                               |
| 20 |      19 |      26 | 19_sister_brother_daughter_family             | ['sister', 'brother', 'daughter', 'family', 'relationship', 'incest', 'father', 'mother', 'incestuous', 'girl']                |
| 21 |      20 |      25 | 20_bigfoot_sasquatch_creature_feature         | ['bigfoot', 'sasquatch', 'creature', 'feature', 'abominable', 'yeti', 'snowman', 'legend', 'monster', 'native']                |
| 22 |      21 |      24 | 21_children_party_student_school              | ['children', 'party', 'student', 'school', 'tradition', 'stars', 'apple', 'college', 'bartender', 'government']                |
| 23 |      22 |      24 | 22_shot_police_head_cigarette                 | ['shot', 'police', 'head', 'cigarette', 'neck', 'severed', 'reference', 'gun', 'blood', 'car']                                 |
| 24 |      23 |      23 | 23_fight_marine_destruction_obsession         | ['fight', 'marine', 'destruction', 'obsession', 'shooting', 'life', 'murder', 'shot', 'insane', 'killing']                     |
| 25 |      24 |      23 | 24_rape_sexual_revenge_panties                | ['rape', 'sexual', 'revenge', 'panties', 'sex', 'female', 'nudity', 'exploitation', 'misogynist', 'victim']                    |
| 26 |      25 |      22 | 25_paranormal_power_phenomenon_supernatural   | ['paranormal', 'power', 'phenomenon', 'supernatural', 'psychotronic', 'ghost', 'film', 'haunting', 'psychic', 'independent']   |
| 27 |      26 |      20 | 26_sex_pubic_nudity_frontal                   | ['sex', 'pubic', 'nudity', 'frontal', 'hair', 'male', 'sexual', 'female', 'simulated', 'explicit']                             |
| 28 |      27 |      19 | 27_title_word_character_unholiness            | ['title', 'word', 'character', 'unholiness', 'wailing', 'gaelic', 'miserable', 'caligari', 'marriage', 'nickname']             |
| 29 |      28 |      18 | 28_experiment_mad_doctor_deterioration        | ['experiment', 'mad', 'doctor', 'deterioration', 'personality', 'scientist', 'mr', 'dr', 'hyde', 'laboratory']                 |
| 30 |      29 |      17 | 29_zombie_independent_gore_undead             | ['zombie', 'independent', 'gore', 'undead', 'jesus', 'guts', 'pinky', 'hazard', 'pony', 'poole']                               |
| 31 |      30 |      17 | 30_woods_camping_slasher_backwoods            | ['woods', 'camping', 'slasher', 'backwoods', 'cabin', 'bear', 'campfire', 'nudity', 'forest', 'female']                        |
| 32 |      31 |      16 | 31_frankensteins_frankenstein_victor_monster  | ['frankensteins', 'frankenstein', 'victor', 'monster', 'character', 'universal', 'time', 'doctor', 'scientist', 'brain']       |
| 33 |      32 |      16 | 32_slasher_search_camping_biker               | ['slasher', 'search', 'camping', 'biker', 'anthology', 'ed', 'motorcycle', 'son', 'cult', '']                                  |
| 34 |      33 |      16 | 33_mummy_egyptian_egypt_tomb                  | ['mummy', 'egyptian', 'egypt', 'tomb', 'curse', 'archeologist', 'ancient', 'tana', 'egyptology', 'museum']                     |
| 35 |      34 |      16 | 34_robot_cyborg_sci_fi                        | ['robot', 'cyborg', 'sci', 'fi', 'superhero', 'robocop', 'arts', 'martial', 'future', 'alien']                                 |
| 36 |      35 |      15 | 35_zombie_walking_epic_bbq                    | ['zombie', 'walking', 'epic', 'bbq', 'holly', 'hillbilly', 'style', 'unlikely', 'undead', 'brains']                            |
| 37 |      36 |      14 | 36_panties_removes_female_nudity              | ['panties', 'removes', 'female', 'nudity', 'frontal', 'clothes', 'shower', 'scantily', 'clad', 'voyeur']                       |
| 38 |      37 |      14 | 37_mother_relationship_insanity_daughter      | ['mother', 'relationship', 'insanity', 'daughter', 'mental', 'killer', 'psychopath', 'wife', 'emotionally', 'woman']           |
| 39 |      38 |      14 | 38_grave_child_spider_ghost                   | ['grave', 'child', 'spider', 'ghost', 'doll', 'dead', 'lonely', 'girl', 'death', 'eyes']                                       |
| 40 |      39 |      14 | 39_delivery_ghost_house_comanche              | ['delivery', 'ghost', 'house', 'comanche', 'painting', 'spirit', 'son', 'haunted', 'portrait', 'haunting']                     |
| 41 |      40 |      13 | 40_vampire_stolen_cache_midwestern            | ['vampire', 'stolen', 'cache', 'midwestern', 'illuminati', 'goods', 'muscle', 'moonshine', 'weapons', 'mechanic']              |
| 42 |      41 |      13 | 41_punctuation_title_apostrophe_exclamation   | ['punctuation', 'title', 'apostrophe', 'exclamation', 'point', 'comma', 'pug', 'contraction', 'bunny', 'backwoods']            |
| 43 |      42 |      13 | 42_rock_humor_roll_older                      | ['rock', 'humor', 'roll', 'older', 'singer', 'star', 'tire', 'younger', 'band', 'flat']                                        |
| 44 |      43 |      13 | 43_dream_russian_sleep_nightmare              | ['dream', 'russian', 'sleep', 'nightmare', 'deprivation', 'hallucination', 'assassin', 'political', 'sequence', 'elm']         |
| 45 |      44 |      13 | 44_newspaper_italian_headline_letter          | ['newspaper', 'italian', 'headline', 'letter', 'murder', 'police', 'lawyer', 'usa', 'smoking', 'drink']                        |
| 46 |      45 |      12 | 45_turkey_meat_severed_cannibalism            | ['turkey', 'meat', 'severed', 'cannibalism', 'comedy', 'relationship', 'cannibal', 'talking', 'head', 'aerobics']              |
| 47 |      46 |      12 | 46_rated_comic_book_slasher                   | ['rated', 'comic', 'book', 'slasher', '', '', '', '', '', '']                                                                  |
| 48 |      47 |      11 | 47_superhero_frontal_nudity_softcore          | ['superhero', 'frontal', 'nudity', 'softcore', 'film', 'female', 'philippines', 'psychotronic', 'erotica', 'japan']            |
| 49 |      48 |      10 | 48_directed_rated_title_direction             | ['directed', 'rated', 'title', 'direction', 'cardinal', 'production', 'amateur', 'female', 'campy', 'character']               |
| 50 |      49 |      10 | 49_werewolf_episodic_york_blonde              | ['werewolf', 'episodic', 'york', 'blonde', 'city', 'horror', 'dog', 'bullet', 'new', 'mirror']                                 |
| 51 |      50 |      10 | 50_myers_michael_masked_halloween             | ['myers', 'michael', 'masked', 'halloween', 'killer', 'character', 'returning', 'laurie', 'strode', 'fictional']               |
| 52 |      51 |      10 | 51_witchcraft_magic_power_witch               | ['witchcraft', 'magic', 'power', 'witch', 'psychotronic', 'supernatural', 'telekinesis', 'levitation', 'film', 'independent']  |
| 53 |      52 |      10 | 52_relationships_family_veteran_murder        | ['relationships', 'family', 'veteran', 'murder', 'novel', 'guest', 'independent', 'antichrist', 'noose', 'weekend']            |
| 54 |      53 |       9 | 53_golf_comedy_spoof_parody                   | ['golf', 'comedy', 'spoof', 'parody', 'satire', 'tennessee', 'humor', 'graduation', 'slasher', 'dark']                         |
| 55 |      54 |       8 | 54_voorhees_murderer_jason_machete            | ['voorhees', 'murderer', 'jason', 'machete', 'sadistic', 'mysterious', 'masked', 'murdered', 'friday', 'hockey']               |
| 56 |      55 |       8 | 55_anthology_sov_creepshow_nosferatu          | ['anthology', 'sov', 'creepshow', 'nosferatu', 'meta', 'newspaper', 'book', 'low', 'budget', 'horror']                         |
| 57 |      56 |       8 | 56_baby_mutant_pregnancy_abortion             | ['baby', 'mutant', 'pregnancy', 'abortion', 'steampunk', 'toilet', 'fetus', 'cat', 'court', 'feeding']                         |
| 58 |      57 |       8 | 57_clown_injury_acid_evil                     | ['clown', 'injury', 'acid', 'evil', 'sheriff', 'blood', 'finger', 'director', 'head', 'quote']                                 |
| 59 |      58 |       8 | 58_bikini_breasts_1970s_nudity                | ['bikini', 'breasts', '1970s', 'nudity', 'fights', 'independent', 'female', 'sexual', 'island', 'film']                        |
| 60 |      59 |       7 | 59_murderer_masked_escaped_slasher            | ['murderer', 'masked', 'escaped', 'slasher', 'psychotic', '1989', 'slaughter', 'killer', 'mysterious', 'year']                 |
| 61 |      60 |       7 | 60_elevator_dead_woman_murder                 | ['elevator', 'dead', 'woman', 'murder', 'officer', 'open', 'nude', 'police', 'orphaned', 'floor']                              |
| 62 |      61 |       7 | 61_town_farm_basement_mansion                 | ['town', 'farm', 'basement', 'mansion', 'home', 'house', '', '', '', '']                                                       |
| 63 |      62 |       7 | 62_hp_lovecraft_works_based                   | ['hp', 'lovecraft', 'works', 'based', 'chambers', 'cthulhu', 'micro', 'production', 'lovecraftian', 'acting']                  |
| 64 |      63 |       7 | 63_christian_persecution_propaganda_religious | ['christian', 'persecution', 'propaganda', 'religious', 'religion', 'execution', 'christianity', 'christ', 'jesus', 'fanatic'] |
| 65 |      64 |       7 | 64_alien_dimension_sex_interspecies           | ['alien', 'dimension', 'sex', 'interspecies', 'raped', 'abduction', 'dildo', 'dna', 'sexual', 'masturbation']                  |
| 66 |      65 |       7 | 65_cowboy_modern_west_tent                    | ['cowboy', 'modern', 'west', 'tent', 'stagecoach', 'opium', 'lightning', 'lake', 'hockey', 'crystal']                          |
| 67 |      66 |       7 | 66_lovecraftian_paris_sculpture_reference     | ['lovecraftian', 'paris', 'sculpture', 'reference', 'university', 'hypnotized', 'cliffhanger', 'spilled', 'ending', 'crying']  |
| 68 |      67 |       7 | 67_apartment_husband_adultery_alcoholic       | ['apartment', 'husband', 'adultery', 'alcoholic', 'suicide', 'promiscuous', 'lover', 'marriage', 'psychic', 'wife']            |
| 69 |      68 |       7 | 68_genetic_scientist_screaming_engineering    | ['genetic', 'scientist', 'screaming', 'engineering', 'monster', 'female', 'breasts', 'experiment', 'pool', 'nudity']           |
| 70 |      69 |       6 | 69_organ_kidney_nazi_cloning                  | ['organ', 'kidney', 'nazi', 'cloning', 'harvesting', 'low', 'budget', 'monster', 'cult', 'transplant']                         |
| 71 |      70 |       6 | 70_host_anthology_racial_segregation          | ['host', 'anthology', 'racial', 'segregation', 'usa', 'horror', 'shot', 'drowning', 'driving', 'blaxploitation']               |
| 72 |      71 |       6 | 71_alien_grocery_shootout_store               | ['alien', 'grocery', 'shootout', 'store', 'los', 'angeles', 'urban', 'car', 'police', 'anti']                                  |
| 73 |      72 |       6 | 72_occult_sorcerer_ritual_satanic             | ['occult', 'sorcerer', 'ritual', 'satanic', 'magic', 'leader', 'supernatural', 'levitation', 'magician', 'cult']               |
| 74 |      73 |       6 | 73_nude_ski_scissors_stabbed                  | ['nude', 'ski', 'scissors', 'stabbed', 'impression', 'letter', 'water', 'woman', 'jacuzzi', 'killer']                          |

_Note:_ row 11-74 are removed to save space. 

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

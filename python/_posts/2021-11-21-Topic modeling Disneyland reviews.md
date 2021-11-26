---
layout: post
title: Disneyland as a magical world vs Disneyland as a hassle
image:
  path: /assets/img/blog/bastien-nvs-g3CR0UJ1CyM-unsplash.jpg
description: >
  Using Scikit-learn and additional Python code to study how the prevalence of certain topics changes over time
sitemap: false
---

Each blogpost in this series provides a fully working script which: i) opens and prepares a dataset; ii) runs a model; and iii) retrieves new, useful information from the model's output.

To run the script, you need a working Python programming environment. For this I strongly recommend [Anaconda](https://www.anaconda.com/). The remainder of this post assumes that you have Anaconda installed. Anaconda does not comes with the _Plotly_ package pre-installed (for interactive plots), so you need to install this one first for this specific script to work. [This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/) will help you to get started with Python, Anaconda and their various packages.  

## The case: a decade of Disneyland Paris Tripadvisor reviews

Studying reviews longitudinally can help, for instance, to analyze how understandings of particular a product or leisure activity can change over time. The Disneyland Paris review data and the code below are used to demonstrate how such an analytical process in terms of Python code can look like.

To my surprise, I actually found some amusing dynamics in themes that the reviews discussed. The two distinct review themes that are worth mentioning could be referred to as the "Disneyland as magical world" theme and the "Disneyland as a hassle" theme. But please keep in mind that these data are selected for illustrative purposes.

For a theoretically more meaningful example of how public understandings can change with time, and how we can use topic models to study these, I refer to my paper "How disqualification leads to legitimation: dance music as a societal threat, legitimate leisure activity and established art in a British web of fields, 1985-2005" (available upon request). Drawing on an analysis of newspaper articles, the study shows and explains how distinct frames of the British dance field were present over a 21 year period.


## The code

### 1. Open and prepare the dataset

Download the [original dataset](https://www.kaggle.com/arushchillar/disneyland-reviews) from Kaggle and save it locally on your computer. This step uses some data wrangling, for example, to extract the year of visit from a date collumn.


```python

```

### Run the model

The code below uses an LDA topic model from Scikit-Learn. It creates a plot using pyLDAvis.

```python

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

For me these topics are really great. I expect that many people are familiar with the experience of leisure activities as absolute wonder (e.g. see topic #1) versus leisure activities as moments of pain, resulting in enduring traumas. Scanning over the top terms of topic #3 or topic #9, I can imagine some of the horrors that initially enthusiastic Disneyland visitors were going through. This study could be read as a warning.

<p align="center">
<img src="/assets/img/blog/bastien-nvs-a4UVioeQGGY-unsplash.jpg" alt="disney" width="400"/>
</p>

### 3. Retrieve information from the model

This is the critical step. Several loops and a lambda function are used to calculate new metrics from the so-called "doc-topic" matrix, which are then turned into a time series dataset. Finally, I used the Plotly package for graphing.

```python

```

This gives:

<iframe width="700" height="400" frameborder="0" scrolling="no" src="//plotly.com/~renswilderom/305.embed?link=true"></iframe>

Keeping in mind that the main purpose of these data are to demonstrate the workings and potential use of a Python program, the graph quite clearly shows some trends. That is, the "Disneyland as a hassle" theme (topic #3) became with time less vocal in the landscape of Tripadvisor reviews, while the "Disney as a magical world" theme (topic #1) won terrain. Based on personal experience in the restaurant sector, I know that can managers take reviews quite seriously. So, we may hypotisize that Disneyland learned from its mistakes, and became better in offering a fun, worry-free experience :)


## Sources

Photos by [Bastien Nvs](https://unsplash.com/@bastien_nvs).

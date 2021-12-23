---
layout: post
title: How to get started with Python
image:
  path: /assets/img/blog/bonnie-kittle-GiIZSko7Guk-unsplash.jpg
description: >
  A friendly introduction to some powerful tools
sitemap: false
comments: false
---

This page will help you to start programming in Python. I will explain which software/platforms are both effective and convenient (regardless of how experienced you are). My main message is that programming in Anaconda is the way to go.

## Why Python coding in Anaconda will make your life easy (also if you are new to Python)

Anaconda is said to be the world's most popular data science platform. It allows you to write and execute Python programs and most of these work with several "packages" (small pieces of software). Anaconda comes with many essential packages pre-installed, it allows you to easily install additional packages, and it manages the dependencies between these packages. It does so automatically, so you don't need to worry about it.

Anaconda is open-source, and installing allows you also to use Jupyter Notebooks (which I will explain in a minute), Spyder (a conventional code editor), and R studio (which can be handy if you sometimes also need to program in R), among others.

Jupyter Notebooks are a data scientists' favorite. You write and run your programs/code snippets from a notebook in your browser. It consists of code cells, which can directly show the output of a code. For instance, we have a dataframe in which we like to limit the observations to those occurring before 1953. In one of the code cells, we type ```df = df[df['year'] <= 1952]``` and then inspect the transformed dataframe by calling ```df``` (see screenshot below). Now you can see whether the output is as intended. This makes "intuitive:" the results of your code become directly visible. This is also handy for intermediate checks (is your code still on the right track? Just print your dataframe and examine).

<p align="center">
<img src="/assets/img/blog/Screenshot_1.png" alt="jupyter" width="400" style="padding-top: 15px;"/>
</p>

You can download Anaconda through [this link](https://www.anaconda.com/){:target="_blank"}.


## Pythonistas with useful websites

* [Melanie Walsh](https://melaniewalsh.org/){:target="_blank"}, Assistant Teaching Professor in the iSchool at the University of Washington, created the [online text book](https://melaniewalsh.github.io/Intro-Cultural-Analytics/welcome.html){:target="_blank"} _Introduction to Cultural Analytics & Python._ Besides basic Python coding instructions, it also provides useful, well-organized information on text analysis and network analysis, among other things.  

* [Chris Albon](https://chrisalbon.com/){:target="_blank"}, Director of Machine Learning at the Wikimedia Foundation, has a website with instructive, easy to follow code examples, ranging from tasks such as [Select Rows With A Certain Value](https://chrisalbon.com/code/python/data_wrangling/pandas_select_rows_containing_values/){:target="_blank"} to running [Convolutional Neural Networks](https://chrisalbon.com/code/deep_learning/keras/convolutional_neural_network/){:target="_blank"} in Keras. Albon's website is the most extensive, high-quality Python resource I came across so far. I used it predominantly for data wrangling in Pandas, but it offers much more.  

<!-- Programming Historian (team of Pythonistas who keep each other sharp through peer-review processes). Lots of explaining (also little details that beginners may not know). Aimed at the Humanities (and in my view also highly suitable for social scientists)

Package documentation (some packages provide excellent documentation on how to use them, so also visit their sites).

Libaries for machine learning in Python
Scikit-learn (the one I am using)
Keras
Flair (by Zalando)
Tensorflow
Pytorch
Also see: https://research.zalando.com/post/tempflow/ -->
Photo by [Bonnie Kittle](https://unsplash.com/@bonniekdesign){:target="_blank"}.

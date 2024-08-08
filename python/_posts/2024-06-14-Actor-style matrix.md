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

To start your own topic modeling project, you need a working Python programming environment. For this I strongly recommend [Anaconda](https://www.anaconda.com/){:target="_blank"}, which makes Python programming easier (it comes with many packages pre-installed, helps you to install packages, manage package dependencies, and it includes Jupyter Notebooks, among other useful programs). The remainder of this post assumes that you are working with Anaconda and Jupyter Notebooks. [This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/){:target="_blank"} will help you to get there.

In addition, you need to install the following extra package(s) for the script below to work:
> bertopic


[This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/){:target="_blank"} will help you to get started with Anaconda, the installation of packages, and Jupyter Notebooks.


## The case: Who’s Creating Which Film Styles?

Imagine a director of horror films who develops the innovative idea to produce a "zombie" film. The title resonates with audiences and becomes a success, so more directors follow and more zombie movies are released. At one point, however, audiences would be bored with yet another film about brain-dead walkers. The market for this stylistic variation is saturated and the production of such films declines. Audiences are now looking for something new, and there is an opportunity for a new stylistic variation to emerge. 

In the light of such a "cultural endogneous" dynamic (Van Venrooij, 2015; Godart and Galunic, 2019; Sgourev, Aadland, and Formilan, 2023), it is interesting to ask who is associated with which film styles? Insights into this question could help us to understand how, for instance, film makers' network positions - and, ultimately, the organization of the film industry as a whole - is shaped by such recurring style dynamics. 


## The code

### 1. Open the dataset 

Download a [sample dataset](https://www.kaggle.com/arushchillar/disneyland-reviews){:target="_blank"} and save it locally on your computer. 


```python

```

### 2. Create a new BERTopic model and visualize it 

...

```python

```

```python

```

```python

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

Godart, F.C., & Galunic, C. (2019). Explaining the Popularity of Cultural Elements: Networks, Culture, and the Structural Embeddedness of High Fashion Trends. _Organization Science_, 30(1), 151-168.
Sgourev, S., Aadland, E., & Formilan, G. (2023). Relations in Aesthetic Space: How Color Enables Market Positioning. _Administrative Science Quarterly_, 68(1), 146-185.
Van Venrooij, A.T. (2015). A Community Ecology of Genres. _Poetics_, 52, 104-123.



## Other sources

Photos by [Jakob Owens](https://unsplash.com/@jakobowens1){:target="_blank"}.

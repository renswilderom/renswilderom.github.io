---
layout: post
title: Compute community-level metrics in social networks with NetworkX
image:
  path: /assets/img/blog/Jacques-Cousteau-Swimming.jpg
description: >
  Community detection in dolphin networks (and why social scientists may be interested).
sitemap: false
comments: false
---

This blogpost provides a fully working script which: i) opens and prepares a dataset; ii) runs a model; and iii) retrieves new, useful information from the model's output.

For your own network analysis project, you need a working Python programming environment. For this I strongly recommend [Anaconda](https://www.anaconda.com/){:target="_blank"}, which makes Python programming easier (it comes with many packages pre-installed, helps you to install packages, manage package dependencies, and it includes Jupyter Notebooks, among other useful programs) The remainder of this post assumes that you are working with Anaconda and Jupyter Notebooks.[This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/){:target="_blank"} will help you to get there.

In addition, you need to install the following extra package(s) for the script below to work:
> No additional packages are needed


[This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/){:target="_blank"} will help you to get started with Anaconda, the installation of packages, and Jupyter Notebooks.


## The case

### Community detection in dolphin networks

Social scientists may have some fun now and then, for instance, when renaming their interviewees to protect their anonymity. But, the real fun appears to be reserved for marine biologists, who can invent entirely new names for their research subjects, such as Bumber, Ripplefluke, and TR77. This post draws on the Dolphin dataset by Lusseau et al. (2003) to demonstrate how to unpack the output of community detection algorithms (see the graph below to get an idea of how much fun Lusseau and his team must have had).

Building on the Dolphin dataset, the NetworkX package in Python and existing community detection algorithms, my script below allows you to retrieve community-level metrics, such as density and transitivity, and add them to a Pandas dataframe. The result will look similar to this table (note that the code below adds a few extra variables to the dataframe):

|Community    |   Density |   Transitivity |
|---:|----------:|---------------:|
|  0 |  0.27619  |       0.484615 |
|  1 |  0.226316 |       0.42233  |
|  2 |  0.345455 |       0.482143 |
|  3 |  0.275    |       0.445161 |

<p align="center">
<img src="/assets/img/blog/dolphins.svg" alt="dolphins" width="500" style="padding-top: 15px;"/>
</p>

<sub>_Note:_ This graph is made using a slightly modified script taken from [this chapter](https://melaniewalsh.github.io/Intro-Cultural-Analytics/06-Network-Analysis/02-Making-Network-Viz-with-Bokeh.html ){:target="_blank"} in the formidable online text book _Introduction to Cultural Analytics & Python_. The method to produce such graphs will not be further discussed in this blogpost.</sub>


### Why may social scientists be interested?

Apart from marine biologists, social scientists may also like to compare the communities within a given network. For example, Uzzi and Spiro (2005) found that the cohesion within communities among Broadway musical creators matters for their creative success. In fact, they demonstrated that cohesion and creative success are characterized by an _inverted U-shaped relationship_. So, initially, more cohesion leads more creative success, but up to a certain threshold, when this is crossed, more cohesion will dampen creativity. Let us now turn to the program that can process the output of community detection algorithms.  

## The code

### 1. Open and prepare the dataset

Copy-paste the code below in your Python programming environment (e.g. a Jupyter Notebook). Here we create a new dataframe called ```df_dol```, which we will later turn into a graph object (a technical term for a network). These data are already "in shape," so we don't have to do any data wrangling.

```python
# Create the dataset
import pandas as pd
source = ['Beak', 'Beak', 'Beak', 'Beak', 'Beak', 'Beak', 'Beescratch', 'Beescratch', 'Beescratch', 'Beescratch', 'Beescratch', 'Beescratch', 'Beescratch', 'Beescratch', 'Bumper', 'Bumper', 'Bumper', 'Bumper', 'CCL', 'CCL', 'CCL', 'Cross', 'DN16', 'DN16', 'DN16', 'DN16', 'DN21', 'DN21', 'DN21', 'DN21', 'DN21', 'DN21', 'DN63', 'DN63', 'DN63', 'DN63', 'DN63', 'Double', 'Double', 'Double', 'Double', 'Double', 'Feather', 'Feather', 'Feather', 'Feather', 'Feather', 'Fish', 'Fish', 'Fish', 'Five', 'Fork', 'Gallatin', 'Gallatin', 'Gallatin', 'Gallatin', 'Gallatin', 'Grin', 'Grin', 'Grin', 'Grin', 'Grin', 'Grin', 'Grin', 'Grin', 'Grin', 'Grin', 'Haecksel', 'Haecksel', 'Haecksel', 'Haecksel', 'Haecksel', 'Haecksel', 'Hook', 'Hook', 'Hook', 'Hook', 'Hook', 'Jet', 'Jet', 'Jet', 'Jet', 'Jet', 'Jonah', 'Jonah', 'Jonah', 'Jonah', 'Jonah', 'Jonah', 'Knit', 'Knit', 'Kringel', 'Kringel', 'Kringel', 'Kringel', 'Kringel', 'Kringel', 'MN105', 'MN105', 'MN105', 'MN105', 'MN105', 'MN60', 'MN60', 'MN60', 'MN83', 'MN83', 'MN83', 'Mus', 'Mus', 'Notch', 'Oscar', 'Oscar', 'Patchback', 'Patchback', 'Patchback', 'Patchback', 'Patchback', 'PL', 'PL', 'Ripplefluke', 'Scabs', 'Scabs', 'Scabs', 'Scabs', 'Scabs', 'Scabs', 'Shmuddel', 'Shmuddel', 'Shmuddel', 'SN100', 'SN100', 'SN100', 'SN100', 'SN4', 'SN4', 'SN4', 'SN4', 'SN63', 'SN63', 'SN63', 'SN63', 'SN89', 'SN9', 'SN90', 'SN90', 'SN96', 'SN96', 'Stripes', 'Stripes', 'Topless', 'Topless', 'Topless', 'TR120', 'TR82', 'TR99', 'Trigger', 'TSN83', 'Upbang']
target = ['Fish', 'Grin', 'Haecksel', 'SN9', 'SN96', 'TR77', 'Jet', 'Knit', 'Notch', 'Number1', 'Oscar', 'SN100', 'SN90', 'Upbang', 'Fish', 'SN96', 'Thumper', 'Zipfel', 'Double', 'Grin', 'Zap', 'Trigger', 'Feather', 'Gallatin', 'Wave', 'Web', 'Feather', 'Gallatin', 'Jet', 'Upbang', 'Wave', 'Web', 'Knit', 'Number1', 'PL', 'SN9', 'Upbang', 'Kringel', 'Oscar', 'SN4', 'Topless', 'Zap', 'Gallatin', 'Jet', 'Ripplefluke', 'SN90', 'Web', 'Patchback', 'SN96', 'TR77', 'Trigger', 'Scabs', 'Jet', 'Ripplefluke', 'SN90', 'Upbang', 'Web', 'Hook', 'MN83', 'Scabs', 'Shmuddel', 'SN4', 'SN63', 'SN9', 'Stripes', 'TR99', 'TSN103', 'Jonah', 'MN83', 'SN9', 'Topless', 'Vau', 'Zap', 'Kringel', 'Scabs', 'SN4', 'SN63', 'TR99', 'MN23', 'Mus', 'Number1', 'Quasi', 'Web', 'Kringel', 'MN105', 'MN83', 'Patchback', 'Topless', 'Trigger', 'PL', 'Upbang', 'Oscar', 'SN100', 'SN63', 'Thumper', 'TR77', 'TR99', 'Patchback', 'Scabs', 'SN4', 'Topless', 'Trigger', 'SN100', 'Topless', 'Trigger', 'Patchback', 'Topless', 'Trigger', 'Notch', 'Number1', 'Number1', 'PL', 'TR77', 'SMN5', 'Stripes', 'Topless', 'Trigger', 'TSN103', 'SN96', 'TR77', 'Zig', 'Shmuddel', 'SN4', 'SN63', 'SN9', 'Stripes', 'TR99', 'SN4', 'Thumper', 'TR88', 'SN4', 'SN89', 'SN9', 'Zap', 'SN9', 'Stripes', 'Topless', 'Zipfel', 'Stripes', 'Thumper', 'TSN103', 'Whitetip', 'Web', 'TSN103', 'Upbang', 'Web', 'TR77', 'TR99', 'TR120', 'TSN83', 'TR99', 'Trigger', 'Zap', 'TR88', 'Web', 'Trigger', 'Vau', 'Zipfel', 'Web']
dict = {'source': source, 'target': target}
df_dol = pd.DataFrame(dict)    
print(df_dol)

```

### 2. Run the model

Analyze the data with networkX, which is pre-installed by Anaconda.

```python
# Turn dataframe into graph object
import networkx as nx
g = nx.from_pandas_edgelist(df_dol, 'source', 'target', edge_attr=None, create_using=nx.Graph())
nodes = g.nodes()
edges = g.edges()
print(nx.info(g))
density = nx.density(g)
print('Network density:', density)

# Run the community detection algorithm of your choice (e.g. fluid communities or greedy modularity)
from networkx.algorithms import community
communities = community.asyn_fluidc(g, 4, max_iter=100, seed=None)
# Alternatively, change "asyn_fluid" into "greedy_modularity_communities"
```

### 3. Retrieve information from the model

The code below will loop through the list of communities, calculate different community-level metrics, and adds them to a Pandas dataframe.

```python
# Create empty lists which will later be turned into a dataframe
dens_list=[]
trans_list=[]
top_between_list=[]
member_list=[]
index=0

# Loop through the list of communities
for index,c in enumerate(communities):
        list(c)
        index +=1

        # Take slice of dataframe containing only the people in a given community
        # This is done by creating two filters. Then apply these to the original df_dol
        filter1 = df_dol['source'].isin(c)
        filter2 = df_dol['target'].isin(c)
        name_df = 'df_{}'.format(index)
        name_df = df_dol[filter1 & filter2]

        # Create graph object of this community and calculate various metrics
        g = 'G_{}'.format(index)
        g = nx.from_pandas_edgelist(name_df, 'source', 'target', edge_attr=None, create_using=nx.Graph())

        # Density
        d = 'density_{}'.format(index)
        d = nx.density(g)
        dens_list.append(d)

        # Transitivity
        t = 'trans_{}'.format(index)
        t = nx.transitivity(g)
        trans_list.append(t)

        # Top betweenness nodes in community
        from operator import itemgetter
        betweenness_dict = nx.betweenness_centrality(g)
        nx.set_node_attributes(g, betweenness_dict, 'betweenness')
        sorted_betweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)
        top_between = sorted_betweenness[:5]
        top_between_list.append(top_between)

        # Finally add all members of a given community
        member_list.append(c)

# Create dictionary of lists        
dict = {'density':dens_list,'transitivity':trans_list,'top_betweenness':top_between_list, 'members':member_list}

# And convert to dataframe which contains the specific characteristics of each community
df_community = pd.DataFrame(dict)

# Clean the two columns with lists
df_community['top_betweenness'] = [','.join(map(str, l)) for l in df_community['top_betweenness']]
df_community['top_betweenness'] = df_community['top_betweenness'].str.replace('(', ' ')
df_community['top_betweenness'] = df_community['top_betweenness'].str.replace(')', '')
df_community['members'] = [','.join(map(str, l)) for l in df_community['members']]
df_community['members'] = df_community['members'].str.replace(',', ', ')

# Print df
# When printing in Jupyter Notebooks simply write "df_community" in the bottom of a cell without "print()"
print(df_community)
```
Et voila! You should now have a dataframe called ```df_community```. This can be saved as a .CSV or .XLXS file. You may also copy it to a clipboard or print it to Markdown format, as shown in the code below.

### Optional

```python
# Optional steps with the dataframe

# Save as .CSV
import os
path = 'Path/to/the/location/where/the/file/will/be/saved/' #change this to the relevant or preferred location on your computer
os.chdir(path)
df_community.to_csv('df_community.csv', sep='\t', encoding='utf-8')

# Save as .XLSX
import os
path = 'Path/to/the/location/where/the/file/will/be/saved/' #change this to the relevant or preferred location on your computer
os.chdir(path)
df_community.to_excel("df_community.xlsx")

# Copy to clipboard
df_community.to_clipboard()

# Print to Markdown
print(df_community.to_markdown())
```


## Sources

Cousteau, J.Y. (1975). _Dolphins. The Undersea discoveries of Jacques-Yves Cousteau_.

Lusseau, D. et al. (2003). The bottlenose dolphin community of Doubtful Sound features a large proportion of long-lasting associations, _Behavioral Ecology and Sociobiology_, 54, 396-405.

Uzzi, B. and Spiro, J. (2005). Collaboration and Creativity: The Small
World Problem. _American Journal of Sociology_, 111 (2),447–504.

[NetworkX Documentation](https://networkx.org/){:target="_blank"}.

[Original data](http://www-personal.umich.edu/~mejn/netdata/){:target="_blank"}.

For a general introduction to network analysis in Python, I refer to this [instructive tutorial](https://programminghistorian.org/en/lessons/exploring-and-analyzing-network-data-with-python){:target="_blank"} from the Programming Historian.

<!-- ## Inverted Sidebars
The colors on the sidebar can now be inverted to allow brighter sidebar images. This can be enabled per-page in the fort matter:

```yml
invert_sidebar: true
```


## Code Block Headers
Code blocks can now have headers:

~~~js
// file: 'hello-world.js'
console.log('Hello World!');
~~~

Headers are added by making the first line a comment of the form `(file|title): ['"].*['"]`, e.g.:

    ~~~js
    // file: 'hello-world.js'
    console.log('Hello World!');
    ~~~

Code blocks with and without headers now also come with a copy button.
In the case of header-less code blocks, the button only shows on hover to prevent potential overlap.


## Resume Download Buttons
Resumes can now have download buttons:

![Download Buttons](/assets/img/blog/9.1.0-3.png){:.border.lead width="1776" height="258" loading="lazy"}

Resumes can now have download buttons.
{:.figcaption}

The documentation has been updated with a chapter on [how to configure the buttons](/docs/basics/#downloads).


## SERP Breadcrumbs
Added breadcrumbs above page title:

![Breadcrumbs](/assets/img/blog/9.1.0-2.png){:.border.lead width="1588" height="164" loading="lazy"}

Bread crumbs are now shown above each page title.
{:.figcaption}

Note that this requires a [directory-like URL structure](https://qwtel.com/posts/software/urls-are-directories/) on your entire site,
otherwise the intermediate links will point to nonexisting sites.

On a side note, Hydejack now has built-in tooltips for abbreviations like SERP (activated via tap/click).
See [Example Content](/blog/hyde/2012-02-07-example-content/#inline-html-elements) on how to add them to your content.


## Last Modified At
Blog posts can now have a "last modified at" date in the sub title row.

![Last modified at](/assets/img/blog/9.1.0-1.png){:.border.lead width="1254" height="218" loading="lazy"}

Note that this depends on the `last_modified_at` property of the page, which must be either set manually in the frontmatter (not recommended), or via a plugin like [`jekyll-last-modified-at`](https://github.com/gjtorikian/jekyll-last-modified-at). Note that the later is not available when building on GitHub Pages and can increase build times.


## Clap Button Preview
I've been trying something new with [**getclaps.app**](https://getclaps.app/), a feedback and analytics tool for personal sites like those powered by Hydejack.
It looks like this:

<clap-button style="--clap-button-color:var(--body-color);margin:2rem auto 3rem;width:3rem;height:3rem;font-size:smaller" nowave></clap-button>

It is a separate product from Hydejack and not enabled by default. Because it depends on a backend component, it requires a monthly fee.
If enabled, it is placed below posts and pages where the dingbat character (❖) used to be.

I can't claim that this product is fully baked (feedback welcome), but I've been using it on my personal site and here for the last couple of months with no issues.
For more, see [the dedicated website](https://getclaps.app/).

***
{:style="margin:2rem 0"}

There are many more changes and bugfixes in 9.1. See the [CHANGELOG](/changelog/){:.heading.flip-title} for details.


## Credits

<span>Photo by <a href="https://unsplash.com/@jjying?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">JJ Ying</a> on <a href="https://unsplash.com/?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

*[SERP]: Search Engine Results Page -->

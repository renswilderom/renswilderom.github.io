---
layout: post
title: Web scraping with Beautiful Soup
image:
  path: /assets/img/blog/chris-ried-ieic5Tq8YMk-unsplash.jpg
description: >
  How to download a vast multivariate dataset from a popular online database for films
sitemap: false
comments: false
---

To run the script, you need a working Python programming environment. For this I strongly recommend [Anaconda](https://www.anaconda.com/){:target="_blank"}. The remainder of this post assumes that you have Anaconda installed and that you will be working with Jupyter Notebooks. For the script below to work, you need to install the following extra package(s):
> ...


[This page](https://renswilderom.github.io/blog/python/2021-11-19-How-to-get-started-with-Python/){:target="_blank"} will help you to get started with Anaconda, the installation of packages, and Jupyter Notebooks.


## The case: An extensive movie database

Movie databases provide a wealth of information about films, which can be used to examine both the content of films themselves as well as their production process. The script below retrieves the following variables:

* Publication date
* Genres
* Title
* Image URL
* Title URL
* Keyword URL
* Award URL
* Plot-related all_keywords
* Award outcomes

## The code

### 1. Import packages

```python
# Import packages
import requests, time               
import re                     
from bs4 import BeautifulSoup
import bs4 as bs
import pandas as pd  
import math
import os
headers = {'Accept-Language': 'en-US,en;q=0.5'} # Use this line of code to always change the language settings to English
# This is convenient when you reside in a country
```

### 2. Use a query and get the number of pages with search results

In the step below, we get the number of pages with results for a given query. Fortunately, we can provide a search query as a URL. The url/query below searches for comedy feature films from the US released between 1/1/2021 and 7/1/2021. You can also experiment with some manual searches to understand how you can form your own url/query (after conducting a manual search, the url/query appears in the address bar of the browser). See also [Advanced Title Search](https://www.imdb.com/search/title/){:target="_blank"} for more information.  


```python
query = 'https://www.imdb.com/search/title/?title_type=feature&release_date=2021-01-01,2021-01-07&genres=comedy&countries=us'
query_short = re.sub(r'https://www.imdb.com/search/title/\?title_type=|release_date=|genres=|countries=', '', query)
query_short = re.sub(r'&|,', '_', query_short)

r=requests.get(query)
html = r.text
soup = bs.BeautifulSoup(html, 'html.parser')
# Here we use some Regex to extract the number of titles associated with a query
max_titles = str([item.get_text(strip=True) for item in soup.select("div.desc")[:1]])
max_titles = re.search(r'\d+\stitles', max_titles).group(0)
max_titles = int(re.sub(r'\stitles', '', max_titles))
# If the number titles is less than 50, there is only 1 pages with results, otherwise we divide the number of titles by 50
if max_titles < 50:
    max_pages = 1
else:
    max_pages = math.ceil((max_titles/50))
print(f'There are {max_titles} titles for the search query "{query_short}".')
print(f'So, the request loop below needs to iterate {max_pages} time(s).')

```

### 3. Get the HTML code for the all pages with search results

```python
all_html = []
for page in range(1, max_pages+1):
    start = 1+page*50-50
    print(start)    
    url=f'{query}&start={start}&ref_=adv_nxt'
    print(url)
    time.sleep(2)
    r=requests.get(url, headers=headers)
    html = r.text
    all_html.append(html)

all_html_string = ' '.join(map(str, all_html)) # convert list with html code of each page to one string
soup = bs.BeautifulSoup(all_html_string, 'html.parser')
titles = soup.find_all('div', {'class':'lister-item mode-advanced'})
```

### 4. Loop through html code to get data and add it to a Pandas dataframe

This is the most interesting code. A few nested loops extract data from the lengthy (and, in my opinion, hard to read) HTML code. Several Beautiful Soup statements are used, such as `select`, `get`, and `find_all`. See also the [Beautiful Soup Documentation](https://beautiful-soup-4.readthedocs.io/en/latest/){:target="_blank"} for more information.

```python
title_dicts = []
for title in titles:
    date = title.find('span', {'class':'lister-item-year text-muted unbold'}).get_text(strip=True)
    genre = title.find('span', {'class':'genre'}).get_text(strip=True)     

    for items in title.find_all('div', {'class':'lister-item-image float-left'}):

        for primaryTitle in items.find_all('img', alt=True):
            primaryTitle = primaryTitle['alt']      

        for image_url in items.find_all('img'):
            image_url = image_url['loadlate']

        for title_url in items.find_all('a'):
            title_url = title_url['href']
            url_base = 'https://www.imdb.com'
            title_url = url_base+title_url
            print(title_url)

            keyword_url_bit = 'keywords?ref_=tt_ql_sm'
            keyword_url = title_url+keyword_url_bit
            r = requests.get(keyword_url,headers=headers)
            soup = BeautifulSoup(r.text,'lxml')        
            all_keywords = [e.a.text for e in soup.select('[data-item-keyword]')]

            award_url_bit = 'awards?ref_=tt_ql_sm'
            award_url = title_url+award_url_bit
            r = requests.get(award_url,headers=headers)
            soup = BeautifulSoup(r.text,'lxml')
            outcome = soup.find_all('td', {'class':'title_award_outcome'})

            company_url_bit = 'companycredits?ref_=tt_ql_sm'
            company_url = title_url+company_url_bit
            r = requests.get(company_url,headers=headers)
            soup = BeautifulSoup(r.text,'lxml')
            company = soup.find('div',attrs={'class':'header','id':'company_credits_content'})

    title_dict = {}
    title_dict['date'] = date
    title_dict['genre'] = genre
    title_dict['primaryTitle'] = primaryTitle
    title_dict['image_url'] = image_url
    title_dict['title_url'] = title_url
    title_dict['keyword_url'] = keyword_url
    title_dict['all_keywords'] = all_keywords
    title_dict['award_url'] = award_url
    title_dict['outcome'] = outcome
    title_dict['company'] = company

    title_dicts.append(title_dict)    
df = pd.DataFrame(title_dicts)
print(df.shape)
df.head()
```

### 5. Save the Pandas dataframe as a .CSV or .XLSX file

```python
path = 'P:/My documents/Imdb/Scraped data'
os.chdir(path)
# The name of the file is changed according to the query used
df.to_csv(f'imdb_{query_short}.csv', index=False)
df.to_excel(f'imdb_{query_short}.xlsx', index=False)
```

## Sources

Photos by [Chris Ried](https://unsplash.com/@cdr6934){:target="_blank"}.
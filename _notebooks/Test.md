## Plotting word counts through time with Pandas, Resample, and Plotly

***

### 1. Open and prepare the dataset 

***

This task works with an example dataset of old-fashioned Dutch boy names. Just because the are short and, in my humble opinion, also because they sound quite cool. If you are working with your own data, then you need a dataframe with at least two columns: 
* one column with a date string (e.g. "1950") which can be turned into a Pandas datetime object (here we use years, but the datetime function also recognizes formates such as "10/11/12," which are parsed as "2012-11-10").
* and one column with a text string (here we use a few boy names, but full newspaper articles, for example, would work, too). 


```python
# Import some packages and create an example dataset
import warnings
warnings.filterwarnings('ignore') # only use this when you know the script and want to supress unnecessary warnings

# Create a dataframe
import pandas as pd
dict={'year':['1950', '1951', '1952', '1953', '1954'],'text':['Cees Aart Arie Jan Otto Gijs Sef Toon', 
                                                              'Cees Aart Arie Jan Otto Gijs Sef Toon Cees Aart Arie Jan Otto Gijs Sef Toon', 
                                                              'Aart Arie Toon', 
                                                              'Jan Otto', 
                                                              'Gijs']} 
df=pd.DataFrame(dict,index=['0', '1', '3', '4', '5'])
# in Jupyter Notebooks, you just call the name of a dataframe (e.g. "df") in the bottom of a cell to print it
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
      <th>year</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1950</td>
      <td>Cees Aart Arie Jan Otto Gijs Sef Toon</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1951</td>
      <td>Cees Aart Arie Jan Otto Gijs Sef Toon Cees Aar...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1952</td>
      <td>Aart Arie Toon</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1953</td>
      <td>Jan Otto</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1954</td>
      <td>Gijs</td>
    </tr>
  </tbody>
</table>
</div>


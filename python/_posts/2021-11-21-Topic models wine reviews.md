```python
# Read the .CSV as a dataframe 
import os
corpus_path = 'C:/Users/User/Downloads/winemag data'
os.chdir(corpus_path)
import pandas as pd
df = pd.read_csv("winemag-data-130k-v2.csv", encoding='UTF-8')
df.reset_index(level=0, inplace=True)
print('This dataset has almost 130K rows and 15 collumns')
print(df.shape)
df.head()
```

    This dataset has almost 130K rows and 15 collumns
    (129971, 15)
    




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
      <th>Unnamed: 0</th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print first review (from collumn "description")
df.iloc[0]['description'] # row, collumn
```




    "Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity."




```python
# Extract the production year of the wine from the "title" collumn
# \d{4} is a pattern that matches with four digit numbers (which is useful to extract years from text)
df['date'] = df['title'].str.extract('(\d{4})', expand=True)

# For example, the first reviews concerns a wine from 2013
df.iloc[0]['date']
```




    '2013'




```python
# Convert this string to a datevariable
df['datetime']  = pd.to_datetime(df['date'], errors = 'coerce')

# Add a count (this will be useful later when making the graphs)
df['count'] = 1

# Keep first 2000 rows to speed up the topic modeling
df = df[:2000]
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
      <th>Unnamed: 0</th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
      <th>date</th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
      <td>2013</td>
      <td>2013-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
      <td>2011</td>
      <td>2011-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
      <td>2013</td>
      <td>2013-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
      <td>2013</td>
      <td>2013-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
      <td>2012</td>
      <td>2012-01-01</td>
      <td>1</td>
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
      <td>...</td>
      <td>...</td>
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
      <th>1995</th>
      <td>1995</td>
      <td>1995</td>
      <td>US</td>
      <td>This full-throttle, oaky Chardonnay features a...</td>
      <td>NaN</td>
      <td>84</td>
      <td>20.0</td>
      <td>California</td>
      <td>Russian River Valley</td>
      <td>Sonoma</td>
      <td>Virginie Boone</td>
      <td>@vboone</td>
      <td>McIlroy 2013 Chardonnay (Russian River Valley)</td>
      <td>Chardonnay</td>
      <td>McIlroy</td>
      <td>2013</td>
      <td>2013-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>1996</td>
      <td>1996</td>
      <td>US</td>
      <td>Despite its single-vineyard designation, this ...</td>
      <td>Guadalupe Vineyard</td>
      <td>84</td>
      <td>25.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Wildewood 2013 Guadalupe Vineyard Pinot Noir (...</td>
      <td>Pinot Noir</td>
      <td>Wildewood</td>
      <td>2013</td>
      <td>2013-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>1997</td>
      <td>1997</td>
      <td>Italy</td>
      <td>Fruity aromas that recall blueberry and pomegr...</td>
      <td>NaN</td>
      <td>84</td>
      <td>11.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Terre Siciliane</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Vino dei Fratelli 2013 Nero d'Avola (Terre Sic...</td>
      <td>Nero d'Avola</td>
      <td>Vino dei Fratelli</td>
      <td>2013</td>
      <td>2013-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>1998</td>
      <td>1998</td>
      <td>France</td>
      <td>Towards the dry end of the Pinot Gris spectrum...</td>
      <td>Réserve Particulière</td>
      <td>84</td>
      <td>15.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Cave de Beblenheim 2013 Réserve Particulière P...</td>
      <td>Pinot Gris</td>
      <td>Cave de Beblenheim</td>
      <td>2013</td>
      <td>2013-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>1999</td>
      <td>1999</td>
      <td>France</td>
      <td>This wine has a light, fresh character to go w...</td>
      <td>Anne de K</td>
      <td>84</td>
      <td>19.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Cave de Kientzheim-Kaysersberg 2013 Anne de K ...</td>
      <td>Gewürztraminer</td>
      <td>Cave de Kientzheim-Kaysersberg</td>
      <td>2013</td>
      <td>2013-01-01</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 18 columns</p>
</div>




```python
# Frequencies by year, month, or day (i.e. A-DEC, M, D)
df_1 = df.set_index('datetime').resample('A-DEC')['count'].sum()
df_1 = df_1.reset_index()
df_1
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
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1877-12-31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1878-12-31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1879-12-31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1880-12-31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1881-12-31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2013-12-31</td>
      <td>273</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2014-12-31</td>
      <td>282</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2015-12-31</td>
      <td>169</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2016-12-31</td>
      <td>62</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2017-12-31</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>141 rows × 2 columns</p>
</div>




```python
# Limit the data to all reviews concerning wines from 1990
# The dataset is generated in 2017. Most reviews focus on wines from 2014.
df_1 = df_1[(df_1['datetime'] > '1990-12-31')]
df_1
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
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>114</th>
      <td>1991-12-31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>115</th>
      <td>1992-12-31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>1993-12-31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>1994-12-31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>1995-12-31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>119</th>
      <td>1996-12-31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>120</th>
      <td>1997-12-31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>121</th>
      <td>1998-12-31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>122</th>
      <td>1999-12-31</td>
      <td>3</td>
    </tr>
    <tr>
      <th>123</th>
      <td>2000-12-31</td>
      <td>8</td>
    </tr>
    <tr>
      <th>124</th>
      <td>2001-12-31</td>
      <td>14</td>
    </tr>
    <tr>
      <th>125</th>
      <td>2002-12-31</td>
      <td>10</td>
    </tr>
    <tr>
      <th>126</th>
      <td>2003-12-31</td>
      <td>26</td>
    </tr>
    <tr>
      <th>127</th>
      <td>2004-12-31</td>
      <td>28</td>
    </tr>
    <tr>
      <th>128</th>
      <td>2005-12-31</td>
      <td>62</td>
    </tr>
    <tr>
      <th>129</th>
      <td>2006-12-31</td>
      <td>118</td>
    </tr>
    <tr>
      <th>130</th>
      <td>2007-12-31</td>
      <td>121</td>
    </tr>
    <tr>
      <th>131</th>
      <td>2008-12-31</td>
      <td>104</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2009-12-31</td>
      <td>124</td>
    </tr>
    <tr>
      <th>133</th>
      <td>2010-12-31</td>
      <td>167</td>
    </tr>
    <tr>
      <th>134</th>
      <td>2011-12-31</td>
      <td>174</td>
    </tr>
    <tr>
      <th>135</th>
      <td>2012-12-31</td>
      <td>187</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2013-12-31</td>
      <td>273</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2014-12-31</td>
      <td>282</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2015-12-31</td>
      <td>169</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2016-12-31</td>
      <td>62</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2017-12-31</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Graph
df_melt1 = df_1.melt(id_vars='datetime', value_vars=['count'])

import plotly.graph_objects as go
fig = go.Figure()


# Create and style traces
fig.add_trace(go.Scatter(x=df_melt1['datetime'], y=df_melt1['value'], opacity=0.7, name='Reviewed wines',
                         line=dict(color='#E1D89F', width=2), mode='lines+markers', marker_symbol='circle'))

# Edit the layout
fig.update_layout(showlegend=True,
xaxis_rangeslider_visible=True,
width=1000,
height=500,
xaxis_title='Datum',
yaxis_title='# artikelen',
paper_bgcolor='rgba(0,0,0,0)',
plot_bgcolor='rgba(0,0,0,0)')


fig.update_xaxes(showgrid=True, gridwidth=0.3, gridcolor='LightGrey')
fig.update_yaxes(showgrid=True, gridwidth=0.3, gridcolor='LightGrey')


fig.show() # 'svg' helps to share graph through nbviewer. If this is not necessary, it can be removed for a better interactive plot.

```


<div>                            <div id="4f959042-fd49-4101-aa38-0740149d4ea3" class="plotly-graph-div" style="height:500px; width:1000px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("4f959042-fd49-4101-aa38-0740149d4ea3")) {                    Plotly.newPlot(                        "4f959042-fd49-4101-aa38-0740149d4ea3",                        [{"line":{"color":"#E1D89F","width":2},"marker":{"symbol":"circle"},"mode":"lines+markers","name":"Reviewed wines","opacity":0.7,"x":["1991-12-31T00:00:00","1992-12-31T00:00:00","1993-12-31T00:00:00","1994-12-31T00:00:00","1995-12-31T00:00:00","1996-12-31T00:00:00","1997-12-31T00:00:00","1998-12-31T00:00:00","1999-12-31T00:00:00","2000-12-31T00:00:00","2001-12-31T00:00:00","2002-12-31T00:00:00","2003-12-31T00:00:00","2004-12-31T00:00:00","2005-12-31T00:00:00","2006-12-31T00:00:00","2007-12-31T00:00:00","2008-12-31T00:00:00","2009-12-31T00:00:00","2010-12-31T00:00:00","2011-12-31T00:00:00","2012-12-31T00:00:00","2013-12-31T00:00:00","2014-12-31T00:00:00","2015-12-31T00:00:00","2016-12-31T00:00:00","2017-12-31T00:00:00"],"y":[1,0,0,0,0,1,1,0,3,8,14,10,26,28,62,118,121,104,124,167,174,187,273,282,169,62,2],"type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"rangeslider":{"visible":true},"title":{"text":"Datum"},"showgrid":true,"gridwidth":0.3,"gridcolor":"LightGrey"},"showlegend":true,"width":1000,"height":500,"yaxis":{"title":{"text":"# artikelen"},"showgrid":true,"gridwidth":0.3,"gridcolor":"LightGrey"},"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('4f959042-fd49-4101-aa38-0740149d4ea3');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
# # Save graph 
# import chart_studio
# username = 'renswilderom' # your username
# api_key = 'JQ0okV7QsrTwEJJfp7dp' # your api key - go to profile > settings > regenerate key
# chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

# # Online file
# import chart_studio.plotly as py
# py.plot(fig, filename = 'Figuur 1. Wekelijkse media aandacht voor Covid-19', auto_open=True)

# # Local html file
# import plotly.io as pio
# pio.write_html(fig, file='Figuur 1. Wekelijkse media aandacht voor Covid-19', auto_open=False)
```


```python
# Figure 
# import chart_studio
# username = 'renswilderom' # your username
# api_key = 'JQ0okV7QsrTwEJJfp7dp' # your api key - go to profile > settings > regenerate key
# chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
# import chart_studio.plotly as py
# py.plot(fig, filename = 'Dutch news on COVID-19', auto_open=True)
# import plotly.io as pio
# pio.write_html(fig, file='Dutch news on COVID-19', auto_open=True)
```

***

## Topic modeling


***


```python
# Import necessary packages and such
from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import warnings
warnings.filterwarnings('ignore') # only use this when you know the script and want to supress unnecessary warnings
```


```python
%%time
# Apply a count vectorizer to the data
# The run time of this cell is rather quick
tf_vectorizer = CountVectorizer(lowercase = True,
                                         strip_accents = 'unicode',
                                         stop_words = 'english',
                                         token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                                         max_df = 0.5, # ignore words occuring in > 50 % of the corpus (i.e. corpus specific stop words)
                                         min_df = 10) # ignore words in <10 documents of the corpus
dtm_tf = tf_vectorizer.fit_transform(df['description'].values.astype('U')) # import articles from df 'content' as unicode string
print(dtm_tf.shape)
```

    (2000, 783)
    Wall time: 133 ms
    

**5-topic model**


```python
%%time
# run a LDA model with 10 topics
lda_tf = LatentDirichletAllocation(n_components=5, random_state=0)
lda_tf.fit(dtm_tf)
```

    Wall time: 9.44 s
    




    LatentDirichletAllocation(n_components=5, random_state=0)




```python
# Print the topics in a conventional way
n_top_words = 30

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

tf_feature_names = tf_vectorizer.get_feature_names() 
print_top_words(lda_tf, tf_feature_names, n_top_words)
```

    Topic #0:
    wine acidity flavors fruit drink crisp fresh apple citrus ripe light fruity dry white palate texture lemon finish rich pear soft green bright peach pineapple ready notes character balanced lime
    Topic #1:
    wine tannins fruit drink black ripe flavors rich firm fruits dark years wood acidity structure dense aging blackberry structured dry character juicy finish oak chocolate age tannic concentrated currant cherries
    Topic #2:
    palate cherry aromas black tannins pepper nose alongside red dried offers berry spice bright opens savory licorice crushed acidity delivers drink tobacco notes hint herb dark clove note lead raspberry
    Topic #3:
    aromas finish flavors palate notes berry fruit nose mouth like fresh herbal peach feels sweet green bit white apple slightly tastes oak plum acidity creamy offers stone note tones oaky
    Topic #4:
    flavors cherry fruit wine cabernet blend oak black red aromas spice finish chocolate palate sauvignon notes vanilla plum tannins blackberry berry bodied merlot syrah soft shows cassis good raspberry ripe
    
    

Well, these topics are pretty similar, which is not so surprising. Yet, I believe that it could be said that topic #0 relates more to a citrussy/acidity range of flavours, whereas topic #2 relates more to a spice/pepper range of flavors. 


```python
%%time
# LDA visualization
pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)
```

    Wall time: 4.54 s
    





<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v1.0.0.css">


<div id="ldavis_el322019902666585923079103309"></div>
<script type="text/javascript">

var ldavis_el322019902666585923079103309_data = {"mdsDat": {"x": [0.03468379970249645, -0.04949892854510558, -0.1962473440305274, 0.21298375234040323, -0.0019212794672670045], "y": [-0.15761330704266818, 0.021006520488722993, -0.053273596615884725, -0.016675086212213763, 0.20655546938204367], "topics": [1, 2, 3, 4, 5], "cluster": [1, 1, 1, 1, 1], "Freq": [29.554631157500822, 22.191468705845832, 19.0411690991452, 15.515039497842059, 13.69769153966609]}, "tinfo": {"Term": ["aromas", "wine", "tannins", "cherry", "palate", "black", "finish", "cabernet", "crisp", "alongside", "pepper", "firm", "apple", "nose", "citrus", "dark", "flavors", "drink", "dried", "years", "wood", "peach", "offers", "rich", "chocolate", "fresh", "berry", "oak", "mouth", "white", "pineapple", "yellow", "refreshing", "blanc", "tropical", "zest", "tangerine", "crisp", "pink", "mango", "screwcap", "kiwi", "peaches", "passion", "crispness", "apricots", "spring", "balancing", "ready", "aperitif", "butterscotch", "cru", "stainless", "steel", "steely", "skins", "touched", "vivid", "fermented", "apricot", "fruitiness", "grapefruit", "citrus", "lime", "gooseberry", "chardonnay", "zesty", "lemon", "apple", "fruity", "apples", "pear", "minerality", "acidity", "lively", "honey", "fresh", "freshness", "attractive", "wine", "light", "clean", "white", "texture", "drink", "peach", "dry", "fruit", "flavors", "ripe", "green", "mineral", "bright", "character", "soft", "rich", "balanced", "palate", "finish", "notes", "franc", "grenache", "petit", "verdot", "french", "petite", "sirah", "mourvedre", "cab", "bacon", "rhubarb", "zinfandel", "bell", "additional", "amounts", "valley", "cabernet", "overripe", "purple", "syrah", "nice", "milk", "cassis", "napa", "chalky", "merlot", "shiraz", "cooked", "winemaker", "jam", "barrel", "cranberry", "cedar", "sauvignon", "cola", "blend", "cherry", "chocolate", "oak", "meat", "vanilla", "sangiovese", "red", "flavors", "blackberry", "fruit", "bodied", "spice", "plum", "black", "shows", "raspberry", "wine", "notes", "aromas", "soft", "berry", "finish", "good", "tannins", "palate", "rich", "ripe", "needs", "develop", "need", "future", "soften", "impressive", "cherries", "aging", "tough", "potential", "powerful", "raspberries", "selection", "blackberries", "years", "better", "firm", "provide", "density", "wood", "property", "dense", "leaving", "family", "cigar", "cellar", "structured", "extracted", "currants", "help", "time", "structure", "age", "tannins", "old", "dark", "fruits", "wine", "solid", "new", "rich", "drink", "ripe", "black", "concentrated", "fruit", "tannic", "blackberry", "character", "juicy", "flavors", "fine", "chocolate", "acidity", "dry", "currant", "oak", "finish", "feels", "prosecco", "chunky", "rubbery", "smells", "briny", "salty", "sparkler", "basic", "generic", "freshly", "filling", "mildly", "oily", "luminous", "nero", "tones", "informal", "tastes", "avola", "dessert", "jasmine", "mild", "tonic", "leafy", "raisin", "sugar", "hay", "citric", "include", "oaky", "mouth", "aromas", "finish", "like", "herbal", "bit", "lean", "sweetness", "slightly", "flavors", "palate", "peach", "berry", "notes", "floral", "creamy", "nose", "stone", "baked", "green", "melon", "sweet", "fresh", "white", "fruit", "apple", "note", "oak", "plum", "offers", "acidity", "alongside", "anise", "underbrush", "violet", "soon", "balsamic", "blue", "skinned", "fragrance", "petal", "game", "elderberry", "extremely", "framed", "pine", "crushed", "meet", "lavender", "enjoy", "mingle", "violets", "menthol", "slowly", "soil", "luscious", "graphite", "sage", "chopped", "clove", "forest", "dried", "pepper", "opens", "delivers", "savory", "palate", "glass", "cherry", "tobacco", "black", "offers", "nose", "licorice", "rose", "aromas", "leather", "herb", "hint", "red", "tannins", "lead", "bright", "note", "berry", "spice", "dark", "notes", "drink", "acidity", "raspberry"], "Freq": [626.0, 1207.0, 452.0, 436.0, 583.0, 438.0, 512.0, 183.0, 197.0, 88.0, 159.0, 135.0, 201.0, 255.0, 179.0, 179.0, 901.0, 493.0, 132.0, 110.0, 114.0, 138.0, 208.0, 281.0, 139.0, 285.0, 275.0, 242.0, 82.0, 190.0, 79.73735315542963, 54.81669366931051, 51.94282218134366, 46.193656204998334, 42.35316673338174, 40.42325642022317, 35.648479552622234, 190.0193837831431, 20.298827700391108, 17.437912931266595, 17.43195734639115, 16.47055554825567, 14.5651440098858, 13.594770680998918, 12.644771555015252, 11.691575247320074, 11.68671593078591, 11.68152411889871, 79.73640125441831, 10.731278159985711, 10.72510021369829, 10.722470135883702, 9.774330896112872, 9.773433102261832, 9.7716908990093, 10.65756977412186, 9.767101046415942, 9.76138064268383, 22.149186969982303, 56.27054470998336, 29.835366977886906, 67.21641058158757, 151.8646587488349, 69.38447755675426, 15.637161113765963, 53.62115224812132, 38.70585536639754, 102.55988535052421, 153.81315971466114, 121.91601691244317, 18.00862424254702, 89.1701057801116, 45.11419347786404, 328.9293560679385, 53.58694901812152, 60.61751083108427, 183.73400977448657, 44.63676807459294, 54.313281546206944, 524.4282071732265, 122.75074467194494, 64.2508383994157, 110.84646963539612, 103.5160577318188, 199.4142194615528, 81.98159676747848, 114.39316610334154, 236.70835160549584, 251.44549216277838, 146.2098434413131, 82.38413203348304, 68.73655446142917, 82.22253860894793, 74.09513321588432, 83.4255559444457, 89.87714076615218, 72.00606632902918, 105.29337739855538, 92.15942352837722, 76.5733475314867, 49.084395286290764, 32.45262730421664, 29.526194888922817, 27.574389435609127, 27.56740235645318, 24.45310076642245, 23.45122643194895, 15.838596375972191, 24.10834013920942, 13.87399181724879, 11.915744363073768, 10.94822101003478, 10.923114551094567, 9.969937460611728, 9.967148819284523, 29.354802147132922, 163.69421166818228, 10.877540517163913, 17.378662018958433, 70.35747969709443, 32.867817564718386, 12.306969758375187, 60.45628810663572, 10.54129223090317, 13.68867211353168, 70.88212762003668, 8.53999794832667, 8.511415688129981, 9.257987275705046, 17.737550968362832, 37.01696023784785, 44.03587357472815, 30.003757080661615, 82.1163175982117, 52.075542090400674, 128.24263985993895, 228.11597741033768, 82.40388178857928, 127.56217899036162, 29.47776477096039, 80.25025201786409, 21.200695460721352, 119.43132700117606, 246.99897495679468, 73.75218355293225, 205.40510471352684, 72.40891500617788, 102.75646090098215, 80.07880345985343, 121.52402761483042, 68.06765596466285, 57.471693476812305, 172.0935318255767, 81.34656489903547, 110.49037126483252, 68.17622733683699, 73.0092973867746, 91.60920291904988, 58.2122252557335, 79.87148734067628, 82.23051743083407, 54.360690650967925, 54.39449942582936, 23.77744802962001, 21.7820835355092, 17.9141523847812, 13.978683414918219, 12.960553997404237, 33.869016588103335, 48.13684924218815, 69.36386842517048, 13.243701612932547, 27.43716726766604, 37.618367305457205, 17.643190558948447, 10.919312616087902, 24.41180241041902, 93.51281984298234, 15.036944771398977, 113.90695103303155, 14.806624567239275, 16.00466348916498, 92.30341920377502, 9.967259646335515, 74.35817512344701, 9.667245729320811, 17.09940453592661, 8.991973115704255, 20.935155493232504, 62.2243093435523, 12.445494700141502, 18.128179962290258, 7.91998568909466, 47.97330909413153, 81.00866394529977, 54.3767454270955, 269.24757965921214, 23.297164503000936, 99.90295267583136, 110.64481220706183, 470.3188989652636, 44.081220421022934, 46.232663335527164, 131.01152997629825, 195.50363722930666, 152.89228331894387, 168.26282124144095, 53.21155735542546, 216.87344037315788, 53.65229183170339, 66.04273488742902, 60.16141131459523, 58.53330430422252, 143.72484866584105, 47.11247403925094, 55.02423532208218, 84.35140678891388, 61.721663433472244, 49.0665445223992, 55.61058508997721, 56.72356291391368, 55.0478714505206, 22.723127978986593, 18.80469568389749, 17.826476399939303, 16.844607563356863, 15.865795700276555, 15.863257635115477, 14.8865476401855, 14.885055218038538, 13.905082864851634, 13.9039509608649, 13.897088026446957, 12.928744991666699, 13.805419044110126, 11.947970420295016, 11.942319372807797, 35.75525989184012, 11.918259073548617, 45.76708795000783, 9.986248260241007, 9.97913244939444, 10.819047145325111, 25.828543516826297, 9.655293781562632, 11.35312170629927, 21.326265631745994, 25.03846646339362, 10.908654216194085, 11.606002171027868, 15.42569670089663, 35.66869815785326, 63.8819289873063, 290.8972218248028, 244.8359852950634, 62.129218040459556, 58.40897101662748, 48.336784880091805, 34.47432548119955, 31.28531715296347, 46.1841740240904, 231.89220620698302, 155.43658748032988, 55.703222081419135, 83.84535994792076, 87.32192255462571, 29.268154493413356, 41.637796241073275, 72.65272001108896, 36.68892010085199, 29.82009557976605, 48.80357428527354, 35.379444906456406, 53.5121246555287, 60.36970071805129, 47.354139030997665, 76.24335762310763, 46.939309339224835, 35.77473862438165, 44.910828796875784, 42.70460672247471, 39.48566175341638, 41.77743657873544, 88.13948746579794, 33.60148219699876, 20.654534813898678, 20.637932004412562, 19.622943046303586, 17.564153982420954, 16.545792317397844, 15.54198996805277, 11.4433771196327, 10.42776461520448, 10.427305351767632, 10.425616919923586, 10.42254322007989, 16.969777019345333, 14.11534768457498, 49.34075954267393, 13.698236606441862, 15.249479081449756, 20.177180302643247, 12.168397320357888, 11.265030963375597, 18.875611686351682, 12.70761894165294, 12.701477548205371, 12.278803866038606, 17.23826947010331, 17.786502991956564, 17.62938690504529, 42.74112007525586, 18.69954558315174, 84.83519456390961, 98.95566842989439, 55.02795019604935, 47.91569369772207, 51.72553242303265, 225.11530200884556, 31.702428005946828, 159.06199204209724, 47.83292120931628, 148.72552340693278, 84.46466270922639, 96.60781149649631, 49.97540510278819, 32.036478995440554, 159.00485479550608, 35.82947557526947, 45.98175078493403, 46.1249145654623, 87.08311644390095, 102.66220094522359, 38.116147150289926, 56.808353105821915, 41.10766621657527, 66.65329495497113, 65.58055940437235, 45.36943552082808, 46.48804305740946, 47.87721678767972, 48.23740593899943, 37.02562180750576], "Total": [626.0, 1207.0, 452.0, 436.0, 583.0, 438.0, 512.0, 183.0, 197.0, 88.0, 159.0, 135.0, 201.0, 255.0, 179.0, 179.0, 901.0, 493.0, 132.0, 110.0, 114.0, 138.0, 208.0, 281.0, 139.0, 285.0, 275.0, 242.0, 82.0, 190.0, 80.53540763219678, 55.61598966648483, 52.740667470204485, 46.98991449327782, 43.15633713913537, 41.24047652916466, 36.44716612747547, 197.61231423511813, 21.11236563648329, 18.23672423352559, 18.236845892213452, 17.278490490065167, 15.361350211376696, 14.403266741384554, 13.44455280696672, 12.485977277593923, 12.486202020794343, 12.486245808086677, 85.49110903303423, 11.527571356335235, 11.527755997118648, 11.527827112099276, 10.569102523357959, 10.56911415802051, 10.56915805925773, 11.532244529869924, 10.569321804016345, 10.569505815380717, 24.008547324563565, 61.46161629763404, 32.65377725514321, 75.91406987379682, 179.84280432858242, 82.72354299856207, 17.296908087659972, 63.47944284834742, 45.19361886090329, 128.03275146551104, 201.35157493381226, 162.23347302587757, 20.183384199077082, 119.5121101900387, 56.101765530436, 525.1256857807972, 68.42288952982679, 78.99885146102241, 285.2195006133334, 55.87827622019063, 72.61509046433507, 1207.560360231032, 203.27002183416673, 91.71605764380294, 190.91512566626125, 183.1393486100132, 493.23939937895426, 138.2839871718412, 231.88898755860419, 756.6021401334116, 901.052330793763, 384.54153554609354, 148.3889960966267, 109.01911135919555, 164.1484673418304, 145.6243290351578, 216.55244716232036, 281.8151903029142, 141.3580741076464, 583.0837949701875, 512.5610901512706, 309.7725189544555, 49.88058190787469, 33.256014108848376, 30.322214579848776, 28.366402489864733, 28.366462956802245, 25.441206157668734, 24.464380266277345, 16.631436477330634, 25.436127940978693, 14.675918044793844, 12.719954938412787, 11.74177834473147, 11.74235344316077, 10.763865266738401, 10.763876187415606, 32.23565027475241, 183.9884129422089, 12.721630584137479, 20.559979231993374, 83.3898021013777, 39.02420692277384, 14.747542443518384, 72.5524322565933, 12.696728703398405, 16.58870219902704, 88.13860455397321, 10.773706326073384, 10.76660926057559, 11.753467959907246, 22.541907232531813, 48.725062327089, 58.29193219148762, 40.165593956578135, 119.60254299299933, 74.4620563247925, 203.7930575591613, 436.56425846420285, 139.30107812335845, 242.69833455825983, 42.45285828430075, 155.7769856790154, 28.65739724392773, 316.1314729344405, 901.052330793763, 164.65325383433375, 756.6021401334116, 165.50526345931462, 284.468962854555, 207.19935752712047, 438.9041641284976, 166.27442157548518, 127.78219240565828, 1207.560360231032, 309.7725189544555, 626.5250384895317, 216.55244716232036, 275.08633162289766, 512.5610901512706, 159.77204114239663, 452.24910368938106, 583.0837949701875, 281.8151903029142, 384.54153554609354, 24.614863438742667, 22.64500318741242, 18.70820809719646, 14.769602451624499, 13.784271200303753, 36.4997430824277, 52.30888110601326, 75.69655409995788, 14.764621209440785, 31.421276769886347, 43.28672241950436, 20.62247253287958, 12.771599010462971, 28.683615371483228, 110.01844225233543, 17.69678623324257, 135.66750865650334, 17.66847586161104, 19.672182521908038, 114.79377554467443, 12.861756761730204, 96.62683974067515, 12.73814092462636, 22.54374390838396, 11.871200325657043, 27.795006485661688, 83.22648159134104, 16.715209105863224, 24.533849747734852, 10.774854151378653, 66.95575093274338, 118.52665925360314, 80.16078405228474, 452.24910368938106, 33.23215663164142, 179.66380590285246, 208.90493847494088, 1207.560360231032, 70.97267143211094, 75.37495271195883, 281.8151903029142, 493.23939937895426, 384.54153554609354, 438.9041641284976, 99.6702382886092, 756.6021401334116, 104.0481334178942, 164.65325383433375, 145.6243290351578, 147.5232380633812, 901.052330793763, 98.14462002955035, 139.30107812335845, 525.1256857807972, 231.88898755860419, 108.52732220218385, 242.69833455825983, 512.5610901512706, 55.84392515810671, 23.516053814127073, 19.597618679658932, 18.61799457076055, 17.638360344643015, 16.65878050893161, 16.6587670732037, 15.679169600001018, 15.679019819139976, 14.699396113916972, 14.699514733263795, 14.699448438761516, 13.719844172574048, 14.703938647307384, 12.740239009292152, 12.74047353574204, 38.21029072480333, 12.740117809792759, 48.96395765614316, 10.781056175544501, 10.78090600842093, 11.757443052160955, 28.411132194049696, 10.773662482572812, 12.741226045001019, 24.51461646197882, 29.31693093011744, 12.785733508730972, 13.691304029952985, 18.614850877687527, 43.097163091796034, 82.35060906969069, 626.5250384895317, 512.5610901512706, 104.45951940461615, 97.68088233249779, 82.28502529396637, 54.382023789710374, 48.73633356409076, 87.86559779942755, 901.052330793763, 583.0837949701875, 138.2839871718412, 275.08633162289766, 309.7725189544555, 50.100892466553574, 92.12410651484377, 255.57373641864865, 80.36627420307477, 55.92178813685108, 148.3889960966267, 78.42102141566929, 194.5987384438539, 285.2195006133334, 190.91512566626125, 756.6021401334116, 201.35157493381226, 94.46756070740146, 242.69833455825983, 207.19935752712047, 208.08587771254798, 525.1256857807972, 88.92673486590611, 34.71481626565999, 21.436322251340965, 21.43555601853184, 20.413296429712986, 18.367558693026435, 17.345213140117757, 16.32339797084052, 12.232673308060095, 11.210379945582462, 11.210382247759133, 11.210327170534203, 11.210120968727033, 18.34262438909676, 15.283102814487712, 53.940423630522446, 15.26489655635881, 17.291514564411276, 23.3822068487741, 14.22236313797398, 13.210717089348687, 22.340628329085135, 15.233491656110473, 15.233176203201644, 15.206609215226095, 22.28816620130909, 23.238895522373795, 23.163628599994535, 57.60662321778469, 25.27578052651552, 132.89783998087262, 159.909464432632, 88.53571450123903, 77.43874817231875, 89.15159279762574, 583.0837949701875, 49.004784852515485, 436.56425846420285, 87.24365686078588, 438.9041641284976, 208.08587771254798, 255.57373641864865, 102.06729592748628, 55.69584686606611, 626.5250384895317, 68.14949404056128, 102.66414163525026, 104.16944902394303, 316.1314729344405, 452.24910368938106, 78.94869951498055, 164.1484673418304, 94.46756070740146, 275.08633162289766, 284.468962854555, 179.66380590285246, 309.7725189544555, 493.23939937895426, 525.1256857807972, 127.78219240565828], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -4.9289, -5.3036, -5.3575, -5.4748, -5.5616, -5.6082, -5.7339, -4.0605, -6.297, -6.449, -6.4493, -6.506, -6.629, -6.6979, -6.7704, -6.8487, -6.8492, -6.8496, -4.9289, -6.9344, -6.935, -6.9353, -7.0278, -7.0279, -7.0281, -6.9413, -7.0286, -7.0292, -6.2098, -5.2774, -5.9119, -5.0997, -4.2846, -5.0679, -6.558, -5.3257, -5.6516, -4.6772, -4.2719, -4.5043, -6.4168, -4.8171, -5.4984, -3.5118, -5.3263, -5.203, -4.0941, -5.509, -5.3128, -3.0453, -4.4974, -5.1448, -4.5995, -4.6679, -4.0122, -4.9011, -4.568, -3.8408, -3.7804, -4.3226, -4.8962, -5.0773, -4.8982, -5.0023, -4.8836, -4.8092, -5.0309, -4.6509, -4.7841, -4.9694, -5.1275, -5.5413, -5.6358, -5.7042, -5.7044, -5.8243, -5.8661, -6.2586, -5.8385, -6.3911, -6.5432, -6.6279, -6.6302, -6.7215, -6.7218, -5.6416, -3.9231, -6.6344, -6.1658, -4.7675, -5.5286, -6.5109, -4.9192, -6.6658, -6.4045, -4.7601, -6.8763, -6.8797, -6.7956, -6.1454, -5.4097, -5.2361, -5.6197, -4.6129, -5.0684, -4.1671, -3.5912, -4.6094, -4.1725, -5.6374, -4.6359, -5.967, -4.2383, -3.5117, -4.7204, -3.6961, -4.7387, -4.3887, -4.6381, -4.221, -4.8006, -4.9698, -3.873, -4.6224, -4.3161, -4.799, -4.7305, -4.5035, -4.957, -4.6407, -4.6115, -5.0254, -5.0248, -5.6992, -5.7869, -5.9824, -6.2304, -6.3061, -5.3455, -4.9939, -4.6286, -6.2844, -5.5561, -5.2405, -5.9976, -6.4774, -5.6729, -4.3299, -6.1575, -4.1326, -6.1729, -6.0951, -4.3429, -6.5687, -4.5591, -6.5992, -6.0289, -6.6716, -5.8265, -4.7372, -6.3466, -5.9705, -6.7986, -4.9973, -4.4734, -4.872, -3.2723, -5.7196, -4.2638, -4.1616, -2.7146, -5.0819, -5.0343, -3.9927, -3.5924, -3.8382, -3.7424, -4.8937, -3.4887, -4.8854, -4.6777, -4.7709, -4.7984, -3.9001, -5.0154, -4.8602, -4.433, -4.7453, -4.9748, -4.8496, -4.8298, -4.655, -5.5398, -5.7291, -5.7825, -5.8391, -5.899, -5.8992, -5.9627, -5.9628, -6.0309, -6.031, -6.0315, -6.1037, -6.0381, -6.1826, -6.1831, -5.0865, -6.1851, -4.8396, -6.362, -6.3627, -6.2819, -5.4117, -6.3957, -6.2337, -5.6032, -5.4428, -6.2736, -6.2117, -5.9271, -5.0889, -4.5061, -2.9902, -3.1626, -4.534, -4.5957, -4.785, -5.123, -5.22, -4.8305, -3.2169, -3.6169, -4.6431, -4.2342, -4.1936, -5.2867, -4.9342, -4.3775, -5.0607, -5.268, -4.7754, -5.097, -4.6833, -4.5627, -4.8055, -4.3292, -4.8143, -5.0859, -4.8585, -4.9089, -4.9872, -4.9308, -4.0597, -5.024, -5.5107, -5.5115, -5.5619, -5.6727, -5.7325, -5.795, -6.1012, -6.1941, -6.1942, -6.1943, -6.1946, -5.7072, -5.8913, -4.6398, -5.9213, -5.814, -5.534, -6.0397, -6.1169, -5.6007, -5.9964, -5.9969, -6.0307, -5.6915, -5.6602, -5.669, -4.7834, -5.6101, -4.0979, -3.9439, -4.5307, -4.6691, -4.5926, -3.122, -5.0822, -3.4693, -4.6709, -3.5365, -4.1023, -3.9679, -4.6271, -5.0717, -3.4697, -4.9598, -4.7103, -4.7072, -4.0717, -3.9071, -4.898, -4.4989, -4.8224, -4.3391, -4.3553, -4.7238, -4.6994, -4.67, -4.6625, -4.927], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.209, 1.2045, 1.2037, 1.2018, 1.2001, 1.1989, 1.1968, 1.1797, 1.1796, 1.1741, 1.1738, 1.171, 1.1657, 1.1612, 1.1576, 1.1532, 1.1528, 1.1523, 1.1492, 1.1474, 1.1468, 1.1465, 1.1408, 1.1407, 1.1405, 1.1401, 1.14, 1.1394, 1.1383, 1.1307, 1.1287, 1.0972, 1.0498, 1.0431, 1.1181, 1.0502, 1.064, 0.9971, 0.9496, 0.9332, 1.1049, 0.9261, 1.001, 0.7511, 0.9745, 0.9541, 0.7792, 0.9943, 0.9285, 0.3849, 0.7146, 0.863, 0.6752, 0.6484, 0.3133, 0.6961, 0.5123, 0.0569, -0.0574, 0.2519, 0.6305, 0.7577, 0.5276, 0.5432, 0.2651, 0.0761, 0.5444, -0.4927, -0.497, -0.1787, 1.4894, 1.481, 1.4789, 1.4771, 1.4769, 1.4658, 1.4632, 1.4566, 1.4518, 1.4493, 1.4402, 1.4355, 1.4331, 1.4288, 1.4286, 1.4118, 1.3886, 1.3489, 1.3374, 1.3355, 1.3338, 1.3246, 1.3231, 1.3194, 1.3133, 1.2876, 1.2731, 1.2704, 1.2668, 1.2658, 1.2306, 1.225, 1.2138, 1.1294, 1.1479, 1.0423, 0.8564, 0.9805, 0.8622, 1.1407, 0.8422, 1.2041, 0.532, 0.2113, 0.7023, 0.2016, 0.6788, 0.4872, 0.5548, 0.2213, 0.6123, 0.7064, -0.4429, 0.1683, -0.2298, 0.3497, 0.179, -0.2164, 0.4958, -0.2284, -0.4533, -0.1401, -0.4503, 1.624, 1.6197, 1.6152, 1.6035, 1.5969, 1.5838, 1.5754, 1.5712, 1.5499, 1.523, 1.5182, 1.5025, 1.5019, 1.4973, 1.496, 1.4957, 1.4837, 1.4819, 1.4522, 1.4405, 1.4036, 1.3966, 1.3827, 1.3822, 1.3808, 1.3751, 1.3677, 1.3636, 1.356, 1.3507, 1.3252, 1.278, 1.2705, 1.14, 1.3034, 1.0717, 1.023, 0.7156, 1.1823, 1.1698, 0.8926, 0.7332, 0.7362, 0.6998, 1.031, 0.409, 0.9962, 0.745, 0.7746, 0.7342, -0.1771, 0.9247, 0.7297, -0.1701, 0.3349, 0.8647, 0.1851, -0.5427, 1.849, 1.8291, 1.8221, 1.8199, 1.8173, 1.8146, 1.8144, 1.8115, 1.8114, 1.8078, 1.8077, 1.8072, 1.804, 1.8003, 1.7992, 1.7987, 1.797, 1.7967, 1.7958, 1.7868, 1.7861, 1.7802, 1.7681, 1.7538, 1.748, 1.724, 1.7056, 1.7046, 1.6981, 1.6754, 1.6742, 1.6094, 1.0961, 1.1245, 1.3438, 1.3491, 1.3314, 1.4075, 1.4201, 1.2202, 0.5061, 0.5413, 0.9541, 0.6752, 0.5971, 1.3258, 1.0692, 0.6055, 1.0792, 1.2346, 0.7513, 1.0674, 0.5723, 0.3106, 0.4692, -0.4315, 0.4072, 0.8923, 0.1762, 0.284, 0.2013, -0.6679, 1.9791, 1.9553, 1.9508, 1.95, 1.9485, 1.9432, 1.9408, 1.9389, 1.9212, 1.9156, 1.9155, 1.9154, 1.9151, 1.9101, 1.9085, 1.8988, 1.8797, 1.8623, 1.8405, 1.832, 1.8286, 1.8194, 1.8066, 1.8062, 1.7741, 1.731, 1.7206, 1.7149, 1.6895, 1.6866, 1.5391, 1.508, 1.5124, 1.5079, 1.4436, 1.0362, 1.5524, 0.9783, 1.387, 0.9058, 1.0863, 1.0151, 1.2738, 1.4349, 0.6167, 1.345, 1.1847, 1.1733, 0.6986, 0.5052, 1.2598, 0.9269, 1.1559, 0.5704, 0.5206, 0.6117, 0.0913, -0.3444, -0.3996, 0.7492]}, "token.table": {"Topic": [1, 2, 3, 4, 5, 2, 1, 2, 3, 1, 2, 3, 5, 2, 2, 5, 1, 1, 4, 1, 4, 1, 4, 1, 1, 2, 3, 4, 5, 1, 2, 4, 5, 4, 2, 4, 5, 1, 2, 3, 4, 5, 1, 5, 1, 2, 3, 4, 2, 1, 2, 3, 4, 5, 1, 2, 3, 2, 3, 4, 2, 3, 5, 3, 5, 2, 3, 4, 5, 1, 1, 2, 3, 4, 5, 5, 1, 2, 3, 4, 5, 1, 2, 4, 5, 4, 1, 2, 3, 2, 3, 2, 3, 4, 5, 2, 3, 3, 5, 1, 2, 1, 2, 3, 5, 1, 4, 3, 5, 2, 3, 4, 5, 2, 3, 5, 1, 5, 4, 2, 3, 5, 1, 4, 1, 4, 1, 2, 4, 5, 2, 5, 2, 3, 4, 1, 2, 3, 4, 5, 2, 4, 2, 5, 1, 2, 4, 1, 4, 1, 1, 1, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 2, 3, 5, 2, 3, 4, 5, 2, 3, 5, 2, 3, 4, 4, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 4, 5, 2, 3, 5, 1, 2, 3, 4, 1, 2, 4, 1, 2, 3, 5, 1, 2, 3, 4, 5, 3, 5, 1, 2, 3, 4, 5, 1, 4, 5, 2, 3, 5, 5, 4, 5, 2, 2, 1, 2, 3, 4, 5, 4, 1, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 5, 4, 1, 2, 3, 5, 1, 2, 3, 4, 1, 4, 1, 4, 3, 4, 5, 1, 2, 4, 5, 2, 4, 5, 1, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 4, 3, 5, 2, 4, 4, 1, 2, 4, 5, 4, 1, 2, 3, 4, 5, 1, 4, 5, 1, 2, 3, 4, 5, 2, 4, 2, 3, 4, 5, 2, 3, 5, 1, 3, 1, 4, 2, 3, 5, 1, 2, 4, 5, 1, 2, 3, 4, 5, 1, 4, 1, 2, 4, 5, 4, 4, 5, 1, 2, 4, 5, 2, 5, 1, 4, 2, 3, 5, 2, 3, 2, 4, 4, 2, 5, 1, 3, 4, 5, 1, 2, 5, 4, 5, 2, 1, 2, 3, 4, 1, 2, 3, 3, 4, 1, 2, 3, 1, 2, 1, 2, 3, 4, 5, 1, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 2, 4, 1, 2, 3, 4, 5, 4, 1, 3, 2, 3, 4, 5, 2, 4, 1, 2, 3, 4, 5, 1, 1, 4, 1, 1, 4, 1, 2, 3, 5, 5, 2, 2, 2, 5, 1, 1, 1, 2, 3, 4, 5, 1, 3, 2, 3, 3, 5, 4, 1, 3, 2, 3, 3, 4, 1, 3, 1, 2, 3, 4, 5, 1, 2, 5, 1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 5, 4, 1, 2, 5, 4, 2, 5, 1, 2, 3, 1, 3, 4, 5, 1, 1, 3, 2, 3, 1, 2, 3, 4, 5, 2, 5, 1, 1, 2, 3, 4, 3, 5, 4, 1, 2, 3, 4, 5, 3, 3, 5, 2, 3, 4, 5, 5, 4, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 4, 1, 2, 3, 1, 3, 5, 1, 4, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 5, 1, 2, 3, 4, 2, 3, 5, 1, 2, 4, 1, 2, 3, 4, 5, 1, 3, 5, 2, 3, 4, 5, 3, 4, 1, 4, 1, 2, 3, 1, 5, 1, 2, 1, 2, 4, 5, 2, 5, 3, 5, 1, 1, 4, 5, 1, 2, 3, 4, 5, 2, 3, 1, 2, 3, 5, 1, 3, 5, 1, 1, 1, 2, 4, 2], "Freq": [0.6265166776422629, 0.04189473224355558, 0.1599617049299395, 0.07998085246496975, 0.091406688531394, 0.9290342968990114, 0.24949855763580148, 0.06237463940895037, 0.6736461056166639, 0.05284256393914536, 0.02642128196957268, 0.9115342279502574, 0.9895786698195593, 0.9290333543311583, 0.028806144107097097, 0.9794088996413013, 0.9542339544013929, 0.764831365489058, 0.2334225595973099, 0.8918226905091109, 0.09909141005656788, 0.9111377697718587, 0.08135158658677309, 0.9610781545737707, 0.08140137563050026, 0.17557159449715742, 0.023941581067794194, 0.46446667271520736, 0.2537807593186185, 0.7436470801688545, 0.04131372667604747, 0.0963986955774441, 0.11016993780279326, 0.9275529073564953, 0.9539437299437891, 0.5364635323638863, 0.4470529436365719, 0.5093448001079222, 0.12733620002698054, 0.18393006670563858, 0.042445400008993515, 0.1344104333618128, 0.9610574855276547, 0.9799887018645551, 0.18470987147401544, 0.7593628049487301, 0.041046638105336763, 0.9566924573747224, 0.9367798417281378, 0.025446556936154704, 0.2653712366198991, 0.16358500887528024, 0.30535868323385645, 0.24355990210319503, 0.05650743512522899, 0.05650743512522899, 0.8476115268784349, 0.3159748679314859, 0.09722303628661104, 0.5833382177196662, 0.2779650091546686, 0.3827714880162649, 0.33948185544299686, 0.8367146082937787, 0.1394524347156298, 0.44942932056754414, 0.4008423669926745, 0.024293476787434817, 0.12146738393717409, 0.978933468937905, 0.201184478465846, 0.6280881278933729, 0.034348569494168826, 0.1177665239800074, 0.014720815497500926, 0.9800974979477587, 0.24772626044052573, 0.4350314817492159, 0.09063155869775331, 0.09667366261093686, 0.12688418217685465, 0.4995477650683109, 0.06701250507013927, 0.08528864281654089, 0.34724661718163075, 0.9604544577210556, 0.9542186703769094, 0.9435398365540917, 0.039314159856420484, 0.8913604795945096, 0.10870249751152557, 0.8269881261568238, 0.1102650834875765, 0.013783135435947063, 0.04134940630784119, 0.7469079140826882, 0.24896930469422937, 0.755531394131282, 0.2158661126089377, 0.12056398240227038, 0.8439478768158927, 0.5081568477622603, 0.06866984429219734, 0.4120190657531841, 0.006866984429219735, 0.8506690918665775, 0.1417781819777629, 0.9176262039082704, 0.0764688503256892, 0.5222598863271245, 0.10307760914351143, 0.009162454146089904, 0.3642075523070737, 0.5886530176556471, 0.3948282435495194, 0.007178695337263989, 0.21585564534570287, 0.7770803232445302, 0.9695055460855955, 0.08423747999928156, 0.7581373199935342, 0.16847495999856313, 0.07303906171481271, 0.8764687405777525, 0.8451825502136181, 0.15013111089320846, 0.6978058329606401, 0.05451608070005001, 0.21806432280020005, 0.021806432280020004, 0.24302761068067308, 0.7464419470906387, 0.6983422506247166, 0.25516351465133874, 0.0402889759975798, 0.1906286202003734, 0.010033085273703864, 0.5317535195063048, 0.050165426368519325, 0.21069479074778116, 0.8359177696691905, 0.18575950437093122, 0.754821436617696, 0.24017045710563054, 0.4559067283136431, 0.0868393768216463, 0.4559067283136431, 0.9614785431536365, 0.035422893695133975, 0.9669343552478472, 0.954212783817231, 0.03707794387562597, 0.03707794387562597, 0.9084096249528363, 0.06449988682996494, 0.3224994341498247, 0.45149920780975455, 0.009214269547137848, 0.15664258230134342, 0.08152002317470192, 0.16304004634940383, 0.7336802085723173, 0.18924234533016865, 0.5565951333240254, 0.2504678099958115, 0.05165372755121369, 0.012913431887803422, 0.3099223653072821, 0.6198447306145642, 0.15523637159464854, 0.7658327665335994, 0.07244364007750265, 0.050833200580888485, 0.8133312092942158, 0.10166640116177697, 0.9275658272309427, 0.971516754399445, 0.15049153547475647, 0.09029492128485388, 0.11286865160606735, 0.639589025767715, 0.40345519893699516, 0.08312393545938092, 0.39737295975704046, 0.018246717539864104, 0.09731582687927522, 0.4916145488417786, 0.09487298310981691, 0.2673693160367568, 0.14662188298789888, 0.8920346255624466, 0.0855351256164624, 0.0427675628082312, 0.8553512561646239, 0.23930301886543034, 0.717909056596291, 0.8920510338735044, 0.17743281755930568, 0.04435820438982642, 0.7540894746270492, 0.9848877893930743, 0.9163403225771771, 0.04165183284441714, 0.9524166881719783, 0.2954823197773693, 0.08151236407651566, 0.47888513894952955, 0.1426466371339024, 0.17949079976564025, 0.17949079976564025, 0.11120625637653798, 0.4779918037237159, 0.05267664775730747, 0.8402896251941699, 0.1547901941147155, 0.2785631771008093, 0.2741239232824697, 0.15981313746022527, 0.25747672146369627, 0.029964963273792236, 0.21955696712076328, 0.5788320042274668, 0.17963751855335178, 0.15825426224934205, 0.07912713112467103, 0.7517077456843748, 0.899231077539863, 0.05451782573678066, 0.9268030375252713, 0.9823461981758542, 0.9870811190891049, 0.6451171802921192, 0.02103642979213432, 0.017530358160111933, 0.2103642979213432, 0.1051821489606716, 0.9524123927927464, 0.8053219076171152, 0.07158416956596579, 0.1073762543489487, 0.017896042391491448, 0.3132425715293507, 0.2709482158798181, 0.28680859924839286, 0.10044909466763989, 0.027755670895005757, 0.9187298536886657, 0.061248656912577706, 0.325506905180975, 0.014360598757984191, 0.5313421540454151, 0.11009792381121214, 0.023934331263306987, 0.7520026399270883, 0.0184918681949284, 0.0739674727797136, 0.11711516523454653, 0.04314769245483294, 0.9478928120005119, 0.8920302429472394, 0.9524200784510594, 0.3060925590254893, 0.020406170601699285, 0.020406170601699285, 0.6529974592543771, 0.27539236330332073, 0.36301720617255917, 0.20654427247749055, 0.15021401634726586, 0.925020814061837, 0.05781380087886481, 0.8825768413073362, 0.10538230940983118, 0.13460057561055858, 0.044866858536852856, 0.7627365951264986, 0.5526016224720863, 0.06065139758839972, 0.330213164647954, 0.05391235341191086, 0.9622319708929222, 0.8603339020392102, 0.07821217291265548, 0.1856173616739024, 0.7424694466956095, 0.058442995815589316, 0.39936047140652703, 0.07792399442078575, 0.019480998605196437, 0.4480629679195181, 0.11261159540468618, 0.2764102796296843, 0.01023741776406238, 0.593770230315618, 0.26879282037446783, 0.13439641018723392, 0.019199487169604842, 0.14399615377203634, 0.4415882049009114, 0.7721631248031126, 0.2278514138763283, 0.9315134060866537, 0.05479490624039139, 0.10744109706499323, 0.8058082279874492, 0.9419065175972028, 0.0443618186200691, 0.7985127351612438, 0.13308545586020729, 0.0443618186200691, 0.9355775699868909, 0.29825809531848835, 0.05422874460336152, 0.3999369914497912, 0.04745015152794133, 0.2033577922626057, 0.9260068180840059, 0.057831834005921096, 0.8674775100888165, 0.012666453103641683, 0.1519974372437002, 0.012666453103641683, 0.34199423379832544, 0.48132521793838395, 0.07848538252661696, 0.8633392077927866, 0.05516528791941778, 0.09194214653236296, 0.6252065964200682, 0.2206611516776711, 0.27879884168606683, 0.19075710220625627, 0.5282504368788635, 0.15700878266572205, 0.7850439133286102, 0.8044816566153835, 0.19526253801344262, 0.3037211843255253, 0.20574660873664619, 0.48987287794439566, 0.6051064435873716, 0.2410586645185464, 0.11314998538625648, 0.03935651665608921, 0.21060789983902417, 0.10530394991951209, 0.06701160449423496, 0.5935313540917955, 0.019146172712638562, 0.8341035393176932, 0.15714994219029002, 0.7892095813413494, 0.13153493022355822, 0.0584599689882481, 0.02922998449412405, 0.941897557121789, 0.13152175949898398, 0.7891305569939039, 0.9321849572495016, 0.6831106590230304, 0.07066661989893418, 0.21199985969680252, 0.06550977900884862, 0.9171369061238808, 0.5355706829853658, 0.44630890248780486, 0.0895229968709614, 0.0447614984354807, 0.8504684702741332, 0.8055493998265189, 0.18153225911583523, 0.07039494189601045, 0.9151342446481358, 0.9475326276654792, 0.8136948949941187, 0.1356158158323531, 0.6329165514169272, 0.03669081457489433, 0.1559359619433009, 0.17428136923074808, 0.8021137940050543, 0.07129900391156038, 0.12477325684523068, 0.07031180334089354, 0.8437416400907225, 0.9620335574626215, 0.012143201019360185, 0.012143201019360185, 0.19429121630976295, 0.7771648652390518, 0.07876044478546196, 0.8663648926400815, 0.9621445253592945, 0.9750206439181418, 0.9418802186853792, 0.15920407997942407, 0.2122721066392321, 0.6102823065877923, 0.1281256018833297, 0.845628972429976, 0.21520208128861074, 0.10564465808713618, 0.019563825571691885, 0.2856318533467015, 0.37953821609082256, 0.1587846652086229, 0.021171288694483054, 0.38108319650069494, 0.4340114182369026, 0.24856949951496818, 0.261482200788473, 0.058107155730771784, 0.2808512526987303, 0.14849606464530568, 0.05768477985430631, 0.5274037015250863, 0.23073911941722525, 0.18541536381741314, 0.16242368401581678, 0.8353218035099148, 0.2498968241940635, 0.14897695288492246, 0.00961141631515629, 0.18742261814554761, 0.4036794852365641, 0.952125844361008, 0.2708220263812432, 0.6921007340853993, 0.045179507756093404, 0.06776926163414011, 0.2597821695975371, 0.6212182316462843, 0.8646690317918704, 0.07860627561744275, 0.18007703336253505, 0.1406315879593131, 0.02572529048036215, 0.2658280016304089, 0.38587935720543226, 0.972001716789299, 0.5929826126440884, 0.40496373546425546, 0.9764766634179671, 0.7446944067716589, 0.2510205865522446, 0.006253538547878061, 0.2751556961066347, 0.10005661676604898, 0.619100316239928, 0.8920304261356082, 0.9893736462091094, 0.9433515003676698, 0.06543173936198635, 0.9160443510678089, 0.9933518976567179, 0.9473121271374221, 0.024131348956260863, 0.3861015833001738, 0.20752960102384344, 0.20752960102384344, 0.17374571248507822, 0.0954767058630524, 0.8592903527674716, 0.1155088609283819, 0.8778673430557025, 0.7774987651573942, 0.15549975303147887, 0.9780552545845487, 0.11319595508209483, 0.8489696631157112, 0.8268490842415983, 0.1459145442779291, 0.0815839808508495, 0.8566317989339197, 0.09698158146707608, 0.8728342332036848, 0.023477449741010688, 0.4460715450792031, 0.023477449741010688, 0.21912286424943309, 0.28955521347246516, 0.9357698233752888, 0.04678849116876444, 0.02339424558438222, 0.14234583979347137, 0.3764256652316243, 0.12652963537197454, 0.07908102210748409, 0.27520195693404464, 0.9859564259283802, 0.9433995684812839, 0.31935822871457664, 0.19161493722874598, 0.46484364401788375, 0.014193699053981183, 0.010645274290485887, 0.3796729000747945, 0.14042696304136235, 0.39787639528386, 0.010401997262323137, 0.07021348152068117, 0.4129571825222258, 0.5745491235091836, 0.9668065984007156, 0.08606260990650148, 0.12909391485975222, 0.7745634891585134, 0.960455232352498, 0.7327950902606736, 0.2442650300868912, 0.2926359183019097, 0.6856041514501885, 0.016722052474394844, 0.11216849510137207, 0.011216849510137206, 0.30285493677370456, 0.5832761745271348, 0.9321787386084375, 0.07829873136329779, 0.8612860449962756, 0.8353671176481906, 0.18563713725515346, 0.25259447365410237, 0.408962481154261, 0.22853785711561642, 0.02405661653848594, 0.08419815788470078, 0.940142351846292, 0.9801880728866486, 0.9538472733134173, 0.17071527851253893, 0.2048583342150467, 0.10242916710752335, 0.5235268541051193, 0.13128966392926456, 0.8533828155402197, 0.9638084078015283, 0.38327897508258596, 0.3140116904291066, 0.07850292260727665, 0.11544547442246565, 0.1062098364686684, 0.9431039052477094, 0.13129238271265112, 0.8534004876322323, 0.08453958233401589, 0.6199569371161164, 0.2113489558350397, 0.08453958233401589, 0.9797535674291485, 0.9566833182287299, 0.2249806072261117, 0.36207816475452353, 0.13358223554050383, 0.049214507830711934, 0.2320112512019277, 0.961060855816314, 0.9461541297285906, 0.94615308818586, 0.9461491581385527, 0.5350503109220266, 0.46039212800267404, 0.15186456880967739, 0.1603014892991039, 0.6833905596435482, 0.048061625621047085, 0.7449551971262298, 0.2042619088894501, 0.13643992986628686, 0.8527495616642929, 0.3288818854211916, 0.22610629622706926, 0.1644409427105958, 0.27749409082413046, 0.24622287157115316, 0.06155571789278829, 0.041037145261858865, 0.6360757515588124, 0.8394311802647091, 0.09593499203025246, 0.0599593700189078, 0.9877311139661314, 0.4036593316990426, 0.5189905693273406, 0.0768874917521986, 0.17689366180578772, 0.5948049378219612, 0.2277505895749517, 0.020423185703709903, 0.040846371407419806, 0.9394665423706555, 0.5678735934649588, 0.1801906594648427, 0.2347938896057041, 0.005460323014086142, 0.016380969042258425, 0.16428760557176678, 0.7168913697677095, 0.11948189496128492, 0.2980159353188052, 0.12608366494257142, 0.02292430271683117, 0.5501832652039481, 0.026170960257857267, 0.9421545692828616, 0.09281894635344046, 0.9281894635344046, 0.9461344999638479, 0.06772947208158518, 0.8804831370606073, 0.9732058553670263, 0.9796456572062556, 0.06204311012663018, 0.8996250968361376, 0.30813280787770464, 0.5135546797961744, 0.14122753694394796, 0.0320971674872609, 0.9870832231899817, 0.9796806755021756, 0.0756961180257401, 0.8326572982831412, 0.9461180280962641, 0.581410192684466, 0.24618269419972885, 0.16761374924236858, 0.43393276001519926, 0.14243594412712648, 0.38921449848691536, 0.007453043588047315, 0.026499710535279344, 0.7657314446000348, 0.17016254324445218, 0.1219578320651332, 0.06969018975150469, 0.801437182142304, 0.008711273718938086, 0.11816200751309984, 0.854402208171645, 0.027268155579946118, 0.9889242343761432, 0.9699208972940113, 0.8629536864492754, 0.044254035202526944, 0.08850807040505389, 0.9368257240978913], "Term": ["acidity", "acidity", "acidity", "acidity", "acidity", "additional", "age", "age", "age", "aging", "aging", "aging", "alongside", "amounts", "anise", "anise", "aperitif", "apple", "apple", "apples", "apples", "apricot", "apricot", "apricots", "aromas", "aromas", "aromas", "aromas", "aromas", "attractive", "attractive", "attractive", "attractive", "avola", "bacon", "baked", "baked", "balanced", "balanced", "balanced", "balanced", "balanced", "balancing", "balsamic", "barrel", "barrel", "barrel", "basic", "bell", "berry", "berry", "berry", "berry", "berry", "better", "better", "better", "bit", "bit", "bit", "black", "black", "black", "blackberries", "blackberries", "blackberry", "blackberry", "blackberry", "blackberry", "blanc", "blend", "blend", "blend", "blend", "blend", "blue", "bodied", "bodied", "bodied", "bodied", "bodied", "bright", "bright", "bright", "bright", "briny", "butterscotch", "cab", "cab", "cabernet", "cabernet", "cassis", "cassis", "cassis", "cassis", "cedar", "cedar", "cellar", "cellar", "chalky", "chalky", "character", "character", "character", "character", "chardonnay", "chardonnay", "cherries", "cherries", "cherry", "cherry", "cherry", "cherry", "chocolate", "chocolate", "chocolate", "chopped", "chopped", "chunky", "cigar", "cigar", "cigar", "citric", "citric", "citrus", "citrus", "clean", "clean", "clean", "clean", "clove", "clove", "cola", "cola", "cola", "concentrated", "concentrated", "concentrated", "concentrated", "concentrated", "cooked", "cooked", "cranberry", "cranberry", "creamy", "creamy", "creamy", "crisp", "crisp", "crispness", "cru", "crushed", "crushed", "crushed", "currant", "currant", "currant", "currant", "currant", "currants", "currants", "currants", "dark", "dark", "dark", "delivers", "delivers", "delivers", "delivers", "dense", "dense", "dense", "density", "density", "density", "dessert", "develop", "dried", "dried", "dried", "dried", "drink", "drink", "drink", "drink", "drink", "dry", "dry", "dry", "dry", "elderberry", "enjoy", "enjoy", "enjoy", "extracted", "extracted", "extremely", "family", "family", "family", "feels", "fermented", "fermented", "filling", "fine", "fine", "fine", "fine", "finish", "finish", "finish", "finish", "finish", "firm", "firm", "flavors", "flavors", "flavors", "flavors", "flavors", "floral", "floral", "floral", "forest", "forest", "forest", "fragrance", "framed", "framed", "franc", "french", "fresh", "fresh", "fresh", "fresh", "fresh", "freshly", "freshness", "freshness", "freshness", "freshness", "fruit", "fruit", "fruit", "fruit", "fruit", "fruitiness", "fruitiness", "fruits", "fruits", "fruits", "fruits", "fruits", "fruity", "fruity", "fruity", "fruity", "fruity", "future", "game", "generic", "glass", "glass", "glass", "glass", "good", "good", "good", "good", "gooseberry", "gooseberry", "grapefruit", "grapefruit", "graphite", "graphite", "graphite", "green", "green", "green", "green", "grenache", "hay", "hay", "help", "help", "herb", "herb", "herb", "herb", "herb", "herbal", "herbal", "herbal", "herbal", "hint", "hint", "hint", "hint", "hint", "honey", "honey", "impressive", "impressive", "include", "include", "informal", "jam", "jam", "jam", "jam", "jasmine", "juicy", "juicy", "juicy", "juicy", "juicy", "kiwi", "lavender", "lavender", "lead", "lead", "lead", "lead", "lead", "leafy", "leafy", "lean", "lean", "lean", "lean", "leather", "leather", "leather", "leaving", "leaving", "lemon", "lemon", "licorice", "licorice", "licorice", "light", "light", "light", "light", "like", "like", "like", "like", "like", "lime", "lime", "lively", "lively", "lively", "lively", "luminous", "luscious", "luscious", "mango", "meat", "meat", "meat", "meet", "meet", "melon", "melon", "menthol", "menthol", "menthol", "merlot", "merlot", "mild", "mild", "mildly", "milk", "milk", "mineral", "mineral", "mineral", "mineral", "minerality", "minerality", "minerality", "mingle", "mingle", "mourvedre", "mouth", "mouth", "mouth", "mouth", "napa", "napa", "need", "needs", "nero", "new", "new", "new", "nice", "nice", "nose", "nose", "nose", "nose", "nose", "note", "note", "note", "note", "notes", "notes", "notes", "notes", "notes", "oak", "oak", "oak", "oak", "oaky", "oaky", "offers", "offers", "offers", "offers", "offers", "oily", "old", "old", "opens", "opens", "opens", "opens", "overripe", "overripe", "palate", "palate", "palate", "palate", "palate", "passion", "peach", "peach", "peaches", "pear", "pear", "pepper", "pepper", "pepper", "pepper", "petal", "petit", "petite", "pine", "pine", "pineapple", "pink", "plum", "plum", "plum", "plum", "plum", "potential", "potential", "powerful", "powerful", "property", "property", "prosecco", "provide", "provide", "purple", "purple", "raisin", "raisin", "raspberries", "raspberries", "raspberry", "raspberry", "raspberry", "raspberry", "raspberry", "ready", "ready", "ready", "red", "red", "red", "red", "red", "refreshing", "rhubarb", "rich", "rich", "rich", "rich", "rich", "ripe", "ripe", "ripe", "ripe", "ripe", "rose", "rose", "rubbery", "sage", "sage", "sage", "salty", "sangiovese", "sangiovese", "sauvignon", "sauvignon", "sauvignon", "savory", "savory", "savory", "savory", "screwcap", "selection", "selection", "shiraz", "shiraz", "shows", "shows", "shows", "shows", "shows", "sirah", "skinned", "skins", "slightly", "slightly", "slightly", "slightly", "slowly", "slowly", "smells", "soft", "soft", "soft", "soft", "soft", "soften", "soil", "soil", "solid", "solid", "solid", "solid", "soon", "sparkler", "spice", "spice", "spice", "spice", "spice", "spring", "stainless", "steel", "steely", "stone", "stone", "structure", "structure", "structure", "structured", "structured", "structured", "sugar", "sugar", "sweet", "sweet", "sweet", "sweet", "sweetness", "sweetness", "sweetness", "sweetness", "syrah", "syrah", "syrah", "tangerine", "tannic", "tannic", "tannic", "tannins", "tannins", "tannins", "tastes", "tastes", "tastes", "texture", "texture", "texture", "texture", "texture", "time", "time", "time", "tobacco", "tobacco", "tobacco", "tobacco", "tones", "tones", "tonic", "tonic", "touched", "tough", "tough", "tropical", "underbrush", "valley", "valley", "vanilla", "vanilla", "vanilla", "vanilla", "verdot", "violet", "violets", "violets", "vivid", "white", "white", "white", "wine", "wine", "wine", "wine", "wine", "winemaker", "winemaker", "wood", "wood", "wood", "wood", "years", "years", "years", "yellow", "zest", "zesty", "zesty", "zesty", "zinfandel"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 5, 2, 4, 3]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el322019902666585923079103309", ldavis_el322019902666585923079103309_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://d3js.org/d3.v5"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
        new LDAvis("#" + "ldavis_el322019902666585923079103309", ldavis_el322019902666585923079103309_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
         LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el322019902666585923079103309", ldavis_el322019902666585923079103309_data);
            })
         });
}
</script>



***

## How to work with topic models' output


***


```python
# create a doc-topic matrix

path = 'C:/Users/User/Desktop' # Change this path to a preferred location on your computer
os.chdir(path)

import numpy as np

filenames = df['index'].values.astype('U')

dates = df['date'].values.astype('U') # its better to use the date string here

dtm_transformed = tf_vectorizer.fit_transform(df['description'].values.astype('U'))

doctopic = lda_tf.fit_transform(dtm_transformed)

doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

# Write doctopic to a csv file
os.chdir(path)  

# filenamesclean = [fn.split('\\')[-1] for fn in filenames]
i=0
with open('doctopic_wine.csv',mode='w') as fo:
    for rij in doctopic:
        fo.write('"'+filenames[i]+'"')
        fo.write(',')
        fo.write('"'+dates[i]+'"')
        fo.write(',')
        for kolom in rij:
            fo.write(str(kolom))
            fo.write(',')
        fo.write('\n')
        i+=1
print("finsihed with creating doctopic matrix")
```

    finsihed with creating doctopic matrix
    


```python
dfm = pd.read_csv('C:/Users/User/Desktop/doctopic_wine.csv', header=None, index_col=False,
                  names = ["file", "date", "t_0", "t_1","t_2", "t_3", "t_4"])

# dfm['datetime'] = pd.to_datetime(dfm['date'], format='%d/%m/%Y')
dfm.head()
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
      <th>file</th>
      <th>date</th>
      <th>t_0</th>
      <th>t_1</th>
      <th>t_2</th>
      <th>t_3</th>
      <th>t_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2013.0</td>
      <td>0.259376</td>
      <td>0.150591</td>
      <td>0.362116</td>
      <td>0.217230</td>
      <td>0.010686</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2011.0</td>
      <td>0.231586</td>
      <td>0.732592</td>
      <td>0.011970</td>
      <td>0.011885</td>
      <td>0.011967</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2013.0</td>
      <td>0.946179</td>
      <td>0.013425</td>
      <td>0.013369</td>
      <td>0.013508</td>
      <td>0.013520</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2013.0</td>
      <td>0.574284</td>
      <td>0.074802</td>
      <td>0.010157</td>
      <td>0.330554</td>
      <td>0.010203</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2012.0</td>
      <td>0.015628</td>
      <td>0.015812</td>
      <td>0.015809</td>
      <td>0.207511</td>
      <td>0.745240</td>
    </tr>
  </tbody>
</table>
</div>




```python
# calculate mean, std, cutoff high, and cutoff low
dfm1 = dfm.describe().loc[['mean','std']]
dfm2 = dfm1.transpose()
dfm2['cutoff_low'] = dfm2['mean'] + dfm2['std'] 
# dfm2.reset_index(level=0, inplace=True)

# Drop first two rows
dfm2 = dfm2.iloc[2:]
dfm2.reset_index(level=0, inplace=True)
dfm2
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
      <th>mean</th>
      <th>std</th>
      <th>cutoff_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>t_0</td>
      <td>0.308767</td>
      <td>0.348688</td>
      <td>0.657455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>t_1</td>
      <td>0.194721</td>
      <td>0.279242</td>
      <td>0.473962</td>
    </tr>
    <tr>
      <th>2</th>
      <td>t_2</td>
      <td>0.120975</td>
      <td>0.217141</td>
      <td>0.338116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>t_3</td>
      <td>0.160273</td>
      <td>0.247667</td>
      <td>0.407940</td>
    </tr>
    <tr>
      <th>4</th>
      <td>t_4</td>
      <td>0.215264</td>
      <td>0.299245</td>
      <td>0.514509</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get cutoff_low from dfm2
d = {}
for i, row in dfm2.iterrows():
    d['t_{}_cutoff_low'.format(i)] = dfm2.at[i,'cutoff_low']
print(d)
```

    {'t_0_cutoff_low': 0.6574552316170613, 't_1_cutoff_low': 0.473962220387246, 't_2_cutoff_low': 0.338115691126619, 't_3_cutoff_low': 0.407940380706719, 't_4_cutoff_low': 0.5145088774248614}
    


```python
%%time
for column in dfm.columns[-5:]:
    dfm['{}_low'.format(column)]=dfm['{}'.format(column)].apply(lambda x: 1 if x> d['{}_cutoff_low'.format(column)] else 0)

dfm['datetime'] = pd.to_datetime(df['date'], errors = 'coerce')
dfm

```

    Wall time: 20.9 ms
    




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
      <th>file</th>
      <th>date</th>
      <th>t_0</th>
      <th>t_1</th>
      <th>t_2</th>
      <th>t_3</th>
      <th>t_4</th>
      <th>t_0_low</th>
      <th>t_1_low</th>
      <th>t_2_low</th>
      <th>t_3_low</th>
      <th>t_4_low</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2013.0</td>
      <td>0.259376</td>
      <td>0.150591</td>
      <td>0.362116</td>
      <td>0.217230</td>
      <td>0.010686</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2011.0</td>
      <td>0.231586</td>
      <td>0.732592</td>
      <td>0.011970</td>
      <td>0.011885</td>
      <td>0.011967</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2013.0</td>
      <td>0.946179</td>
      <td>0.013425</td>
      <td>0.013369</td>
      <td>0.013508</td>
      <td>0.013520</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2013.0</td>
      <td>0.574284</td>
      <td>0.074802</td>
      <td>0.010157</td>
      <td>0.330554</td>
      <td>0.010203</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2012.0</td>
      <td>0.015628</td>
      <td>0.015812</td>
      <td>0.015809</td>
      <td>0.207511</td>
      <td>0.745240</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2012-01-01</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>1995</td>
      <td>2013.0</td>
      <td>0.019005</td>
      <td>0.018267</td>
      <td>0.347313</td>
      <td>0.595944</td>
      <td>0.019471</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2013-01-01</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>1996</td>
      <td>2013.0</td>
      <td>0.011941</td>
      <td>0.320276</td>
      <td>0.011846</td>
      <td>0.012219</td>
      <td>0.643719</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2013-01-01</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>1997</td>
      <td>2013.0</td>
      <td>0.021026</td>
      <td>0.229797</td>
      <td>0.020542</td>
      <td>0.708360</td>
      <td>0.020275</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2013-01-01</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>1998</td>
      <td>2013.0</td>
      <td>0.946296</td>
      <td>0.013514</td>
      <td>0.013357</td>
      <td>0.013420</td>
      <td>0.013413</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-01</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>1999</td>
      <td>2013.0</td>
      <td>0.952355</td>
      <td>0.012014</td>
      <td>0.011819</td>
      <td>0.011887</td>
      <td>0.011925</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-01</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 13 columns</p>
</div>




```python
# Create five topic model time series datasets
g_1 = dfm.set_index('datetime').resample('A-DEC')['t_0_low'].sum()
g_1 = g_1.reset_index()

g_2 = dfm.set_index('datetime').resample('A-DEC')['t_1_low'].sum()
g_2 = g_2.reset_index()

g_3 = dfm.set_index('datetime').resample('A-DEC')['t_2_low'].sum()
g_3 = g_3.reset_index()

g_4 = dfm.set_index('datetime').resample('A-DEC')['t_3_low'].sum()
g_4 = g_4.reset_index()

g_5 = dfm.set_index('datetime').resample('A-DEC')['t_4_low'].sum()
g_5 = g_5.reset_index()
```


```python
# Merge
dfs = [g_1, g_2, g_3, g_4, g_5]
from functools import reduce
df_topic_year = reduce(lambda  left,right: pd.merge(left,right,on=['datetime'],
                                            how='left'), dfs)
df_topic_year
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
      <th>t_0_low</th>
      <th>t_1_low</th>
      <th>t_2_low</th>
      <th>t_3_low</th>
      <th>t_4_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1877-12-31</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1878-12-31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1879-12-31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1880-12-31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1881-12-31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2013-12-31</td>
      <td>51</td>
      <td>51</td>
      <td>60</td>
      <td>33</td>
      <td>52</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2014-12-31</td>
      <td>73</td>
      <td>47</td>
      <td>42</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2015-12-31</td>
      <td>55</td>
      <td>19</td>
      <td>21</td>
      <td>42</td>
      <td>11</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2016-12-31</td>
      <td>36</td>
      <td>2</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2017-12-31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>141 rows × 6 columns</p>
</div>




```python
# Plotly graph with topic year data
# The code of this graph allows the use of two y axis (dual axis), but we will not use that here
import plotly.graph_objects as go
from plotly.subplots import make_subplots

x = df_topic_year['datetime']

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_0_low'], name="Topic 0", opacity=0.7, line=dict(color='#3A405A', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_1_low'], name="Topic 1", opacity=0.7, line=dict(color='#99B2DD', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_2_low'], name="Topic 2", opacity=0.7, line=dict(color='#E9AFA3', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_3_low'], name="Topic 3", opacity=0.7, line=dict(color='#685044', width=2)),
    secondary_y=False)

fig.add_trace(
    go.Scatter(x=x, y=df_topic_year['t_4_low'], name="Topic 4", opacity=0.7, line=dict(color='#F9DEC9', width=2)),
    secondary_y=False)

fig.update_layout(showlegend=True,
    xaxis_rangeslider_visible=True,
    width=1200,
    height=600,    
    annotations=[
            dict(
            x="2014-12-31",
            y=0,         
            text="2014 has the highest number of reviewed wines",
            ax=100,
            ay=-350,
            opacity=0.5,
            arrowhead=7)])


fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(title_text="Year", showgrid=True, gridwidth=0.3, gridcolor='LightGrey')
fig.update_yaxes(title_text="# Wines reviewed", showgrid=True, gridwidth=0.3, gridcolor='LightGrey')
# fig.update_yaxes(title_text="Something", showgrid=False, secondary_y=True)
fig.show()
```


<div>                            <div id="ee02ed71-0363-4d72-9b09-48f9304368f0" class="plotly-graph-div" style="height:600px; width:1200px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ee02ed71-0363-4d72-9b09-48f9304368f0")) {                    Plotly.newPlot(                        "ee02ed71-0363-4d72-9b09-48f9304368f0",                        [{"line":{"color":"#3A405A","width":2},"name":"Topic 0","opacity":0.7,"x":["1877-12-31T00:00:00","1878-12-31T00:00:00","1879-12-31T00:00:00","1880-12-31T00:00:00","1881-12-31T00:00:00","1882-12-31T00:00:00","1883-12-31T00:00:00","1884-12-31T00:00:00","1885-12-31T00:00:00","1886-12-31T00:00:00","1887-12-31T00:00:00","1888-12-31T00:00:00","1889-12-31T00:00:00","1890-12-31T00:00:00","1891-12-31T00:00:00","1892-12-31T00:00:00","1893-12-31T00:00:00","1894-12-31T00:00:00","1895-12-31T00:00:00","1896-12-31T00:00:00","1897-12-31T00:00:00","1898-12-31T00:00:00","1899-12-31T00:00:00","1900-12-31T00:00:00","1901-12-31T00:00:00","1902-12-31T00:00:00","1903-12-31T00:00:00","1904-12-31T00:00:00","1905-12-31T00:00:00","1906-12-31T00:00:00","1907-12-31T00:00:00","1908-12-31T00:00:00","1909-12-31T00:00:00","1910-12-31T00:00:00","1911-12-31T00:00:00","1912-12-31T00:00:00","1913-12-31T00:00:00","1914-12-31T00:00:00","1915-12-31T00:00:00","1916-12-31T00:00:00","1917-12-31T00:00:00","1918-12-31T00:00:00","1919-12-31T00:00:00","1920-12-31T00:00:00","1921-12-31T00:00:00","1922-12-31T00:00:00","1923-12-31T00:00:00","1924-12-31T00:00:00","1925-12-31T00:00:00","1926-12-31T00:00:00","1927-12-31T00:00:00","1928-12-31T00:00:00","1929-12-31T00:00:00","1930-12-31T00:00:00","1931-12-31T00:00:00","1932-12-31T00:00:00","1933-12-31T00:00:00","1934-12-31T00:00:00","1935-12-31T00:00:00","1936-12-31T00:00:00","1937-12-31T00:00:00","1938-12-31T00:00:00","1939-12-31T00:00:00","1940-12-31T00:00:00","1941-12-31T00:00:00","1942-12-31T00:00:00","1943-12-31T00:00:00","1944-12-31T00:00:00","1945-12-31T00:00:00","1946-12-31T00:00:00","1947-12-31T00:00:00","1948-12-31T00:00:00","1949-12-31T00:00:00","1950-12-31T00:00:00","1951-12-31T00:00:00","1952-12-31T00:00:00","1953-12-31T00:00:00","1954-12-31T00:00:00","1955-12-31T00:00:00","1956-12-31T00:00:00","1957-12-31T00:00:00","1958-12-31T00:00:00","1959-12-31T00:00:00","1960-12-31T00:00:00","1961-12-31T00:00:00","1962-12-31T00:00:00","1963-12-31T00:00:00","1964-12-31T00:00:00","1965-12-31T00:00:00","1966-12-31T00:00:00","1967-12-31T00:00:00","1968-12-31T00:00:00","1969-12-31T00:00:00","1970-12-31T00:00:00","1971-12-31T00:00:00","1972-12-31T00:00:00","1973-12-31T00:00:00","1974-12-31T00:00:00","1975-12-31T00:00:00","1976-12-31T00:00:00","1977-12-31T00:00:00","1978-12-31T00:00:00","1979-12-31T00:00:00","1980-12-31T00:00:00","1981-12-31T00:00:00","1982-12-31T00:00:00","1983-12-31T00:00:00","1984-12-31T00:00:00","1985-12-31T00:00:00","1986-12-31T00:00:00","1987-12-31T00:00:00","1988-12-31T00:00:00","1989-12-31T00:00:00","1990-12-31T00:00:00","1991-12-31T00:00:00","1992-12-31T00:00:00","1993-12-31T00:00:00","1994-12-31T00:00:00","1995-12-31T00:00:00","1996-12-31T00:00:00","1997-12-31T00:00:00","1998-12-31T00:00:00","1999-12-31T00:00:00","2000-12-31T00:00:00","2001-12-31T00:00:00","2002-12-31T00:00:00","2003-12-31T00:00:00","2004-12-31T00:00:00","2005-12-31T00:00:00","2006-12-31T00:00:00","2007-12-31T00:00:00","2008-12-31T00:00:00","2009-12-31T00:00:00","2010-12-31T00:00:00","2011-12-31T00:00:00","2012-12-31T00:00:00","2013-12-31T00:00:00","2014-12-31T00:00:00","2015-12-31T00:00:00","2016-12-31T00:00:00","2017-12-31T00:00:00"],"y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,9,6,1,1,7,32,33,20,10,19,26,30,51,73,55,36,0],"type":"scatter","xaxis":"x","yaxis":"y"},{"line":{"color":"#99B2DD","width":2},"name":"Topic 1","opacity":0.7,"x":["1877-12-31T00:00:00","1878-12-31T00:00:00","1879-12-31T00:00:00","1880-12-31T00:00:00","1881-12-31T00:00:00","1882-12-31T00:00:00","1883-12-31T00:00:00","1884-12-31T00:00:00","1885-12-31T00:00:00","1886-12-31T00:00:00","1887-12-31T00:00:00","1888-12-31T00:00:00","1889-12-31T00:00:00","1890-12-31T00:00:00","1891-12-31T00:00:00","1892-12-31T00:00:00","1893-12-31T00:00:00","1894-12-31T00:00:00","1895-12-31T00:00:00","1896-12-31T00:00:00","1897-12-31T00:00:00","1898-12-31T00:00:00","1899-12-31T00:00:00","1900-12-31T00:00:00","1901-12-31T00:00:00","1902-12-31T00:00:00","1903-12-31T00:00:00","1904-12-31T00:00:00","1905-12-31T00:00:00","1906-12-31T00:00:00","1907-12-31T00:00:00","1908-12-31T00:00:00","1909-12-31T00:00:00","1910-12-31T00:00:00","1911-12-31T00:00:00","1912-12-31T00:00:00","1913-12-31T00:00:00","1914-12-31T00:00:00","1915-12-31T00:00:00","1916-12-31T00:00:00","1917-12-31T00:00:00","1918-12-31T00:00:00","1919-12-31T00:00:00","1920-12-31T00:00:00","1921-12-31T00:00:00","1922-12-31T00:00:00","1923-12-31T00:00:00","1924-12-31T00:00:00","1925-12-31T00:00:00","1926-12-31T00:00:00","1927-12-31T00:00:00","1928-12-31T00:00:00","1929-12-31T00:00:00","1930-12-31T00:00:00","1931-12-31T00:00:00","1932-12-31T00:00:00","1933-12-31T00:00:00","1934-12-31T00:00:00","1935-12-31T00:00:00","1936-12-31T00:00:00","1937-12-31T00:00:00","1938-12-31T00:00:00","1939-12-31T00:00:00","1940-12-31T00:00:00","1941-12-31T00:00:00","1942-12-31T00:00:00","1943-12-31T00:00:00","1944-12-31T00:00:00","1945-12-31T00:00:00","1946-12-31T00:00:00","1947-12-31T00:00:00","1948-12-31T00:00:00","1949-12-31T00:00:00","1950-12-31T00:00:00","1951-12-31T00:00:00","1952-12-31T00:00:00","1953-12-31T00:00:00","1954-12-31T00:00:00","1955-12-31T00:00:00","1956-12-31T00:00:00","1957-12-31T00:00:00","1958-12-31T00:00:00","1959-12-31T00:00:00","1960-12-31T00:00:00","1961-12-31T00:00:00","1962-12-31T00:00:00","1963-12-31T00:00:00","1964-12-31T00:00:00","1965-12-31T00:00:00","1966-12-31T00:00:00","1967-12-31T00:00:00","1968-12-31T00:00:00","1969-12-31T00:00:00","1970-12-31T00:00:00","1971-12-31T00:00:00","1972-12-31T00:00:00","1973-12-31T00:00:00","1974-12-31T00:00:00","1975-12-31T00:00:00","1976-12-31T00:00:00","1977-12-31T00:00:00","1978-12-31T00:00:00","1979-12-31T00:00:00","1980-12-31T00:00:00","1981-12-31T00:00:00","1982-12-31T00:00:00","1983-12-31T00:00:00","1984-12-31T00:00:00","1985-12-31T00:00:00","1986-12-31T00:00:00","1987-12-31T00:00:00","1988-12-31T00:00:00","1989-12-31T00:00:00","1990-12-31T00:00:00","1991-12-31T00:00:00","1992-12-31T00:00:00","1993-12-31T00:00:00","1994-12-31T00:00:00","1995-12-31T00:00:00","1996-12-31T00:00:00","1997-12-31T00:00:00","1998-12-31T00:00:00","1999-12-31T00:00:00","2000-12-31T00:00:00","2001-12-31T00:00:00","2002-12-31T00:00:00","2003-12-31T00:00:00","2004-12-31T00:00:00","2005-12-31T00:00:00","2006-12-31T00:00:00","2007-12-31T00:00:00","2008-12-31T00:00:00","2009-12-31T00:00:00","2010-12-31T00:00:00","2011-12-31T00:00:00","2012-12-31T00:00:00","2013-12-31T00:00:00","2014-12-31T00:00:00","2015-12-31T00:00:00","2016-12-31T00:00:00","2017-12-31T00:00:00"],"y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2,6,7,15,12,14,18,48,23,39,26,51,47,19,2,0],"type":"scatter","xaxis":"x","yaxis":"y"},{"line":{"color":"#E9AFA3","width":2},"name":"Topic 2","opacity":0.7,"x":["1877-12-31T00:00:00","1878-12-31T00:00:00","1879-12-31T00:00:00","1880-12-31T00:00:00","1881-12-31T00:00:00","1882-12-31T00:00:00","1883-12-31T00:00:00","1884-12-31T00:00:00","1885-12-31T00:00:00","1886-12-31T00:00:00","1887-12-31T00:00:00","1888-12-31T00:00:00","1889-12-31T00:00:00","1890-12-31T00:00:00","1891-12-31T00:00:00","1892-12-31T00:00:00","1893-12-31T00:00:00","1894-12-31T00:00:00","1895-12-31T00:00:00","1896-12-31T00:00:00","1897-12-31T00:00:00","1898-12-31T00:00:00","1899-12-31T00:00:00","1900-12-31T00:00:00","1901-12-31T00:00:00","1902-12-31T00:00:00","1903-12-31T00:00:00","1904-12-31T00:00:00","1905-12-31T00:00:00","1906-12-31T00:00:00","1907-12-31T00:00:00","1908-12-31T00:00:00","1909-12-31T00:00:00","1910-12-31T00:00:00","1911-12-31T00:00:00","1912-12-31T00:00:00","1913-12-31T00:00:00","1914-12-31T00:00:00","1915-12-31T00:00:00","1916-12-31T00:00:00","1917-12-31T00:00:00","1918-12-31T00:00:00","1919-12-31T00:00:00","1920-12-31T00:00:00","1921-12-31T00:00:00","1922-12-31T00:00:00","1923-12-31T00:00:00","1924-12-31T00:00:00","1925-12-31T00:00:00","1926-12-31T00:00:00","1927-12-31T00:00:00","1928-12-31T00:00:00","1929-12-31T00:00:00","1930-12-31T00:00:00","1931-12-31T00:00:00","1932-12-31T00:00:00","1933-12-31T00:00:00","1934-12-31T00:00:00","1935-12-31T00:00:00","1936-12-31T00:00:00","1937-12-31T00:00:00","1938-12-31T00:00:00","1939-12-31T00:00:00","1940-12-31T00:00:00","1941-12-31T00:00:00","1942-12-31T00:00:00","1943-12-31T00:00:00","1944-12-31T00:00:00","1945-12-31T00:00:00","1946-12-31T00:00:00","1947-12-31T00:00:00","1948-12-31T00:00:00","1949-12-31T00:00:00","1950-12-31T00:00:00","1951-12-31T00:00:00","1952-12-31T00:00:00","1953-12-31T00:00:00","1954-12-31T00:00:00","1955-12-31T00:00:00","1956-12-31T00:00:00","1957-12-31T00:00:00","1958-12-31T00:00:00","1959-12-31T00:00:00","1960-12-31T00:00:00","1961-12-31T00:00:00","1962-12-31T00:00:00","1963-12-31T00:00:00","1964-12-31T00:00:00","1965-12-31T00:00:00","1966-12-31T00:00:00","1967-12-31T00:00:00","1968-12-31T00:00:00","1969-12-31T00:00:00","1970-12-31T00:00:00","1971-12-31T00:00:00","1972-12-31T00:00:00","1973-12-31T00:00:00","1974-12-31T00:00:00","1975-12-31T00:00:00","1976-12-31T00:00:00","1977-12-31T00:00:00","1978-12-31T00:00:00","1979-12-31T00:00:00","1980-12-31T00:00:00","1981-12-31T00:00:00","1982-12-31T00:00:00","1983-12-31T00:00:00","1984-12-31T00:00:00","1985-12-31T00:00:00","1986-12-31T00:00:00","1987-12-31T00:00:00","1988-12-31T00:00:00","1989-12-31T00:00:00","1990-12-31T00:00:00","1991-12-31T00:00:00","1992-12-31T00:00:00","1993-12-31T00:00:00","1994-12-31T00:00:00","1995-12-31T00:00:00","1996-12-31T00:00:00","1997-12-31T00:00:00","1998-12-31T00:00:00","1999-12-31T00:00:00","2000-12-31T00:00:00","2001-12-31T00:00:00","2002-12-31T00:00:00","2003-12-31T00:00:00","2004-12-31T00:00:00","2005-12-31T00:00:00","2006-12-31T00:00:00","2007-12-31T00:00:00","2008-12-31T00:00:00","2009-12-31T00:00:00","2010-12-31T00:00:00","2011-12-31T00:00:00","2012-12-31T00:00:00","2013-12-31T00:00:00","2014-12-31T00:00:00","2015-12-31T00:00:00","2016-12-31T00:00:00","2017-12-31T00:00:00"],"y":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,3,4,9,5,8,12,20,29,46,60,42,21,3,0],"type":"scatter","xaxis":"x","yaxis":"y"},{"line":{"color":"#685044","width":2},"name":"Topic 3","opacity":0.7,"x":["1877-12-31T00:00:00","1878-12-31T00:00:00","1879-12-31T00:00:00","1880-12-31T00:00:00","1881-12-31T00:00:00","1882-12-31T00:00:00","1883-12-31T00:00:00","1884-12-31T00:00:00","1885-12-31T00:00:00","1886-12-31T00:00:00","1887-12-31T00:00:00","1888-12-31T00:00:00","1889-12-31T00:00:00","1890-12-31T00:00:00","1891-12-31T00:00:00","1892-12-31T00:00:00","1893-12-31T00:00:00","1894-12-31T00:00:00","1895-12-31T00:00:00","1896-12-31T00:00:00","1897-12-31T00:00:00","1898-12-31T00:00:00","1899-12-31T00:00:00","1900-12-31T00:00:00","1901-12-31T00:00:00","1902-12-31T00:00:00","1903-12-31T00:00:00","1904-12-31T00:00:00","1905-12-31T00:00:00","1906-12-31T00:00:00","1907-12-31T00:00:00","1908-12-31T00:00:00","1909-12-31T00:00:00","1910-12-31T00:00:00","1911-12-31T00:00:00","1912-12-31T00:00:00","1913-12-31T00:00:00","1914-12-31T00:00:00","1915-12-31T00:00:00","1916-12-31T00:00:00","1917-12-31T00:00:00","1918-12-31T00:00:00","1919-12-31T00:00:00","1920-12-31T00:00:00","1921-12-31T00:00:00","1922-12-31T00:00:00","1923-12-31T00:00:00","1924-12-31T00:00:00","1925-12-31T00:00:00","1926-12-31T00:00:00","1927-12-31T00:00:00","1928-12-31T00:00:00","1929-12-31T00:00:00","1930-12-31T00:00:00","1931-12-31T00:00:00","1932-12-31T00:00:00","1933-12-31T00:00:00","1934-12-31T00:00:00","1935-12-31T00:00:00","1936-12-31T00:00:00","1937-12-31T00:00:00","1938-12-31T00:00:00","1939-12-31T00:00:00","1940-12-31T00:00:00","1941-12-31T00:00:00","1942-12-31T00:00:00","1943-12-31T00:00:00","1944-12-31T00:00:00","1945-12-31T00:00:00","1946-12-31T00:00:00","1947-12-31T00:00:00","1948-12-31T00:00:00","1949-12-31T00:00:00","1950-12-31T00:00:00","1951-12-31T00:00:00","1952-12-31T00:00:00","1953-12-31T00:00:00","1954-12-31T00:00:00","1955-12-31T00:00:00","1956-12-31T00:00:00","1957-12-31T00:00:00","1958-12-31T00:00:00","1959-12-31T00:00:00","1960-12-31T00:00:00","1961-12-31T00:00:00","1962-12-31T00:00:00","1963-12-31T00:00:00","1964-12-31T00:00:00","1965-12-31T00:00:00","1966-12-31T00:00:00","1967-12-31T00:00:00","1968-12-31T00:00:00","1969-12-31T00:00:00","1970-12-31T00:00:00","1971-12-31T00:00:00","1972-12-31T00:00:00","1973-12-31T00:00:00","1974-12-31T00:00:00","1975-12-31T00:00:00","1976-12-31T00:00:00","1977-12-31T00:00:00","1978-12-31T00:00:00","1979-12-31T00:00:00","1980-12-31T00:00:00","1981-12-31T00:00:00","1982-12-31T00:00:00","1983-12-31T00:00:00","1984-12-31T00:00:00","1985-12-31T00:00:00","1986-12-31T00:00:00","1987-12-31T00:00:00","1988-12-31T00:00:00","1989-12-31T00:00:00","1990-12-31T00:00:00","1991-12-31T00:00:00","1992-12-31T00:00:00","1993-12-31T00:00:00","1994-12-31T00:00:00","1995-12-31T00:00:00","1996-12-31T00:00:00","1997-12-31T00:00:00","1998-12-31T00:00:00","1999-12-31T00:00:00","2000-12-31T00:00:00","2001-12-31T00:00:00","2002-12-31T00:00:00","2003-12-31T00:00:00","2004-12-31T00:00:00","2005-12-31T00:00:00","2006-12-31T00:00:00","2007-12-31T00:00:00","2008-12-31T00:00:00","2009-12-31T00:00:00","2010-12-31T00:00:00","2011-12-31T00:00:00","2012-12-31T00:00:00","2013-12-31T00:00:00","2014-12-31T00:00:00","2015-12-31T00:00:00","2016-12-31T00:00:00","2017-12-31T00:00:00"],"y":[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,2,1,1,0,3,5,4,19,29,16,7,36,28,27,33,39,42,8,1],"type":"scatter","xaxis":"x","yaxis":"y"},{"line":{"color":"#F9DEC9","width":2},"name":"Topic 4","opacity":0.7,"x":["1877-12-31T00:00:00","1878-12-31T00:00:00","1879-12-31T00:00:00","1880-12-31T00:00:00","1881-12-31T00:00:00","1882-12-31T00:00:00","1883-12-31T00:00:00","1884-12-31T00:00:00","1885-12-31T00:00:00","1886-12-31T00:00:00","1887-12-31T00:00:00","1888-12-31T00:00:00","1889-12-31T00:00:00","1890-12-31T00:00:00","1891-12-31T00:00:00","1892-12-31T00:00:00","1893-12-31T00:00:00","1894-12-31T00:00:00","1895-12-31T00:00:00","1896-12-31T00:00:00","1897-12-31T00:00:00","1898-12-31T00:00:00","1899-12-31T00:00:00","1900-12-31T00:00:00","1901-12-31T00:00:00","1902-12-31T00:00:00","1903-12-31T00:00:00","1904-12-31T00:00:00","1905-12-31T00:00:00","1906-12-31T00:00:00","1907-12-31T00:00:00","1908-12-31T00:00:00","1909-12-31T00:00:00","1910-12-31T00:00:00","1911-12-31T00:00:00","1912-12-31T00:00:00","1913-12-31T00:00:00","1914-12-31T00:00:00","1915-12-31T00:00:00","1916-12-31T00:00:00","1917-12-31T00:00:00","1918-12-31T00:00:00","1919-12-31T00:00:00","1920-12-31T00:00:00","1921-12-31T00:00:00","1922-12-31T00:00:00","1923-12-31T00:00:00","1924-12-31T00:00:00","1925-12-31T00:00:00","1926-12-31T00:00:00","1927-12-31T00:00:00","1928-12-31T00:00:00","1929-12-31T00:00:00","1930-12-31T00:00:00","1931-12-31T00:00:00","1932-12-31T00:00:00","1933-12-31T00:00:00","1934-12-31T00:00:00","1935-12-31T00:00:00","1936-12-31T00:00:00","1937-12-31T00:00:00","1938-12-31T00:00:00","1939-12-31T00:00:00","1940-12-31T00:00:00","1941-12-31T00:00:00","1942-12-31T00:00:00","1943-12-31T00:00:00","1944-12-31T00:00:00","1945-12-31T00:00:00","1946-12-31T00:00:00","1947-12-31T00:00:00","1948-12-31T00:00:00","1949-12-31T00:00:00","1950-12-31T00:00:00","1951-12-31T00:00:00","1952-12-31T00:00:00","1953-12-31T00:00:00","1954-12-31T00:00:00","1955-12-31T00:00:00","1956-12-31T00:00:00","1957-12-31T00:00:00","1958-12-31T00:00:00","1959-12-31T00:00:00","1960-12-31T00:00:00","1961-12-31T00:00:00","1962-12-31T00:00:00","1963-12-31T00:00:00","1964-12-31T00:00:00","1965-12-31T00:00:00","1966-12-31T00:00:00","1967-12-31T00:00:00","1968-12-31T00:00:00","1969-12-31T00:00:00","1970-12-31T00:00:00","1971-12-31T00:00:00","1972-12-31T00:00:00","1973-12-31T00:00:00","1974-12-31T00:00:00","1975-12-31T00:00:00","1976-12-31T00:00:00","1977-12-31T00:00:00","1978-12-31T00:00:00","1979-12-31T00:00:00","1980-12-31T00:00:00","1981-12-31T00:00:00","1982-12-31T00:00:00","1983-12-31T00:00:00","1984-12-31T00:00:00","1985-12-31T00:00:00","1986-12-31T00:00:00","1987-12-31T00:00:00","1988-12-31T00:00:00","1989-12-31T00:00:00","1990-12-31T00:00:00","1991-12-31T00:00:00","1992-12-31T00:00:00","1993-12-31T00:00:00","1994-12-31T00:00:00","1995-12-31T00:00:00","1996-12-31T00:00:00","1997-12-31T00:00:00","1998-12-31T00:00:00","1999-12-31T00:00:00","2000-12-31T00:00:00","2001-12-31T00:00:00","2002-12-31T00:00:00","2003-12-31T00:00:00","2004-12-31T00:00:00","2005-12-31T00:00:00","2006-12-31T00:00:00","2007-12-31T00:00:00","2008-12-31T00:00:00","2009-12-31T00:00:00","2010-12-31T00:00:00","2011-12-31T00:00:00","2012-12-31T00:00:00","2013-12-31T00:00:00","2014-12-31T00:00:00","2015-12-31T00:00:00","2016-12-31T00:00:00","2017-12-31T00:00:00"],"y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,12,11,21,31,22,31,32,46,33,43,52,39,11,1,0],"type":"scatter","xaxis":"x","yaxis":"y"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"rangeslider":{"visible":true},"title":{"text":"Year"},"showgrid":true,"gridwidth":0.3,"gridcolor":"LightGrey"},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"# Wines reviewed"},"showgrid":true,"gridwidth":0.3,"gridcolor":"LightGrey"},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"# Wines reviewed"},"showgrid":true,"gridwidth":0.3,"gridcolor":"LightGrey"},"showlegend":true,"width":1200,"height":600,"annotations":[{"arrowhead":7,"ax":100,"ay":-350,"opacity":0.5,"text":"2014 has the highest number of reviewed wines","x":"2014-12-31","y":0}],"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('ee02ed71-0363-4d72-9b09-48f9304368f0');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
# Save figure
import chart_studio
username = 'renswilderom' # your username
api_key = 'JQ0okV7QsrTwEJJfp7dp' # your api key - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
import chart_studio.plotly as py
py.plot(fig, filename = 'Figuur 2. Verschillende Covid-19 nieuws topics (week data).', auto_open=True)
import plotly.io as pio
pio.write_html(fig, file='Figuur 2. Verschillende Covid-19 nieuws topics (week data).', auto_open=False)
```


```python
# # width 900; height 400; length annotation 230 for pdf and eps
# # width 1200; height 600; length annotation 250 for interactive
fig.write_image("P:/My documents/Corona project/Figuur_2.PDF")
fig.write_image("P:/My documents/Corona project/Figuur_2.PNG")
```

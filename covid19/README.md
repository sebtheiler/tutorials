# COVID-19 - An Analysis Through Data
#### Learn to create choropleth maps and scatter plots in Python using Plotly

Available here: COMING SOON

Explanation of files:
* `corona-choropleth.py`: Create the choropleth map of COVID-19 cases, deaths, daily cases, etc. Edit the `metric` and `multiplier` variables to make your own visualization.
* `correlations.py`: Find correlations between COVID-19 statistics and population statistics. Also makes a scatter plot with `metric1` (COVID-19 statistic) as the x-axis and `metric2` (population statistic) as the y-axis.
* `top_and_bottom.py`: Print the top and bottom countries in each category (death rate, recovery rate, cases per capita, etc.).

Please download the "Population, surface area and density" data from [UNSD](https://data.un.org/), or directly use [this](https://data.un.org/_Docs/SYB/CSV/SYB62_1_201907_Population,%20Surface%20Area%20and%20Density.csv) link and put the downloaded data in the `data/` folder.  Afterwards, make the changes mentioned in the article (delete the first line, modify the second line) and make sure the file is named `SYB62_1_201907_Population, Surface Area and Density.csv`.

## Requirements
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* pycountry==19.8.18
* pandas==1.0.4
* plotly==4.8.1
* scipy==1.4.1

The code will likely still work with other versions of these packages, but that hasn't been tested.

### Conda Instructions
```
conda create -n covid19 -y python=3.7.7
conda activate covid19
conda install -y -c derickl pycountry
conda install -y pandas==1.0.4
conda install -y plotly==4.8.1
conda install -y scipy==1.4.1
```


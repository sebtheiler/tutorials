import numpy as np
import pandas as pd
import plotly.express as px
import pycountry
from plotly.offline import plot

metric = 'Recovery Rate'
multiplier = 1 # Use this to manually lower the color scale (e.g. for Death Rate)

# Load CSVs
covid_df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')
population_df = pd.read_csv('data/SYB62_1_201907_Population, Surface Area and Density.csv', encoding="ISO-8859-1")

# These are extra country name -> 3 letter ISO code conversions
# for countries with multiple or unofficial names.
extra_conversion = {
    'US': 'USA',
    'Russia': 'RUS',
    'Taiwan*': 'TWN',
    'Korea, South': 'KOR',
    'Kosovo': 'CS-KM',
    'Syria': 'SYR',
    'Venezuela': 'VEN',
    "Cote d'Ivoire": 'CIV',
    'Iran': 'IRN',
    'Laos': 'LAO',
    'Moldova': 'MDA',
    'Tanzania': 'TZA',
    'Vietnam': 'VNM',
    'West Bank and Gaza': 'PSE',
    'Congo (Kinshasa)': 'COD',
    'Congo (Brazzaville)': 'COG',
    'Bolivia': 'BOL',
    'Brunei': 'BRN',
    'Burma': 'MMR',
    'Holy See': 'VAT',
    'Saint Vincent & Grenadines': 'VCT'
}

# This function takes a country and uses a combination of
# `pycountry` and the above dict to get it's 3 letter code
def get_country_id(country):
    try:
        if isinstance(country, str):
            # If we pass the raw country name...
            return pycountry.countries.get(name=country).alpha_3
        elif isinstance(country, int):
            # If we pass the country ID...

            # Pycountry requires the numeric code to be 3
            # digits long, so we pad it with zeros (str.zfill)
            return pycountry.countries.get(numeric=str(country).zfill(3)).alpha_3
    except AttributeError:
        if country in extra_conversion.keys():
            return extra_conversion[country]
        else:
            # print('No id for', country)
            return 'Unknown'

# Find IDs for each country in both dataframes
population_df['id'] = population_df['Number'].map(get_country_id)
covid_df['id'] = covid_df['Country'].map(get_country_id)

# This function searches the U.N. database for the population
# information of a specified country in a specific year.  It
# has options for passing in a raw 3-letter ID (instead of using
# a raw country name and coverting it to an ID) and for returning
# a function that only takes a country as a parameter (we use the
# latter case for mapping inside a Pandas Series).
def get_country_info(country=None, information='Population mid-year estimates (millions)', year=2019, pass_raw_id=False, return_function=False):
    """
    Possible values for `information`:
    'Population mid-year estimates (millions)'
    'Population mid-year estimates for males (millions)'
    'Population mid-year estimates for females (millions)'
    'Sex ratio (males per 100 females)'
    'Population aged 0 to 14 years old (percentage)'
    'Population aged 60+ years old (percentage)'
    'Population density'
    """
    def func(country):
        country_id = country if pass_raw_id else get_country_id(country)
        try:
            return float(population_df[(population_df['id'] == country_id) &
                                (population_df['Series'] == information) &
                                (population_df['Year'] == year)]['Value'].values[0])
        except IndexError:
            # print('No data for', country, information, year)
            return np.NaN
    return func if return_function else func(country)


if __name__ == "__main__":
    # Calculate information
    covid_df['Population'] = covid_df['id'].map(get_country_info(information='Population mid-year estimates (millions)', pass_raw_id=True, return_function=True)).mul(1000000)
    covid_df['Cases by Population'] = covid_df['Confirmed'].div(covid_df['Population'])
    covid_df['Recovery Rate'] = covid_df['Recovered'].div(covid_df['Confirmed'])
    covid_df['Death Rate'] = covid_df['Deaths'].div(covid_df['Confirmed'])

    # Calculate daily statistics
    previous_cases = {}
    previous_deaths = {}
    daily_cases = []
    daily_deaths = []
    for i, row in covid_df.iterrows():
        previous = previous_cases[row.Country] if row.Country in previous_cases.keys() else 0
        daily_cases.append(row.Confirmed - previous)
        previous_cases[row.Country] = row.Confirmed

        previous = previous_deaths[row.Country] if row.Country in previous_deaths.keys() else 0
        daily_deaths.append(row.Deaths - previous)
        previous_deaths[row.Country] = row.Deaths

    covid_df['Cases per Day'] = daily_cases
    covid_df['Cases per Day by Population'] = covid_df['Cases per Day'].div(covid_df['Population'])
    covid_df['Deaths per Day'] = daily_deaths
    covid_df['Deaths per Day by Population'] = covid_df['Deaths per Day'].div(covid_df['Population'])

    # Visualize!
    fig = px.choropleth(covid_df,
                        locations="id",
                        color=metric,
                        hover_name="Country",
                        hover_data=['Population', 'Confirmed', 'Recovered', 'Deaths', 'Cases per Day'],
                        animation_frame='Date',
                        range_color=(0, covid_df[metric].max()*multiplier),
                        color_continuous_scale=px.colors.sequential.Viridis)
    fig.show()

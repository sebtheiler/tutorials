import numpy as np
import pandas as pd
import plotly.express as px
import pycountry
from plotly.offline import plot
from scipy.stats import kendalltau, pearsonr, spearmanr

# We'll import the CSVs and functions from the previous file
from corona_choropleth import get_country_info, get_country_id, covid_df, population_df, extra_conversion

metric1 = 'Recovery Rates'
metric2 = 'Percent of Population >60'

# Get each country ID that appears in both the population dataframe and the COVID-19 dataframe
countries = list(set(covid_df['id'].unique()).intersection(set(population_df['id'].unique())))

# Remove problematic countries
to_remove = ['Unknown', 'VAT', 'LIE', 'AND', 'DMA', # Incomplete pop. data
             'SMR', 'MCO', 'KNA', # Incomplete pop. data
             'QAT'] # Huge outlier

for country in to_remove:
    countries.remove(country)

print(countries)

# Gather population information into list form
country_populations = [get_country_info(country, information='Population mid-year estimates (millions)', pass_raw_id=True) * 1000000 for country in countries]
country_sex_ratios = [get_country_info(country, information='Sex ratio (males per 100 females)', pass_raw_id=True) for country in countries]
country_young_percent = [get_country_info(country, information='Population aged 0 to 14 years old (percentage)', pass_raw_id=True) for country in countries]
country_senior_percent = [get_country_info(country, information='Population aged 60+ years old (percentage)', pass_raw_id=True) for country in countries]
country_pop_density = [get_country_info(country, information='Population density', pass_raw_id=True) for country in countries]

# Calculate total # of cases/deaths and their normalized equivalents
country_covid_cases = [(covid_df[covid_df['id'] == country].iloc[-1])['Confirmed'] for country in countries]
country_covid_cases_pop = [country_covid_cases[i] / country_populations[i] for i in range(len(country_covid_cases))]
country_covid_deaths = [(covid_df[covid_df['id'] == country].iloc[-1])['Deaths'] for country in countries]
country_covid_deaths_pop = [country_covid_deaths[i] / country_populations[i] for i in range(len(country_covid_deaths))]
country_recoveries = [(covid_df[covid_df['id'] == country].iloc[-1])['Recovered'] for country in countries]
country_recovery_rates = [country_recoveries[i] / country_covid_cases[i] for i in range(len(country_recoveries))]
country_death_rates = [country_covid_deaths[i] / country_covid_cases[i] for i in range(len(country_covid_deaths))]

# Put stats into dictionaries
pop_stats = {
    'Population': country_populations,
    'Sex Ratio': country_sex_ratios,
    'Percent of Population <14': country_young_percent,
    'Percent of Population >60': country_senior_percent,
    'Population Density': country_pop_density
}

cov_stats = {
    'COVID-19 Cases': country_covid_cases,
    'COVID-19 Cases Normalized by Population': country_covid_cases_pop,
    'COVID-19 Deaths': country_covid_deaths,
    'COVID-19 Deaths Normalized by Population': country_covid_deaths_pop,
    'Recoveries': country_recoveries,
    'Recovery Rates': country_recovery_rates,
    'Death Rates': country_death_rates
}

if __name__ == "__main__":
    # Calculate correlations
    print('---')
    for cov_stat_name, cov_stat in zip(cov_stats.keys(), cov_stats.values()):
        for pop_stat_name, pop_stat in zip(pop_stats.keys(), pop_stats.values()):
            print(cov_stat_name, 'and', pop_stat_name)
            print(pearsonr(cov_stat, pop_stat)[0],
                spearmanr(cov_stat, pop_stat).correlation,
                kendalltau(cov_stat, pop_stat).correlation,
                '\n')
        print('---')

    # Plot!
    fig = px.scatter(x=cov_stats[metric1],
                     y=pop_stats[metric2],
                     hover_name=countries,
                     hover_data={**cov_stats, **pop_stats})
    fig.update_layout(xaxis_title=metric1, yaxis_title=metric2)
    fig.show()

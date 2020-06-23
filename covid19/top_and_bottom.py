import numpy as np
import pandas as pd
import plotly.express as px
import pycountry

# We'll import the CSVs and functions from the previous file
from corona_choropleth import (covid_df, extra_conversion, get_country_id,
                               get_country_info, population_df)
from correlations import (countries, country_covid_cases,
                          country_covid_cases_pop, country_covid_deaths,
                          country_covid_deaths_pop, country_death_rates,
                          country_recovery_rates, covid_df)

country_data = []
for country, cases, cases_pop, deaths, deaths_pop, recovery_rate, death_rate in \
            zip(countries, country_covid_cases,
                country_covid_cases_pop, country_covid_deaths,
                country_covid_deaths_pop,
                country_recovery_rates, country_death_rates):
    if cases / cases_pop > 2500000 and cases > 1000:
        # If population is greater than 2,500,000 and cases > 1,000
        country_data.append((country, cases, cases_pop, deaths, deaths_pop, recovery_rate, death_rate))


def print_country_names(data_index, num=5, reverse=False):
    country_data.sort(key=lambda data: data[data_index], reverse=reverse)
    return [(pycountry.countries.get(alpha_3=data[0]).name, data[data_index]) for data in country_data[:num]]


print('\nHighest Cases:')
print(print_country_names(1, reverse=True))

print('\nLowest Cases by Population:')
print(print_country_names(2))

print('\nHighest Cases by Population:')
print(print_country_names(2, reverse=True))

print('\nHighest Deaths:')
print(print_country_names(3, reverse=True))

print('\nLowest Deaths by Population:')
print(print_country_names(4))

print('\nHighest Deaths by Population:')
print(print_country_names(4, reverse=True))

print('\nLowest Recovery Rate:')
print(print_country_names(5, num=8)[3:])

print('\nHighest Recovery Rate:')
print(print_country_names(5, reverse=True))

print('\nLowest Death Rate:')
print(print_country_names(6))

print('\nHighest Death Rate')
print(print_country_names(6, reverse=True))

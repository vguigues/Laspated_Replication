import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
from subprocess import PIPE, Popen
import os
from shutil import copyfile
import time
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import numpy as np

import laspated as spated
from shapely.geometry import Polygon, MultiPolygon, Point
import sys

def process_land_use():
    land_use = gpd.read_file(r"regressores/uso_do_solo/")
    land_types = [
        "Afloramentos rochosos e depósitos sedimentares",
        "Áreas agrícolas",
        "Áreas de comércio e serviços",
        "Áreas de educação e saúde",
        "Áreas de exploração mineral",
        "Áreas de lazer",
        "Áreas de transporte",
        "Áreas industriais",
        "Áreas institucionais e de infraestrutura pública",
        "Áreas não edificadas",
        "Áreas residenciais",
        "Áreas sujeitas à inundação",
        "Cobertura arbórea e arbustiva",
        "Cobertura gramíneo lenhosa",
        "Corpos hídricos",
        "Favela",
    ]

    groups = {
        # Urbanized Areas. Public Activities
        0: [
            "Áreas de comércio e serviços",
            "Áreas de educação e saúde",
            "Áreas de lazer",
            "Áreas institucionais e de infraestrutura pública",
            "Áreas de transporte",
            "Áreas industriais",
        ],
        # Non Urbanized Areas. Non-Populational
        1: [
            "Afloramentos rochosos e depósitos sedimentares",
            "Áreas agrícolas",
            "Áreas de exploração mineral",
            "Áreas sujeitas à inundação",
            "Cobertura arbórea e arbustiva",
            "Cobertura gramíneo lenhosa",
            "Corpos hídricos",
        ],
        # Urbanized Areas. Residential
        2: ["Áreas residenciais", "Favela"],
    }

    for i in range(3):
        land_use[f"subgroup_{i}"] = 0
        for j, row in land_use.iterrows():
            if row["usoagregad"] in groups[i]:
                land_use.at[j, f"subgroup_{i}"] = 1

    land_use = land_use[["subgroup_0", "subgroup_1", "subgroup_2", "geometry"]].copy()
    return land_use



app = spated.DataAggregator(crs="epsg:4326")
events = pd.read_csv(r"eventos_desagregados.csv", encoding="ISO-8859-1", sep=",")
print(events.columns)
# show count by column prioridade
events = events[events["Prioridade"] <= 2]
# Swap rows: prioridade 1 becomes 2, 2 becomes 0 and 0 becomes 1
events["Prioridade"] = events["Prioridade"].map({0: 1, 1: 2, 2: 0})

app.add_events_data(
    events,
    datetime_col="data_idx",
    lat_col="latitude",
    lon_col="longitude",
    feature_cols=["Prioridade"],
    datetime_format="%m/%d/%y %H:%M",
)

max_borders = gpd.read_file(
    r"rj/rj.shp"
)  # Load the geometry of region of interest
app.add_max_borders(max_borders)

# Time discretizations
num_periods, period_length = (30, 60 * 24)
app.add_time_discretization("m", num_periods, period_length, column_name="hhs")
app.add_time_discretization("D", 1, 7, column_name="dow")

app.add_geo_discretization(
        discr_type="R", rect_discr_param_x=10, rect_discr_param_y=10
    )

land_use = process_land_use()
app.add_geo_variable(land_use, type_geo_variable="area")

# Geo Features
population = gpd.read_file(r"populacao/")
population = population[["population", "geometry"]].copy()
app.add_geo_variable(population)

# scaling the population 
app.geo_discretization["population"] /= 10**4
R = int(np.max(app.events_data["gdiscr"]) + 1)

# print app.events_data without the geometry column
print(app.events_data.columns)
# force the print of 300 rows
print(app.events_data[['ts', 'Prioridade', 'hhs', 'dow', 'gdiscr']].sample(300).to_string(index=False))
# print distinct gdiscr values
print(app.events_data['gdiscr'].count())
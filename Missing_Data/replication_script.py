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

def read_model1(path, C, R, Total_T):
    arq = open(path, "r")
    lam_emp = np.zeros((C, R, Total_T))
    lam_est = np.zeros((C, R, Total_T))
    for line in arq.readlines():
        tokens = line.split()
        c = int(tokens[0])
        r = int(tokens[1])
        t = int(tokens[2])
        val_emp = float(tokens[3])
        val_est = float(tokens[4])
        lam_emp[c, r, t] = val_emp
        lam_est[c, r, t] = val_est
    arq.close()

    return lam_emp, lam_est

def read_p_model_1(path, C, Total_T):
    arq = open(path, "r")
    p = float(arq.readline())
    p_ct = np.zeros((C, Total_T))
    for line in arq.readlines():
        tokens = line.split()
        c = int(tokens[0])
        t = int(tokens[1])
        p_val = float(tokens[2])
        p_ct[c, t] = p_val
    arq.close()

    return p, p_ct

def read_conf_intervals_model1(path,C,R,Total_T):
    arq = open(path, "r")
    conf_intervals = np.zeros((C, R, Total_T, 2))
    lam = np.zeros((C, R, Total_T))
    for line in arq.readlines():
        tokens = line.split()
        c = int(tokens[0])
        r = int(tokens[1])
        t = int(tokens[2])
        ci_lower = float(tokens[3])
        lam[c,r,t] = float(tokens[4])
        ci_upper = float(tokens[5])
        conf_intervals[c, r, t, 0] = ci_lower
        conf_intervals[c, r, t, 1] = ci_upper
    arq.close()

    return conf_intervals, lam

def read_model2(path,C, R, Total_T):
    lam = np.zeros((C, R, Total_T))
    arq = open(path, "r")
    for line in arq.readlines():
        tokens = line.split()
        c = int(tokens[0])
        r = int(tokens[1])
        t = int(tokens[2])
        val = float(tokens[3])
        lam[c, r, t] = val
    arq.close()
    return lam


def read_model3(path, C, R, Total_T):
    arq = open(path, "r")
    lam = np.zeros((C, R, Total_T))
    for line in arq.readlines():
        tokens = line.split()
        c = int(tokens[0])
        r = int(tokens[1])
        t = int(tokens[2])
        val_m1 = float(tokens[3])
        val_m2 = float(tokens[4])
        lam[c, r, t] = max(val_m1, val_m2)
    arq.close()
    return lam

def read_model4(path):
    arq = open(path, "r")
    lam = np.zeros((3,100,7*48))
    for line in arq.readlines():
        tokens = line.split()
        c = int(tokens[0])
        r = int(tokens[1])
        t = int(tokens[2])
        val = float(tokens[3])
        lam[c, r, t] = val
    arq.close()
    return lam

def write_empirical_estimation(arrivals_path, output_path, T, G, R, P, max_obs):
    nb_arrivals = np.zeros((T, G, R, P), dtype=float)
    with open(arrivals_path, "r") as arq:
        for line in arq:
            if line.strip() == "END":
                break
            tokens = line.split()
            t = int(tokens[0])
            g = int(tokens[1])
            r = int(tokens[2])
            p = int(tokens[3])
            val = int(tokens[5])
            nb_arrivals[t, g, r, p] += val

    with open(output_path, "w") as arq:
        arq.write(f"{T} {G} {R} {P} {max_obs}\n")
        denom = max_obs * 0.5
        for t in range(T):
            for g in range(G):
                for r in range(R):
                    for p in range(P):
                        emp = nb_arrivals[t, g, r, p] / denom
                        arq.write(f"{t} {g} {r} {p} {emp}\n")

class Stats(object):
    def __init__(self, lam):
        self.mean_total = np.zeros((T))
        self.q05_total = np.zeros((T))
        self.q95_total = np.zeros((T))
        self.mean_p0 = np.zeros((T))
        self.q05_p0 = np.zeros((T))
        self.q95_p0 = np.zeros((T))
        self.mean_p1 = np.zeros((T))
        self.q05_p1 = np.zeros((T))
        self.q95_p1 = np.zeros((T))
        self.mean_p2 = np.zeros((T))
        self.q05_p2 = np.zeros((T))
        self.q95_p2 = np.zeros((T))
        for t in range(T):
            self.mean_total[t] = np.sum(lam[:, :, t])
            self.q05_total[t] = stats.poisson.ppf(0.05, self.mean_total[t])
            self.q95_total[t] = stats.poisson.ppf(0.95, self.mean_total[t])
            self.mean_p0[t] = np.sum(lam[0, :, t])
            self.q05_p0[t] = stats.poisson.ppf(0.05, self.mean_p0[t])
            self.q95_p0[t] = stats.poisson.ppf(0.95, self.mean_p0[t])
            self.mean_p1[t] = np.sum(lam[1, :, t])
            self.q05_p1[t] = stats.poisson.ppf(0.05, self.mean_p1[t])
            self.q95_p1[t] = stats.poisson.ppf(0.95, self.mean_p1[t])
            self.mean_p2[t] = np.sum(lam[2, :, t])
            self.q05_p2[t] = stats.poisson.ppf(0.05, self.mean_p2[t])
            self.q95_p2[t] = stats.poisson.ppf(0.95, self.mean_p2[t])

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

def write_samples_file(geo_discretization, output_path, num_samples=100, rng=None, max_attempts=10000):
    if rng is None:
        rng = np.random.default_rng()

    geo_disc = geo_discretization.copy()
    if "id" in geo_disc.columns:
        geo_disc = geo_disc.sort_values("id")

    with open(output_path, "w") as arq:
        for idx, row in geo_disc.iterrows():
            zone_id = row["id"] if "id" in geo_disc.columns else idx
            geometry = row["geometry"]
            minx, miny, maxx, maxy = geometry.bounds
            samples_written = 0
            attempts = 0
            while samples_written < num_samples and attempts < max_attempts:
                lon = rng.uniform(minx, maxx)
                lat = rng.uniform(miny, maxy)
                if geometry.contains(Point(lon, lat)):
                    arq.write(f"{zone_id} {samples_written} {lon} {lat}\n")
                    samples_written += 1
                attempts += 1
            if samples_written < num_samples:
                raise ValueError(f"Could not sample {num_samples} points for zone {zone_id}")

def generate_discretization(base_folder, time_tuple):
    app = spated.DataAggregator(crs="epsg:4326")
    events = pd.read_csv(r"sorted_events.csv", encoding="ISO-8859-1", sep=",")
    app.add_events_data(
        events,
        datetime_col="data_hora",
        lat_col="lat",
        lon_col="long",
        feature_cols=["prioridade"],
        datetime_format="%m/%d/%y %H:%M:%S",
    )
    
    max_borders = gpd.read_file(
        r"rj/rj.shp"
    )  # Load the geometry of region of interest
    app.add_max_borders(max_borders)
    
    # Time discretizations
    num_periods, period_length = time_tuple
    app.add_time_discretization("m", num_periods, period_length, column_name="hhs")
    app.add_time_discretization("D", 1, 7, column_name="dow")
        
    if base_folder == "Rect10x10" or base_folder == "Rect10x10_1h":
        app.add_geo_discretization(
            discr_type="R", rect_discr_param_x=10, rect_discr_param_y=10
        )
    elif base_folder == "Rect20x20":
        app.add_geo_discretization(
            discr_type="R", rect_discr_param_x=20, rect_discr_param_y=20
        )
    elif base_folder == "Hex7" or base_folder == "Hex7_1h":
        app.add_geo_discretization(discr_type="H", hex_discr_param=7)
    elif base_folder == "District" or base_folder == "District_1h":
        custom_map = gpd.read_file(r'rio_de_janeiro_neighborhoods/rio_neighborhoods.shp')
        app.add_geo_discretization(discr_type="C", custom_data=custom_map)
    else:
        raise ValueError(f"Invalid base_folder {base_folder}")
    
    land_use = process_land_use()
    app.add_geo_variable(land_use, type_geo_variable="area")
    
    # Geo Features
    population = gpd.read_file(r"populacao/")
    population = population[["population", "geometry"]].copy()
    app.add_geo_variable(population)
    
    # scaling the population 
    app.geo_discretization["population"] /= 10**4
    R = int(np.max(app.events_data["gdiscr"]) + 1)
    
    
    app.write_arrivals(f"{base_folder}/arrivals.dat")
    app.write_regions(f"{base_folder}/neighbors.dat")
    app.write_info(obs_index_column="dow", path=f"{base_folder}/info.dat")
    write_samples_file(app.geo_discretization, f"{base_folder}/samples.dat")
    T = time_tuple[1] // time_tuple[0]
    write_empirical_estimation(
        f"{base_folder}/arrivals.dat",
        f"{base_folder}/empirical_estimation.dat",
        T,
        7,
        R,
        3,
        max_obs,
    )
    
    num_regions = int(np.max(app.events_data["gdiscr"]) + 1)
    print(f"Wrote discretization files for disc {base_folder}. {num_regions} regions")
    
    return app, num_regions
    


tic = time.perf_counter()

nb_obs = [104, 104, 104, 104, 105, 105, 105]
max_obs = np.max(nb_obs)




# discs = [("Rect10x10", (30, 60 * 24)), ("Rect10x10_1h", (60, 60 * 24)), ("District", (30, 60 * 24)), ("District_1h", (60, 60 * 24))]    
# discs = [("Rect10x10_1h", (60, 60 * 24))] 
# discs = [("District_1h", (60, 60 * 24))]
# discs = [("Rect10x10", (30, 60 * 24))]
discs = [("Hex7", (30, 60 * 24))]  


for disc in discs:
    base_folder, time_tuple = disc
    print(f"Generating discretization for {base_folder}...")
    app, R = generate_discretization(base_folder, time_tuple)
    T = time_tuple[1] // time_tuple[0]
    D = 7
    Total_T = T * D
    C = 3
    
    try:
        print(f"Running C++ experiments for {base_folder}...")
        args = ["./missing", "-f", "test.cfg", "--model=all",
            f"--info_file={base_folder}/info.dat",
            f"--neighbors_file={base_folder}/neighbors.dat",
            f"--arrivals_file={base_folder}/arrivals.dat",
            f"--missing_file={base_folder}/missing_calls.dat"]
        for arg in args:
            print(arg, end=" ")
        print()
        with Popen(
            args,
            bufsize=1,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        ) as p:
            for line in p.stdout:
                print(line, end="",flush=True)

            for line in p.stderr:
                print(line, end="",flush=True)
        print(f"Finished running C++ experiments for {base_folder}.")
    except:
        print(f"Error with C++ experiments")
        exit(1)
        
    print(f"Plotting results for {base_folder}, R = {R}, T = {T}...")
    lam_no_miss_1, lam_miss_1 = read_model1(f"results/lambda_model1_R{R}_T{T}.txt",C, R, Total_T)
    stats_no_miss_model1 = Stats(lam_no_miss_1)
    stats_miss_model1 = Stats(lam_miss_1)
    p, p_ct = read_p_model_1(f"results/p_model1_R{R}_T{T}.txt",C, Total_T)
    conf_intervals_model1, lam_model1 = read_conf_intervals_model1(f"results/conf_intervals_model1_R{R}_T{T}.txt",C, R, Total_T)
    
    # Plot p_ct for each c
    time_index = [t for t in range(Total_T)]
    plt.figure(figsize=(10, 6))
    plt.plot(time_index, p_ct[0, :], label="High priority calls", color="black", linestyle="solid")
    plt.plot(time_index, p_ct[1, :], label="Intermediate priority calls", color="red", linestyle="--")
    plt.plot(time_index, p_ct[2, :], label="Low priority calls", color="blue", linestyle="dashdot")
    # horizontal line for p
    plt.axhline(y=p, color="gray", linestyle="dotted", label="p")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Probability of not reporting the location")
    plt.savefig(f"{base_folder}/plot_p_ct_model1_R{R}_T{T}.pdf", bbox_inches="tight")
    plt.close()
    
    # plot confidence intervals for each c
    for c in range(C):
        plt.figure(figsize=(10, 6))
        plt.plot(time_index, lam_model1[c, :, :].mean(axis=0), label="Lambda", color="black", linestyle="solid")
        plt.fill_between(time_index,
                         conf_intervals_model1[c, :, :, 0].mean(axis=0),
                         conf_intervals_model1[c, :, :, 1].mean(axis=0),
                         color="gray", alpha=0.5, label="Confidence Interval")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Intensity")
        plt.savefig(f"{base_folder}/conf_intervals_model1_c{c}_R{R}_T{T}.pdf", bbox_inches="tight")
        plt.close()
        
    # Plot heatmaps for stats_no_miss_model1 and stats_miss_model1 given the app.geo_discretization
    geo_disc = app.geo_discretization
    geo_disc = geo_disc.copy()
    sum_r_no_miss = np.sum(lam_no_miss_1, axis=(0, 2))
    sum_r_miss = np.sum(lam_miss_1, axis=(0,2))
    # assign each index i of sum_r_no_miss and sum_r_miss to the corresponding geo_disc id
    geo_disc["mean_no_miss"] = 0
    geo_disc["mean_miss"] = 0
    for i in range(len(geo_disc)):
        geo_disc.at[i, "mean_no_miss"] = sum_r_no_miss[geo_disc.at[i, "id"]]
        geo_disc.at[i, "mean_miss"] = sum_r_miss[geo_disc.at[i, "id"]]
    
    vmin = np.nanmin([geo_disc["mean_no_miss"].min(), geo_disc["mean_miss"].min()])
    vmax = np.nanmax([geo_disc["mean_no_miss"].max(), geo_disc["mean_miss"].max()])
    # Save uncorrected heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    geo_disc.plot(
        column="mean_no_miss",
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.2,
        edgecolor="0.8",
        legend=True,
        ax=ax,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"results/heatmap_model1_no_miss_R{R}_T{T}.pdf", bbox_inches="tight")
    plt.close()

    # Save corrected heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    geo_disc.plot(
        column="mean_miss",
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.2,
        edgecolor="0.8",
        legend=True,
        ax=ax,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"results/heatmap_model1_miss_R{R}_T{T}.pdf", bbox_inches="tight")
    plt.close()
   
    # Plot mean_total, q.05_total and q.95_total for each model no_miss and miss
    plt.figure(figsize=(10, 6))
    plt.plot(time_index, stats_no_miss_model1.mean_total, label="Mean Corrected", 
             color="black", linestyle="solid")
    plt.plot(time_index, stats_no_miss_model1.q05_total, label="q0.05 Corrected",
             color="black", linestyle="--")
    plt.plot(time_index, stats_no_miss_model1.q95_total, label="q0.95 Corrected",
             color="black", linestyle="--")
    plt.plot(time_index, stats_miss_model1.mean_total, label="Mean Uncorrected", 
             color="red", linestyle="solid")
    plt.plot(time_index, stats_miss_model1.q05_total, label="q0.05 Uncorrected",
             color="red", linestyle="--")
    plt.plot(time_index, stats_miss_model1.q95_total, label="q0.95 Uncorrected",
             color="red", linestyle="--")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of calls")
    plt.savefig(f"{base_folder}/plot_total_model1_R{R}_T{T}.pdf", bbox_inches="tight")
    plt.close()
    
    # Plot mean_total histograms for no_miss and miss
    plt.figure(figsize=(10, 6))
    plt.hist(stats_no_miss_model1.mean_total, bins=20, alpha=0.5, label="Corrected", color="black")
    plt.hist(stats_miss_model1.mean_total, bins=20, alpha=0.5, label="Uncorrected", color="red")
    plt.legend()
    plt.xlabel("Number of calls")
    plt.ylabel("Probabilities")
    plt.savefig(f"{base_folder}/hist_total_model1_R{R}_T{T}.pdf", bbox_inches="tight")
    plt.close()
    
    
    test_weights = [0, 0.001, 0.005, 0.01, 0.03]
    styles =  ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1))]
    colors = ['black', 'red']
    
    # Plot mean_total of model 2 for each test weight
    plt.figure(figsize=(10, 6))
    for i,w in enumerate(test_weights):
        lam_model2 = read_model2(f"results/lambda_model2_w{w}_R{R}_T{T}.txt", C, R, Total_T)
        stats_model2 = Stats(lam_model2)
        plt.plot(time_index, stats_model2.mean_total, label=f"w = {w}", color=colors[i % 2], linestyle=styles[i])
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Lambda")
    plt.savefig(f"{base_folder}/plot_total_model2_R{R}_T{T}.pdf", bbox_inches="tight")
    plt.close()
    
    lam_model2_cv = read_model2(f"results/lambda_model2_cv_R{R}_T{T}.txt", C, R, Total_T)
    stats_model2_cv = Stats(lam_model2_cv)
    # plot no_missing and cross validation for model 2
    plt.figure(figsize=(10, 6))
    plt.plot(time_index, stats_no_miss_model1.mean_total, label="Empirical", color="black", linestyle="solid")
    plt.plot(time_index, stats_model2_cv.mean_total, label="Cross Validation", color="red", linestyle="dotted")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Intensities")
    plt.savefig(f"{base_folder}/plot_total_model2_cv_R{R}_T{T}.pdf", bbox_inches="tight")
    plt.close()
    
    
    
    
# Plot comparison of discretizations
lam_no_miss_rect, lam_miss_rect = read_model1(f"results/lambda_model1_R76_T48.txt", 76)
stats_no_miss_rect = Stats(lam_no_miss_rect)
stats_miss_rect = Stats(lam_miss_rect)

lam_no_miss_district, lam_miss_district = read_model1(f"results/lambda_model1_R160_T48.txt", 160)
stats_no_miss_district = Stats(lam_no_miss_district)
stats_miss_district = Stats(lam_miss_district)

test_weights = [0]
all_stats_rect = []
all_stats_district = []
for w in test_weights:
    lam_rect = read_model2(f"results/lambda_model2_w{w}_R76_T48.txt", 76)
    all_stats_rect.append(Stats(lam_rect))
    lam_district = read_model2(f"results/lambda_model2_w{w}_R160_T48.txt", 160)
    all_stats_district.append(Stats(lam_district))

test_weights = [0, 0.001, 0.005, 0.01, 0.03]

all_stats = []


for w in test_weights:
    lam = read_model2(f"results/lambda_model2_w{w}.txt", 3, 76, 336)
    all_stats.append(Stats(lam))
    
lam_pop_rect = read_model3(f"results/lambda_model3_R76_T48_old.txt",3 , 76, 336)
stats_pop_rect = Stats(lam_pop_rect)
lam_pop_district = read_model3(f"results/lambda_model3_R160_T48_old.txt", 3, 160, 336)
stats_pop_district = Stats(lam_pop_district)
COLORS = ["black", "red", "orange", "blue"]
LINES = ["solid", "dotted", "dashed", (0, (3, 1, 1, 1)), (0, (1, 1))]
print(f"len all_stats = {len(all_stats)}")
print(f"Shape 0 = {all_stats[0].mean_total}, shape 2 = {all_stats[2].mean_total}")
plt.plot([t for t in range(T)], stats_miss_rect.mean_total,label=f"Regularized Rectangular", 
         color=COLORS[0], linestyle=LINES[0])
plt.plot([t for t in range(T)], stats_miss_district.mean_total, label="Regularized District",
         color=COLORS[1], linestyle=LINES[1])
plt.plot([t for t in range(T)], stats_pop_rect.mean_total,label=f"Covariates Rectangular",
         color=COLORS[2], linestyle=LINES[2])
plt.plot([t for t in range(T)], stats_pop_district.mean_total,label=f"Covariates District",
         color=COLORS[3], linestyle=LINES[3])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Intensities")
plt.savefig("results/model1_cv.pdf", bbox_inches="tight")
plt.close()

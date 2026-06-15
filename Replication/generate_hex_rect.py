import os

import geopandas as gpd
import laspated as spated
import numpy as np
import pandas as pd
from shapely.geometry import Point

from replication_script import process_land_use


def _sample_points_in_geometry(geometry, n_points: int, rng: np.random.Generator):
    if hasattr(geometry, "geoms"):
        parts = [g for g in geometry.geoms if not g.is_empty and g.area > 0]
    else:
        parts = [geometry]
    if not parts:
        rp = geometry.representative_point()
        return [(rp.y, rp.x)] * n_points

    areas = np.array([g.area for g in parts], dtype=float)
    probs = areas / areas.sum()

    points = []
    max_tries = n_points * 500
    tries = 0
    while len(points) < n_points and tries < max_tries:
        poly = parts[rng.choice(len(parts), p=probs)]
        minx, miny, maxx, maxy = poly.bounds
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if poly.covers(p):
            points.append((y, x))  # lat, lon
        tries += 1

    if len(points) < n_points:
        rp = geometry.representative_point()
        points.extend([(rp.y, rp.x)] * (n_points - len(points)))
    return points


def write_samples(app: spated.DataAggregator, output_dir: str, n_samples: int = 100) -> None:
    rng = np.random.default_rng(42)
    samples_path = os.path.join(output_dir, "samples.dat")
    with open(samples_path, "w", encoding="utf-8") as f:
        for _, row in app.geo_discretization.iterrows():
            region_id = int(row["id"])
            geometry = row["geometry"]
            sampled = _sample_points_in_geometry(geometry, n_samples, rng)
            for lat, lon in sampled:
                f.write(f"{region_id} {lat:.8f} {lon:.8f}\n")


def write_estimation(output_dir: str, delta_t_hours: float = 0.5) -> None:
    info_path = os.path.join(output_dir, "info.dat")
    arrivals_path = os.path.join(output_dir, "arrivals.dat")
    output_path = os.path.join(output_dir, "sorted_empirical_estimation.dat")

    with open(info_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        t_size, d_size, r_size, p_size = [int(x) for x in header[:4]]
        n_obs_dow = [int(x) for x in f.readline().strip().split()[:d_size]]

    sums = {}
    with open(arrivals_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            t, d, r, p, _n, val = [int(x) for x in parts[:6]]
            key = (t, d, r, p)
            sums[key] = sums.get(key, 0) + val

    with open(output_path, "w", encoding="utf-8") as f:
        for t in range(t_size):
            for d in range(d_size):
                denom = n_obs_dow[d] * delta_t_hours
                for r in range(r_size):
                    for p in range(p_size):
                        num = sums.get((t, d, r, p), 0)
                        lam = num / denom if denom > 0 else 0.0
                        f.write(f"{t} {d} {r} {p} {lam:.10f}\n")


def generate_disc(output_name: str, disc_type: str) -> int:
    app = spated.DataAggregator(crs="epsg:4326")
    max_borders = gpd.read_file("rj/rj.shp")

    events = pd.read_csv("sorted_events.csv", encoding="ISO-8859-1", sep=",")
    app.add_events_data(
        events,
        datetime_col="data_hora",
        lat_col="lat",
        lon_col="long",
        feature_cols=["prioridade"],
        datetime_format="%m/%d/%y %H:%M:%S",
    )

    app.add_max_borders(max_borders)
    app.add_time_discretization("m", 30, 60 * 24, column_name="hhs")
    app.add_time_discretization("D", 1, 7, column_name="dow")

    if disc_type == "rect":
        app.add_geo_discretization(
            discr_type="R",
            rect_discr_param_x=15,
            rect_discr_param_y=15,
        )
    elif disc_type == "hex":
        app.add_geo_discretization(discr_type="H", hex_discr_param=8)
    else:
        raise ValueError(f"disc_type invalido: {disc_type}")

    land_use = process_land_use()
    app.add_geo_variable(land_use, type_geo_variable="area")

    population = gpd.read_file("regressores/populacao/")
    population = population[["population", "geometry"]].copy()
    app.add_geo_variable(population)

    app.geo_discretization["population"] /= 10**4

    num_regions = int(np.max(app.events_data["gdiscr"]) + 1)
    app.plot_discretization(to_file=f"replication_results/disc_{output_name}_r{num_regions}.pdf")

    output_dir = f"../Data/discretizations/{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    app.write_arrivals(f"{output_dir}/arrivals.dat")
    app.write_regions(f"{output_dir}/neighbors.dat")
    app.write_info("dow", path=f"{output_dir}/info.dat")
    write_estimation(output_dir, delta_t_hours=0.5)
    write_samples(app, output_dir, n_samples=100)

    print(f"[ok] {output_name}: {num_regions} regioes")
    return num_regions


def main() -> None:
    R_rect = generate_disc(output_name="rect_15x15", disc_type="rect")
    R_hex = generate_disc(output_name="hex_8", disc_type="hex")
    print(f"Regioes retangulares: {R_rect}, Regioes hexagonais: {R_hex}")
    

if __name__ == "__main__":
    main()

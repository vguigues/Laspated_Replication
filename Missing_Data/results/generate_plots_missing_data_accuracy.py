import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_accuracy_results(file_path):
    
    df = pd.read_csv(file_path, sep=" ", header=None)
    df.columns = ["model", "month", "c", "d", "t", "mse"]
    C = 3
    D = 7
    T = 48
    M = 24
    # Create ((C,D,T)) numpy array to store each model
  
    mse_analytical = np.zeros((M, C, D, T))
    mse_regularized = np.zeros((M, C, D, T))
    mse_arrival_covariates = np.zeros((M, C, D, T))
    for index, row in df.iterrows():
        if row["model"] == 1:
            m = int(row["month"])
            c = int(row["c"])
            d = int(row["d"])
            t = int(row["t"])
            mse_analytical[m, c, d, t] = row["mse"]
        elif row["model"] == 2:
            m = int(row["month"])
            c = int(row["c"])
            d = int(row["d"])
            t = int(row["t"])
            mse_regularized[m, c, d, t] = row["mse"]
        elif row["model"] == 3:
            m = int(row["month"])
            c = int(row["c"])
            d = int(row["d"])
            t = int(row["t"])
            mse_arrival_covariates[m, c, d, t] = row["mse"]
    # Compute the mean over the months for each model
    mse_analytical = np.mean(mse_analytical, axis=0)
    mse_regularized = np.mean(mse_regularized, axis=0)
    mse_arrival_covariates = np.mean(mse_arrival_covariates, axis=0)
    return mse_analytical, mse_regularized, mse_arrival_covariates


def plot_accuracy_results(mse_analytical, mse_regularized, mse_arrival_covariates):
    analytical_vals = mse_analytical.flatten()
    regularized_vals = mse_regularized.flatten()
    arrival_covariates_vals = mse_arrival_covariates.flatten()

    xmin = min(analytical_vals.min(), regularized_vals.min(), arrival_covariates_vals.min())
    xmax = max(analytical_vals.max(), regularized_vals.max(), arrival_covariates_vals.max())
    common_range = (xmin, xmax)
    common_bins = 20

    # Three side-by-side histograms with a shared x-axis range
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(analytical_vals, bins=common_bins, range=common_range)
    plt.xlim(common_range)
    plt.title("Model from Section 2.1")
    plt.xlabel("Mean MSE")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(regularized_vals, bins=common_bins, range=common_range)
    plt.xlim(common_range)
    plt.title("Regularized model from Section 2.2")
    plt.xlabel("Mean MSE")

    plt.subplot(1, 3, 3)
    plt.hist(arrival_covariates_vals, bins=common_bins, range=common_range)
    plt.xlim(common_range)
    plt.title("Covariates model from Section 2.3")
    plt.xlabel("Mean MSE")

    plt.tight_layout()
    plt.savefig("accuracy_results.pdf", bbox_inches="tight")

    # Generate the histograms as separate files (same x-axis and bins)
    plt.figure(figsize=(5, 5))
    plt.hist(analytical_vals, bins=common_bins, range=common_range)
    plt.xlim(common_range)
    plt.savefig("mse_analytical_histogram.pdf", bbox_inches="tight")

    plt.figure(figsize=(5, 5))
    plt.hist(regularized_vals, bins=common_bins, range=common_range)
    plt.xlim(common_range)
    plt.savefig("mse_regularized_histogram.pdf", bbox_inches="tight")

    plt.figure(figsize=(5, 5))
    plt.hist(arrival_covariates_vals, bins=common_bins, range=common_range)
    plt.xlim(common_range)
    plt.savefig("mse_arrival_covariates_histogram.pdf", bbox_inches="tight")
def table_global_errors(mse_analytical, mse_regularized, mse_arrival_covariates):
    # Write a latex table with the mean and std MSE values for each model
    with open("mse_accuracy_table.tex", "w") as f:
        f.write("\\begin{tabular}{lccc}\\n")
        f.write("\\hline\\n")
        f.write("Model & Mean MSE & Std MSE \\\\n")
        f.write("\\hline\\n")
        f.write("Model from Section 2.1 & {:.4f} & {:.4f} \\\\n".format(np.mean(mse_analytical), np.std(mse_analytical)))
        f.write("Regularized model from Section 2.2 & {:.4f} & {:.4f} \\\\n".format(np.mean(mse_regularized), np.std(mse_regularized)))
        f.write("Covariates model from Section 2.3 & {:.4f} & {:.4f} \\\\n".format(np.mean(mse_arrival_covariates), np.std(mse_arrival_covariates)))
        f.write("\\hline\\n")
        f.write("\\end{tabular}\\n")


if __name__ == "__main__":
    mse_analytical, mse_regularized, mse_arrival_covariates = read_accuracy_results("accuracy_results.txt")
    plot_accuracy_results(mse_analytical, mse_regularized, mse_arrival_covariates)
    table_global_errors(mse_analytical, mse_regularized, mse_arrival_covariates)

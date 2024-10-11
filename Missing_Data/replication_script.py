import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
from subprocess import PIPE, Popen
import os
from shutil import copyfile
import time

tic = time.perf_counter()
R = 76
S = 30
D = 7
T = 48
C = 3

nb_obs = [104, 104, 104, 104, 105, 105, 105]
max_obs = np.max(nb_obs)


def read_sample(path="Rect10x10/missing_calls.dat"):
    arq = open(path, "r")
    sample_missing_calls = np.zeros((C, D, T, max_obs))
    for line in arq.readlines():
        if line == "END":
            break
        tokens = line.split()
        t = int(tokens[0])
        d = int(float(tokens[1]))
        c = int(tokens[3])
        n = int(tokens[4])
        val = int(tokens[5])
        sample_missing_calls[c, d, t, n] = val
    arq.close()

    return sample_missing_calls


def read_neighbors(path="Rect10x10/pop.dat"):
    arq = open(path, "r")
    pops = []
    for line in arq.readlines():
        if line == "END":
            break
        tokens = line.split()
        pops.append(float(tokens[4]))
    arq.close()

    return pops


sample_arrivals = read_sample()
pops = read_neighbors()
total_pop = np.sum(pops)
probs = [x / total_pop for x in pops]

mn_samples = np.zeros((C, D, T, max_obs, S, R))
output_file = f"Rect10x10/mn_samples.dat"
arq = open(output_file, "w")
for c in range(C):
    for d in range(D):
        print("Generating multinomial samples")
        for t in range(T):
            for n in range(nb_obs[d]):
                mn = np.random.multinomial(sample_arrivals[c, d, t, n], probs, size=S)
                for s in range(S):
                    for r in range(R):
                        arq.write(f"{c} {d} {t} {n} {s} {r} {mn[s,r]}\n")
arq.write("END\n")
arq.close()

print("Wrote mn_samples.txt")

try:
    with Popen(
        ["./missing", "-f", "test.cfg"],
        bufsize=1,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    ) as p:
        for line in p.stdout:
            print(line, end="",flush=True)

        for line in p.stderr:
            print(line, end="",flush=True)
except:
    print(f"Error with C++ experiments")
    exit(1)


def read_model1(path):
    arq = open(path, "r")
    lam_emp = np.zeros((C, R, T))
    lam_est = np.zeros((C, R, T))
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


def read_model2(path):
    lam = np.zeros((C, R, T))
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


def read_model3(path):
    arq = open(path, "r")
    lam = np.zeros((C, R, T))
    for line in arq.readlines():
        tokens = line.split()
        c = int(tokens[0])
        r = int(tokens[1])
        t = int(tokens[2])
        val_m1 = float(tokens[3])
        val_m2 = float(tokens[4])
        lam[c, r, t] = val_m2
    arq.close()
    return lam


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
            self.mean_total = np.sum(lam[:, :, t])
            self.q05_total = stats.poisson.ppf(0.05, self.mean_total)
            self.q95_total = stats.poisson.ppf(0.95, self.mean_total)
            self.mean_p0 = np.sum(lam[0, :, t])
            self.q05_p0 = stats.poisson.ppf(0.05, self.mean_p0)
            self.q95_p0 = stats.poisson.ppf(0.95, self.mean_p0)
            self.mean_p1 = np.sum(lam[1, :, t])
            self.q05_p1 = stats.poisson.ppf(0.05, self.mean_p1)
            self.q95_p1 = stats.poisson.ppf(0.95, self.mean_p1)
            self.mean_p2 = np.sum(lam[2, :, t])
            self.q05_p2 = stats.poisson.ppf(0.05, self.mean_p2)
            self.q95_p2 = stats.poisson.ppf(0.95, self.mean_p2)


lam_emp, lam_est = read_model1("results/lambda_model1.txt")
stats_emp = Stats(lam_emp)
stats_est = Stats(lam_est)

test_weights = [0, 0.001, 0.005, 0.01, 0.03]

all_stats = []

for w in test_weights:
    lam = read_model2(f"results/lambda_model2_w{w}.txt")
    all_stats.append(Stats(lam))


COLORS = ["black", "red"]
LINES = ["solid", "dotted", "dashed", (0, (3, 1, 1, 1)), (0, (1, 1))]

for i, w in enumerate(test_weights):
    plt.plot(
        [t for t in range(T)],
        all_stats[i].mean_total,
        label=f"W{w}",
        color=COLORS[i % 2],
        linestyle=LINES[i],
    )
plt.xlabel("Time")
plt.ylabel("Intensities")
plt.savefig("results/model1_weights.pdf", bbox_inches="tight")
plt.close()

for i, w in enumerate(test_weights):
    plt.plot(
        [t for t in range(T)],
        all_stats[i].mean_p0,
        label=f"W{w}",
        color=COLORS[i % 2],
        linestyle=LINES[i],
    )
plt.xlabel("Time")
plt.ylabel("Intensities")
plt.savefig("results/model1_weights_p0.pdf", bbox_inches="tight")
plt.close()


for i, w in enumerate(test_weights):
    plt.plot(
        [t for t in range(T)],
        all_stats[i].mean_p1,
        label=f"W{w}",
        color=COLORS[i % 2],
        linestyle=LINES[i],
    )
plt.xlabel("Time")
plt.ylabel("Intensities")
plt.savefig("results/model1_weights_p1.pdf", bbox_inches="tight")
plt.close()


for i, w in enumerate(test_weights):
    plt.plot(
        [t for t in range(T)],
        all_stats[i].mean_p2,
        label=f"W{w}",
        color=COLORS[i % 2],
        linestyle=LINES[i],
    )
plt.xlabel("Time")
plt.ylabel("Intensities")
plt.savefig("results/model1_weights_p2.pdf", bbox_inches="tight")
plt.close()


toc = time.perf_counter()
print(f"Finished replication script in {toc - tic:0.4f} seconds")

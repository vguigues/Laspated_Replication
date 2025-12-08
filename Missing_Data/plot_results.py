import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

C = 3
R = 76
T = 7 * 48


def read_model1(path, R):
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

def read_conf_intervals_model1(path):
    arq = open(path, "r")
    conf_intervals = np.zeros((C, R, T, 2))
    lam = np.zeros((C, R, T))
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

def read_model2(path,R):
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


def read_model3(path, R):
    arq = open(path, "r")
    lam = np.zeros((C, R, T))
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



lam_emp_rect, lam_est_rect = read_model1(f"results/lambda_model1_R76_T48.txt", 76)
stats_emp_rect = Stats(lam_emp_rect)
stats_est_rect = Stats(lam_est_rect)

lam_emp_district, lam_est_district = read_model1(f"results/lambda_model1_R160_T48.txt", 160)
stats_emp_district = Stats(lam_emp_district)
stats_est_district = Stats(lam_est_district)

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
    lam = read_model2(f"results/lambda_model2_w{w}.txt", 76)
    all_stats.append(Stats(lam))
    
lam_cv = read_model2(f"results/lambda_model2_cv.txt", 160)
stats_cv = Stats(lam_cv)

lam_pop_rect = read_model3(f"results/lambda_model3_R76_T48_old.txt", 76)
stats_pop_rect = Stats(lam_pop_rect)
lam_pop_district = read_model3(f"results/lambda_model3_R160_T48_old.txt", 160)
stats_pop_district = Stats(lam_pop_district)
COLORS = ["black", "red", "orange", "blue"]
LINES = ["solid", "dotted", "dashed", (0, (3, 1, 1, 1)), (0, (1, 1))]
print(f"len all_stats = {len(all_stats)}")
print(f"Shape 0 = {all_stats[0].mean_total}, shape 2 = {all_stats[2].mean_total}")
plt.plot([t for t in range(T)], stats_est_rect.mean_total,label=f"Regularized Rectangular", 
         color=COLORS[0], linestyle=LINES[0])
# plt.plot([t for t in range(T)], stats_cv.mean_total,label=f"Cross validation Rectangular", 
#          color=COLORS[1], linestyle=LINES[1])
plt.plot([t for t in range(T)], stats_est_district.mean_total, label="Regularized District",
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
input()


plt.plot(
    [t for t in range(T)],
    stats_est.mean_total,
    label=f"W{0}",
    color=COLORS[0],
    linestyle=LINES[0],
)
for i, w in enumerate(test_weights[1:]):
    plt.plot(
        [t for t in range(T)],
        all_stats[i+1].mean_total,
        label=f"W{w}",
        color=COLORS[(i+1) % 2],
        linestyle=LINES[i+1],
    )
plt.legend()
plt.xlabel("Time")
plt.ylabel("Intensities")
plt.savefig("results/model1_weights.pdf", bbox_inches="tight")
plt.close()

# for i, w in enumerate(test_weights):
#     plt.plot(
#         [t for t in range(T)],
#         all_stats[i].mean_p0,
#         label=f"W{w}",
#         color=COLORS[i % 2],
#         linestyle=LINES[i],
#     )
# plt.xlabel("Time")
# plt.ylabel("Intensities")
# plt.savefig("results/model1_weights_p0.pdf", bbox_inches="tight")
# plt.close()


# for i, w in enumerate(test_weights):
#     plt.plot(
#         [t for t in range(T)],
#         all_stats[i].mean_p1,
#         label=f"W{w}",
#         color=COLORS[i % 2],
#         linestyle=LINES[i],
#     )
# plt.xlabel("Time")
# plt.ylabel("Intensities")
# plt.savefig("results/model1_weights_p1.pdf", bbox_inches="tight")
# plt.close()


# for i, w in enumerate(test_weights):
#     plt.plot(
#         [t for t in range(T)],
#         all_stats[i].mean_p2,
#         label=f"W{w}",
#         color=COLORS[i % 2],
#         linestyle=LINES[i],
#     )
# plt.xlabel("Time")
# plt.ylabel("Intensities")
# plt.savefig("results/model1_weights_p2.pdf", bbox_inches="tight")
# plt.close()


# conf_intervals, lam = read_conf_intervals_model1("results/conf_intervals_model1.txt")

# C,R,T,_ = conf_intervals.shape

# max_pair = None
# max_val = -np.inf
# for c in range(C):
#     for r in range(R):
#         sum_lam = np.sum(lam[c, r, :])
#         if sum_lam > max_val:
#             max_val = sum_lam
#             max_pair = (c, r)

# print("Max pair:", max_pair, "with sum lambda =", max_val, "mean = ", max_val / T)
# c, r = max_pair
# print("lam", lam[c,r,200:336])
# print("ci_low", conf_intervals[c,r,200:336,0])
# print("ci_high", conf_intervals[c,r,200:336,1])
# # input()


# for c in range(C):
#     plt.figure()
#     plt.xlabel("Time")
#     plt.ylabel("Intensity")
#     plt.plot([t for t in range(T)], lam[c, r, :], label="Lambda", color="black")
#     plt.fill_between(
#         [t for t in range(T)],
#         conf_intervals[c, r, :, 0],
#         conf_intervals[c, r, :, 1],
#         color="gray",
#         alpha=0.5,
#         label="Confidence Interval",
#     )
#     #increase graph size
#     plt.gcf().set_size_inches(10, 6)
#     plt.legend()
#     # plt.show()
#     plt.savefig(f"results/c{c}_r{r}.pdf", bbox_inches="tight")
#     plt.close()


# nb_obs_list = [10, 100, 500, 1000, 2000]

# for nb_obs in nb_obs_list:
#     lam_est = read_model4(f"results/missing_time_lambda_nbobs_{nb_obs}.dat")
#     lam_no_missing = read_model4(f"results/missing_time_lambda_no_missing_nbobs_{nb_obs}.dat")
#     plt.figure()
#     plt.xlabel("Number of calls")
#     plt.ylabel("Probabilities")
#     sum_est = np.sum(lam_est, axis=(1,2))
#     sum_no_missing = np.sum(lam_no_missing, axis=(1,2))
#     est_flat = np.random.poisson(sum_est[0]*0.5, 1000)
#     no_missing_flat = np.random.poisson(sum_no_missing[0]*0.5, 1000)
#     plt.hist(
#         no_missing_flat,
#         bins=40,
#         density=True,
#         color="black",
#         label="Uncorrected",
#     )

#     plt.hist(
#         est_flat,
#         bins=40,
#         density=True,
#         color="red",
#         label="Corrected",
#     )
#     plt.ticklabel_format(style="sci", axis="y", scilimits=(-3, -3))
#     plt.legend()
#     plt.savefig(f"results/missing_time_histogram_nb_obs_{nb_obs}.pdf", bbox_inches="tight")
#     plt.close()
    
    


# # for c in range(C):
# #     plt.figure()
# #     # plt.title(f"Confidence Intervals for C={c}")
# #     plt.xlabel("Time")
# #     plt.ylabel("Intensity")

# #     # Plot the sum of lambda over r for each t
# #     plt.plot([t for t in range(T)], np.sum(lam[c, :, :], axis=0), label="Lambda", color="black")

# #     # Plot the sum of the confidence intervals over r for each t
# #     plt.fill_between(
# #         [t for t in range(T)],
# #         np.sum(conf_intervals[c, :, :, 0], axis=0),
# #         np.sum(conf_intervals[c, :, :, 1], axis=0),
# #         color="gray",
# #         alpha=0.5,
# #         label="Confidence Interval",
# #     )

# #     plt.legend()
# #     plt.savefig(f"results/conf_intervals_model1_c{c}.pdf", bbox_inches="tight")
# #     plt.close()
    
# # # Now plot the sum over c in range(C) and over r in range(R)
# # plt.plot([t for t in range(T)], np.sum(lam, axis=(0,1)), label="Lambda", color="black")
# # plt.fill_between(
# #     [t for t in range(T)],
# #     np.sum(conf_intervals[:, :, :, 0], axis=(0,1)),
# #     np.sum(conf_intervals[:, :, :, 1], axis=(0,1)),
# #     color="gray",
# #     alpha=0.5,
# #     label="Confidence Interval",
# # )
# # plt.legend()
# # plt.savefig("results/conf_intervals_model1_total.pdf", bbox_inches="tight")
# # plt.close()
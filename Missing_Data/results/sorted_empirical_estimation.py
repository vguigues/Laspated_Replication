import numpy as np


C = 3
D = 7
R = 76
T = 48
N = 104

header = f"{T} {D} {R} {C} {N}\n"



def read_model1(path, R):
    arq = open(path, "r")
    lam_emp = np.zeros((C, D, T, R))
    lam_est = np.zeros((C, D, T, R))
    for line in arq.readlines():
        tokens = line.split()
        c = int(tokens[0])
        r = int(tokens[1])
        t = int(tokens[2])
        d = t // T
        t = t % T
        val_emp = float(tokens[3])
        val_est = float(tokens[4])
        lam_emp[c, d, t, r] = val_emp
        lam_est[c, d, t, r] = val_est
    arq.close()

    return lam_emp, lam_est


lam_emp, lam_est = read_model1(f"lambda_model1_R76_T48.txt", 76)

arq_miss_emp = open(f"uncorrected_from_missing.txt", "w")
arq_miss_emp.write(header)

for c in range(C):
    for d in range(D):
        for r in range(R):
            for t in range(T):
                arq_miss_emp.write(f"{t} {d} {r} {c} {lam_emp[c, d, t, r]}\n")

arq_miss_emp.close()

arq_miss_est = open(f"corrected_from_missing.txt", "w")
arq_miss_est.write(header)
for c in range(C):
    for d in range(D):
        for r in range(R):
            for t in range(T):
                arq_miss_est.write(f"{t} {d} {r} {c} {lam_est[c, d, t, r]}\n")
arq_miss_est.close()

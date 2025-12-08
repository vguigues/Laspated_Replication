
import numpy as np

C = 3
D = 7
T = 48
R = 100
nx = 10
ny = 10
nb_obs = 10


assert R == nx * ny, "R must equal nx * ny for a nx-by-ny grid"

def idx(i, j):
    return i * ny + j

region_coords = [(i, j) for i in range(nx) for j in range(ny)]

# grid_neighbors[r] will contain the indices of north/south/east/west neighbors of region r
grid_neighbors = [[] for _ in range(R)]
for i in range(nx):
    for j in range(ny):
        r = idx(i, j)
        if i > 0:
            grid_neighbors[r].append(idx(i - 1, j))   # north
        if i < nx - 1:
            grid_neighbors[r].append(idx(i + 1, j))   # south
        if j > 0:
            grid_neighbors[r].append(idx(i, j - 1))   # west
        if j < ny - 1:
            grid_neighbors[r].append(idx(i, j + 1))   # east
            
# Divide the grid in 4, assigning color blue to quadrants lower left and upper right
mid_x = nx // 2
mid_y = ny // 2

# region_color[r] is 'blue' for lower-left and upper-right quadrants, 'gray' otherwise
region_color = [
    'blue' if ((i >= mid_x and j < mid_y) or (i < mid_x and j >= mid_y)) else 'gray'
    for (i, j) in region_coords
]

# boolean mask for convenience
blue_mask = np.array([c == 'blue' for c in region_color], dtype=bool)


# Create four groups of times
time_groups = [[] for _ in range(4)]
for d in range(D):
    for t in range(12):
        time_groups[0].append((d,t))        
    for t in range(12,24):
        time_groups[1].append((d,t))
    for t in range(24,36):
        time_groups[2].append((d,t))
    for t in range(36,48):        
        time_groups[3].append((d,t))
        

lam_theoretical = np.zeros((C,D,T,R))
for c in range(C):
    for d in range(D):
        for t in range(T):
            for r in range(R):
                base_rate = 1.0
                # Increase rate for blue regions during certain time groups
                if blue_mask[r]:
                    if (d,t) in time_groups[1]:
                        base_rate += 2.0
                    elif (d,t) in time_groups[3]:
                        base_rate += 3.0
                lam_theoretical[c,d,t,r] = base_rate

prob_missing = np.zeros((C,D,R))
# Blue regions have higher missingness probability during days 5 and 6
for c in range(C):
    for d in range(D):
        for r in range(R):
            if blue_mask[r] and (d == 5 or d == 6):
                prob_missing[c,d,r] = 0.4
            else:
                prob_missing[c,d,r] = 0.1

# Generate Poisson samples given lam_theoretical.
sample = np.zeros((C,D,T,R,nb_obs), dtype=int)
for c in range(C):
    for d in range(D):
        for r in range(R):
            for t in range(T):
                sample[c,d,t,r,:] = [int(x) for x in np.random.poisson((1-prob_missing[c,d,r])*lam_theoretical[c,d,t,r]*0.5, nb_obs)]


# Generate Poisson missing samples
missing_samples = np.zeros((C,D,R,nb_obs), dtype=int)
for c in range(C):
    for d in range(D):
        for r in range(R):
            sum_lam = 0.0
            for t in range(T):
                sum_lam += lam_theoretical[c,d,t,r]*0.5
            missing_samples[c,d,r,:] = [int(x) for x in np.random.poisson(prob_missing[c,d,r]*sum_lam, nb_obs)]

arq_theo_lambda = open("Artificial/theoretical_lambda.dat", "w")
arq_theo_lambda.write(f"{C} {D} {T} {R} {nb_obs}\n")
for c in range(C):
    for d in range(D):
        for t in range(T):
            for r in range(R):
                arq_theo_lambda.write(f"{c} {d} {t} {r} {lam_theoretical[c,d,t,r]}\n")
arq_theo_lambda.close()

arq_art_samples = open("Artificial/calls.dat", "w")
#t,g,r,p,j,val
for t in range(T):
    for d in range(D):
        for r in range(R):
            for c in range(C):
                for j in range(nb_obs):
                    arq_art_samples.write(f"{t} {d} {r} {c} {j} {sample[c,d,t,r,j]}\n")
arq_art_samples.write("END")                    
arq_art_samples.close()

arq_art_missing = open("Artificial/missing_calls.dat", "w")
for d in range(D):
    for r in range(R):
        for c in range(C):
            for j in range(nb_obs):
                arq_art_missing.write(f"{d} {r} {c} {j} {missing_samples[c,d,r,j]}\n")
arq_art_missing.write("END")                    
arq_art_missing.close()


lambda_est = np.zeros((C,D,T,R))
for c in range(C):
    for d in range(D):
        for r in range(R):
            m0_cdr = np.sum(missing_samples[c,d,r,:])
            m1_cdr = 0.0
            for t in range(T):
                m1_cdr += np.sum(sample[c,d,t,r,:])
            for t in range(T):
                m1_cdtr = np.sum(sample[c,d,t,r,:])
                n_cdtr = nb_obs
                lambda_est[c,d,t,r] = ((m0_cdr + m1_cdr) / m1_cdr) * (m1_cdtr / n_cdtr)
                lambda_est[c,d,t,r] /= 0.5
                # print(f"m0_cdr = {m0_cdr}, m1_cdr = {m1_cdr}, m1_cdtr = {m1_cdtr}, n_cdtr = {n_cdtr}, lambda = {lambda_est[c,d,t,r]}")
                # input("")

sum_diff = 0.0
for c in range(C):
    for d in range(D):
        for t in range(T):
            for r in range(R):
                sum_diff += abs(lambda_est[c,d,t,r] - lam_theoretical[c,d,t,r]) / lam_theoretical[c,d,t,r]

avg_diff = sum_diff / (C*D*T*R)
print(f"Average relative difference between estimated and theoretical lambda: {avg_diff}")

p_est = np.zeros((C,D,R))
for c in range(C):
    for d in range(D):
        for r in range(R):
            m0_cdr = np.sum(missing_samples[c,d,r,:])
            m1_cdr = 0.0
            for t in range(T):
                m1_cdr += np.sum(sample[c,d,t,r,:])
            p_est[c,d,r] = m0_cdr / (m0_cdr + m1_cdr)
            
sum_diff_p = 0.0
for c in range(C):  
    for d in range(D):
        for r in range(R):
            sum_diff_p += abs(p_est[c,d,r] - prob_missing[c,d,r]) / prob_missing[c,d,r] 

avg_diff_p = sum_diff_p / (C*D*R)
print(f"Average relative difference between estimated and theoretical missing probability: {avg_diff_p}")

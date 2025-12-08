
import numpy as np



def read_missing_calls(path):
    # Create (48, 7, 3) numpy array of empty lists
    missing_calls = np.empty((48, 7, 3), dtype=object)
    for i in range(48):
        for j in range(7):
            for k in range(3):
                missing_calls[i, j, k] = []

    
    arq = open(path, "r")
    for line in arq.readlines():
        if line.strip() == "END":
            break
        tokens = line.split()
        t,g,_,p = int(tokens[0]), int(float(tokens[1])), float(tokens[2]), int(tokens[3])
        missing_calls[t, g, p].append(int(tokens[5]))
    arq.close()

    return missing_calls

missing_calls = read_missing_calls("Rect10x10/missing_calls.dat")
arq = open("Rect10x10_1h/missing_calls.dat", "w")
for g in range(7):
    for p in range(3):
        for t in range(0,48,2):
            nb_obs = max(len(missing_calls[t,g,p]), len(missing_calls[t+1,g,p]))
            for n in range(nb_obs):
                val1 = missing_calls[t,g,p][n] if n < len(missing_calls[t,g,p]) else 0
                val2 = missing_calls[t+1,g,p][n] if n < len(missing_calls[t+1,g,p]) else 0
                arq.write(f"{t//2} {float(g)} -1.0 {p} {n} {val1 + val2}\n")
arq.close()
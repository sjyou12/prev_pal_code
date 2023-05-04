import numpy as np

read_filtered = np.load("../results/sanod/filtered_run53.npy")
# read_filtered = np.transpose(read_filtered)

q_val_cut = np.load('../results/signal_per_delay/cut_q_range.npy')

time_delay = [-3, -2, -1.5, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 7, 10, 13, 18, 24, 32, 42, 56, 75, 100, 130, 180, 240, 320, 420, 560, 750, 1e3, 1.3e3, 1.8e3, 2.4e3]

outFp = open("../results/sanod/filtered_run53.dat", 'w')

outFp.write("q_val\t")
for each_t in time_delay:
    outFp.write(str(each_t) + '\t')
outFp.write('\n')

for q_idx, q_val in enumerate(q_val_cut):
    outFp.write("%.5f\t" % q_val)
    for t_idx in range(len(time_delay)):
        outFp.write("%.5f\t" % read_filtered[q_idx][t_idx])
    outFp.write("\n")

outFp.close()
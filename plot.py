import numpy as np
import matplotlib.pyplot as plt

raw = np.load("assignmentErr.npz")

# var implies either assignment/execution/assessment error.
var_arr = raw['assErr']
soc_norm = raw['soc_norm']
c_arr = raw['cfrac']
Labels = ["SJ", "SS", "SH", "IS"]
soc_label = -1
for snorm in soc_norm:
    soc_label = soc_label + 1
    plt.plot(var_arr, c_arr[soc_label,:],
             'o-', label=Labels[soc_label])#, colors=["r","g","b",'k'])
plt.xscale("log")
plt.ylim(0,1)
plt.legend()
plt.ylabel("cooperation index")
plt.xlabel("assignment error")
plt.show()

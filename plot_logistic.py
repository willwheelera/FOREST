import numpy as np
import matplotlib.pyplot as plt

t = np.arange(2025, 2065)
f_now = 0.1
x = -np.log(1/f_now - 1) # = (y-y0)/a

plt.figure(figsize=(3,3))
for y50 in [2035, 2040, 2050]:
    a = (2024 - y50) / x
    p = 1 / (1 + np.exp(-(t - y50)/a))
    plt.plot(t, p, label=y50)
    plt.axvline(x=y50, lw=0.5, c="gray")
plt.axhline(y=0.5, lw=0.5, ls=":", c="gray")
plt.legend()
plt.xlabel("year")
plt.ylabel("adoption level")

plt.savefig("figures/logistic_growth.pdf", bbox_inches="tight")
plt.show()
    


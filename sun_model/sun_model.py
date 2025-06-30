import numpy as np
import matplotlib.pyplot as plt

TAU = 2 * np.pi
t = np.arange(8760) 
BETA = np.deg2rad(45) # Latitude
GAMMA = np.deg2rad(23.44) # Earth tilt
ALPHA = -GAMMA * np.cos(TAU * (t/8760 + 10/365))

#Svec = np.array([np.cos(ALPHA), 0, np.sin(ALPHA)])
#Nvec = np.array([np.cos(BETA) * np.cos(t),
#                 np.cos(BETA) * np.sin(t),
#                 np.sin(BETA)])
dot = np.cos(BETA) * np.cos(t/24 * TAU) * np.cos(ALPHA) + \
      np.sin(BETA) * np.sin(ALPHA)
#np.cos(BETA) * np.sin(t/24 * TAU) + \
Inc = np.maximum(dot, 0)

plt.plot(Inc)
plt.xlabel("hours")
plt.ylabel("solar irradiance")
plt.savefig("solar_model.pdf", bbox_inches="tight")
plt.show()

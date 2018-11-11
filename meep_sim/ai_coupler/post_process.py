import numpy as np
import matplotlib.pyplot as plt

f = np.genfromtxt("taper_data.dat", delimiter=",")
Lt = f[:,0]
Rmode = f[:,1]
Rflux = f[:,2]

plt.figure(dpi=150)
plt.loglog(Lt,Rmode,'bo-',label='mode decomposition')
plt.loglog(Lt,Rflux,'ro-',label='flux')
plt.loglog(Lt,0.005/Lt**2,'k-',label=r'quadratic reference (1/Lt$^2$)')
plt.legend(loc='upper right')
plt.xlabel(r'taper length Lt ($\mu$m)')
plt.ylabel('reflectance')
plt.show()

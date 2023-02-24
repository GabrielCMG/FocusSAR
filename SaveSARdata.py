import matplotlib.pyplot as plt
import scipy
from sarpy.io.complex.sicd import SICDReader
import numpy as np

reader = SICDReader('Data/sicd_example_1_PFA_RE32F_IM32F_VV.nitf')

data = reader[:]

print(np.quantile(np.abs(data), 0.99))
print(np.abs(data).max())

data[np.where(np.abs(data)>np.quantile(np.abs(data), 0.99))] = np.quantile(np.abs(data), 0.99)

plt.figure()
plt.imshow(np.abs(data), cmap='gray', aspect='auto')
plt.ylabel("Range [sample]")
plt.xlabel("Azimuth [sample]")
plt.title("Real SAR image")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(np.rot90(data[2000:, 2000:3500])), cmap='gray', aspect='auto')
plt.xlabel("Range [sample]")
plt.ylabel("Azimuth [sample]")
plt.title("Part of the real SAR image")
plt.colorbar()
plt.show()


# dico = {'imag_f' : np.rot90(data[2000:, 2000:3500])}
#
# print(dico['imag_f'])
#
# scipy.io.savemat('Data/realData.mat', dico)

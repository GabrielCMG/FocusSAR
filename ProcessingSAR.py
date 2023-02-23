import matplotlib.pyplot as plt
import numpy as np
import scipy

# -------------------------------------------- #
# ------      Main SAR Parameters       ------ #
# -------------------------------------------- #

# Radar Parameters
PRF = 300       # Pulse Repetition Frequency (Hz)
dur = 3         # Time of Flight (sec), PRF*dur = received echoes
vp = 200        # Velocity of platform
fc = 4.5e9      # Carrier frequency (4.5GHz)
Tp = 0.25e-5    # Chirp Pulse Duration
B0 = 100e6      # Baseband bandwidth is plus/minus B0

# Configuration Parameters
theta = 45      # Look Angle in degrees
Ro = 20e3       # Minimum Distance to Center of Target

# General Variables
c = 3e8         # Propagation speed

# -------------------------------------------- #
# ------            RAW DATA            ------ #
# -------------------------------------------- #

matInit = scipy.io.loadmat('Data/cpxtarget_3s.mat')
s = matInit['s']


v = np.vstack([s[:40], s[:40], s[:40], s, s[860:], s[860:], s[860:]])

plt.figure()
plt.imshow(np.abs(v), aspect='auto', cmap='gray')
plt.title("Raw SAR data")
plt.xlabel("Range [sample]")
plt.ylabel("Azimuth [sample]")
plt.colorbar()
plt.show()

# -------------------------------------------- #
# ------         INITIALIZATION         ------ #
# -------------------------------------------- #

# General Variables
lamb = c/fc     # Propagation speed

# Range Parameters
Kr = B0/Tp      # Range Chirp FM Rate
dt = 1/(2*B0)   # Time Domain Sampling Interval

# Measurement Parameters
acq, rbins = s.shape    # Number of acquisition (acq) and range samples rbins
tau = np.arange(0, rbins*dt, dt)    # Number of acquisition (acq) and range samples rbins
Td = np.max(tau)        # Range acquisition duration

# Azimuth Parameters
Ka = 2*vp**2/lamb/Ro   # Linear Azimuth FM rate
eta = np.arange(-dur/2, dur/2, dur/acq)

# -------------------------------------------- #
# ------  RANGE DOPLER ALGORITHM (RDA)  ------ #
# -------------------------------------------- #

# Range Reference Signal
wr = lambda x: np.logical_and(x/Tp<1, x/Tp>0)
s0 = wr(tau - (Td/2-Tp/2)) * np.exp(1j*np.pi*Kr*(tau - (Td/2-Tp/2))**2)

# Range Compression
sc = np.fft.ifft(np.conj(np.fft.fft(s0))*np.fft.fft(s, axis=1), axis=1)

plt.figure()
plt.imshow(np.abs(sc), aspect='auto')
plt.title("Image après analyse en distance")
plt.xlabel("Distance [échantillons]")
plt.ylabel("Azimut [échantillons]")
plt.colorbar()
plt.show()

# Range Cell Migration Correction (RCMC)
DR = np.transpose(vp**2*eta**2/2/Ro)
Dt = (2*DR/c).reshape(-1, 1)

Fe = 1/dt
f = np.arange(-Fe/2, Fe/2, Fe/rbins).reshape(1, -1)

sc_rcmc = np.fft.ifft(np.exp(1j*2*np.pi*Dt@f)*np.fft.fftshift(np.fft.fft(sc, axis=1), 1), axis=1)

plt.figure()
plt.imshow(np.abs(sc_rcmc), aspect='auto')
plt.title("Image après correction des migrations")
plt.xlabel("Distance [échantillons]")
plt.ylabel("Azimut [échantillons]")
plt.colorbar()
plt.show()

# Azimuth Reference Signal
s0_a = np.transpose(np.exp(-1j*np.pi*Ka*eta**2))

# Azimuth Compression
imag_f = np.fft.fftshift(np.fft.ifft(np.conj(np.fft.fft(s0_a)).reshape(-1, 1)*np.fft.fft(sc_rcmc, axis=0), axis=0))

# -------------------------------------------- #
# ------       Plot Final Results       ------ #
# -------------------------------------------- #

imag_f = np.vstack([imag_f[:40], imag_f[:40], imag_f[:40], imag_f, imag_f[860:], imag_f[860:], imag_f[860:]])

print(imag_f.shape)

def deletePeak(image):
    image[np.where(np.abs(image) > np.quantile(np.abs(image), 0.99))] = np.quantile(np.abs(image), 0.99)
    return image

plt.figure()
plt.imshow(deletePeak(np.abs(imag_f)), aspect='auto', cmap='gray')
plt.title("Final SAR image")
plt.xlabel("Range [sample]")
plt.ylabel("Azimuth [sample]")
plt.colorbar()
plt.show()



dico = {'imag_f' : imag_f}

print(dico['imag_f'])

scipy.io.savemat('Data/simuData.mat', dico)

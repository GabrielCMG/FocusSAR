import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import Util


def SLambda(g, lamb):
    """
    Function that calculates the threshold of a complex image.

    :param g: Complex image to be thresholded
    :param lamb: Threshold
    :return: Thresholded image
    """
    mask = g - lamb < 0

    gPlus = g - lamb
    gPlus[mask] = 0

    gAngle = np.angle(g)

    return gPlus * np.exp(1j*gAngle)


def PEA(image, NIteration):
    """
    Function that performs Phase Error Autofocus on a SAR image.

    :param image: SAR image that need to be focused
    :param NIteration: Number of iterations the algorithm will perform
    :return: Focused image
    """
    g = image.copy()
    G = np.fft.fft(g, axis=0)

    i = 0
    gHat_i = g

    lamb = 0.65
    alpha = 1

    while i < NIteration:
        gBarre_i = SLambda(gHat_i, alpha**i*lamb)

        GBarre_i = np.fft.ifft(gBarre_i, axis=0)

        Phi_i = np.angle(np.sum(np.conj(G)*GBarre_i, axis=1))

        gHat_ii = Util.phaseCorrection(G, Phi_i)

        print(f"Current entropy of the corrected image : {Util.entropyCalculation(gHat_ii)}")

        gHat_i = gHat_ii
        i += 1

    plt.figure()
    plt.imshow(Util.deletePeak(np.abs(gHat_i)), cmap='gray', aspect='auto')
    plt.xlabel("Range [sample]")
    plt.ylabel("Azimut [sample]")
    plt.title("Focused image by the MEA algorithm")
    plt.colorbar()

    print(f"Focused image by the PEA : {Util.entropyCalculation(gHat_i)}")

    return gHat_i


if __name__ == "__main__":
    matInit = scipy.io.loadmat('Data/simuData.mat')
    matInit = scipy.io.loadmat('Data/realData.mat')
    matInit = matInit['imag_f']

    matInit = matInit/np.abs(matInit).max()

    PEType = 3
    Noise = 100

    matInitBruit = Util.addNoise(matInit, SNR=Noise)
    matDefocus = Util.defocus(matInitBruit, PEType)
    matDefocus = matDefocus/np.abs(matDefocus).max()

    plt.figure()
    plt.imshow(Util.deletePeak(np.abs(matInit)), cmap='gray', aspect='auto')
    plt.xlabel("Range [sample]")
    plt.ylabel("Azimut [sample]")
    plt.title(f"Initial image perfectly focused (noise = {Noise}dB)")
    plt.colorbar()

    plt.figure()
    plt.imshow(Util.deletePeak(np.abs(matDefocus)), cmap='gray', aspect='auto')
    plt.xlabel("Range [sample]")
    plt.ylabel("Azimut [sample]")
    plt.title(f"Defocused image (Type {PEType} phase error)")
    plt.colorbar()

    t0 = time.time()
    PEA(matDefocus, 30)

    print(f"Execution time : {time.time()-t0}s")
    print(f"Initial image entropy : {Util.entropyCalculation(matInit)}")
    print(f"Defocused image entropy : {Util.entropyCalculation(matDefocus)}")

    plt.show()

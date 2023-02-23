import matplotlib.pyplot as plt
import numpy as np
import scipy
import Util


def calculerPhi(betaList, M, k0):
    """
    Calculates the filter to be used in the phaseCorrection() function.

    :param betaList: List of coefficients used to construct the filter
    :param M: Maximum index of the azimuth axe
    :param k0: Doppler centroid
    :return: Phase
    """
    I = np.size(betaList)

    phi = np.zeros((M, 1))

    for i in range(2, I):
        for k in range(0, M):
            if k < k0 + M / 2:
                coefSup = 0
            else:
                coefSup = M

            phi[k] -= np.pi * betaList[i] / i * (2 * (k - k0 - coefSup) / M) ** i

    return phi


def augmenterBetaI(signalToFocus, E, betaList, i, d, k0):
    """
    Function increasing the value of the Beta coefficient of index i until it no longer minimises the entropy. The 
    function returns the minimum entropy obtained as well as the list of Betas containing the Beta that has been 
    modified. If increasing the value of the i-th Beta does not decrease the entropy, the initial entropy and the 
    list of Beta.

    :param signalToFocus: Fourier transform in azimuth of the signal to be focused
    :param E: Initial entropy of the signal to be minimised
    :param betaList: List containing the Beta already calculated and those to be calculated
    :param i: Index of the Beta to be modified
    :param d: Step size to modify Beta
    :param k0: Doppler centroid
    :return: Final entropy and modified Beta list
    """
    M, N = signalToFocus.shape

    lamb = 2*E+1

    while lamb > E:
        lamb = E

        betaList[i] += d
        phi = calculerPhi(betaList, M, k0)

        E = Util.entropyCalculation(Util.phaseCorrection(signalToFocus, phi))

    betaList[i] -= d

    E = lamb

    return E, betaList


def diminuerBetaI(signalToFocus, E, betaList, i, d, k0):
    """
    Function that decreases the value of the Beta coefficient of index i until it no longer minimises the entropy. The
    function returns the minimum entropy obtained as well as the list of Betas containing the Beta that has been
    modified. If reducing the value of the i-th Beta does not decrease the entropy, the initial entropy and the list
    of unmodified Betas are returned.

    :param signalToFocus: Fourier transform in azimuth of the signal to be focused
    :param E: Initial entropy of the signal to be minimised
    :param betaList: List containing the Beta already calculated and those to be calculated
    :param i: Index of the Beta to be modified
    :param d: Step size to modify Beta
    :param k0: Doppler centroid
    :return: Final entropy and list of modified Beta
    """
    M, N = signalToFocus.shape

    lamb = 2*E+1

    while lamb > E:
        lamb = E

        betaList[i] -= d
        phi = calculerPhi(betaList, M, k0)

        E = Util.entropyCalculation(Util.phaseCorrection(signalToFocus, phi))

    betaList[i] += d

    E = lamb

    return E, betaList


def minimiseEntropy(signalToFocus, betaList, i, d, k0):
    """
    Function searching for the i-th Beta coefficient to construct a filter to focus the SAR image.

    The algorithm used here is based on the article SAR Minimum-Entropy Autofocus Using an Adaptive-Order Polynomial
    Model écrit par Junfeng Wang et Xingzhao Liu in 2006.

    :param signalToFocus: Fourier transform in azimuth of the signal to be focused
    :param betaList: List containing the Beta already calculated and those to be calculated
    :param i: Index of the Beta to be modified
    :param d: Step size for modifying Beta
    :param k0: Doppler centroid
    :return: List of modified Beta
    """

    E = Util.entropyCalculation(Util.phaseCorrection(signalToFocus))

    while d >= 2:
        start = betaList[i]

        E, betaList = augmenterBetaI(signalToFocus, E, betaList, i, d, k0)

        if betaList[i] == start:
            E, betaList = diminuerBetaI(signalToFocus, E, betaList, i, d, k0)

        d /= 2

    return betaList


def estimationBeta(signalToFocus, ordrePhi, k0):
    """
    Function estimating the list of Beta coefficients allowing to reconstruct the image by minimising its Entropy and
    thus to perform its focus.

    The algorithm used here is based on the article SAR Minimum-Entropy Autofocus Using an Adaptive-Order Polynomial
    Model écrit par Junfeng Wang et Xingzhao Liu in 2006.

    :param signalToFocus: Fourier transform in azimuth of the signal to be focused
    :param ordrePhi: Maximum order of the polynomial allowing to approach the phases of the Focus filter
    :param k0: Doppler centroid
    :return: Final Beta list
    """
    betaListe = np.zeros((ordrePhi, 1))

    i = 1
    betaI = 1
    gamma = 1

    while (betaI != 0 or gamma != 0) and i < ordrePhi-1:

        gamma = betaI
        i += 1
        d = 32

        betaListe = minimiserEntropie(signalToFocus, betaListe, i, d, k0)

        betaI = betaListe[i]
        print(f"Fin du calcul pour l'ordre {i}, Beta[{i}] = {betaI[0]}")

    print(f"Fin du calcul")

    return betaListe


def MEA(image):
    """
    Function that performs Minimum Entropy Autofocus on a SAR image.

    :param image: SAR image that need to be focused
    :return: Focused image
    """
    imageFFT = np.fft.fft(image, axis=0)

    M, N = imageFFT.shape

    k0 = M//2

    betaList = estimationBeta(imageFFT, 50, k0)

    phi = calculerPhi(betaList, M, k0)

    plt.figure()
    plt.plot(-phi)
    plt.title(f"Estimated phase error")
    plt.xlabel("Azimuth [sample]")
    plt.ylabel("Phase error [rad]")

    image = Util.phaseCorrection(imageFFT, phi)

    print(f"Image reconstruite : {Util.entropyCalculation(image)}")

    plt.figure()
    plt.imshow(Util.deletePeak(np.abs(image)), cmap='gray', aspect='auto')
    plt.xlabel("Range [sample]")
    plt.ylabel("Azimut [sample]")
    plt.title("Focused image by the MEA algorithm")
    plt.colorbar()


if __name__ == "__main__":
    matInit = scipy.io.loadmat('Data/simuData.mat')
    # matInit = scipy.io.loadmat('Data/realData.mat')
    matInit = matInit['imag_f']

    PEType = 1
    Noise = 100

    matInitBruit = Util.addNoise(matInit, SNR=Noise)
    matDefocus = Util.defocus(matInitBruit, PEType)

    plt.figure()
    plt.imshow(Util.deletePeak(np.abs(matInitBruit)), cmap='gray', aspect='auto')
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

    MEA(matDefocus)

    print(f"Initial image entropy : {Util.entropyCalculation(matInitBruit)}")

    print(f"Defocused image entropy : {Util.entropyCalculation(matDefocus)}")

    plt.show()


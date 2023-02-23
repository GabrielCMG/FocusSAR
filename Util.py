import numpy as np
import matplotlib.pyplot as plt


def phaseCorrection(G, focusFiltre=None):
    """
    Function that reconstructs the complex image from its Fourier transform in azimuth and a phase vector.

    :param G: Fourier transform of the image to be reconstructed
    :param focusFiltre: Phase that will be applied for the reconstruction (the phase can be the identity if it is None)
    :return: Reconstructed image
    """
    M, N = G.shape

    if focusFiltre is None:
        focusFiltre = np.zeros((M, 1))

    g = np.fft.ifft(G*np.exp(1j*focusFiltre).reshape(-1, 1), axis=0)

    return g


def entropyCalculation(image):
    """
    Function calculating the entropy of a complex image.

    :param image: A complex image whose entropy is calculated
    :return: Entropy of the image
    """
    imageCarre = image * np.conj(image)  # Calcul de la norme au carré de l'image
    S = np.sum(np.sum(imageCarre, axis=1), axis=0)  # Calcul du facteur de normalisation

    P = imageCarre/S * np.log(imageCarre/S+1e-12)

    entropie = np.real(- np.sum(np.sum(P, axis=1), axis=0))  # Calcul de l'entropie

    return entropie


def defocus(image, i):
    """
    Function that defocuses an image. Several types of phase error are possible for defocusing.

    :param image: Image to be defocused
    :param i: Type of phase error
    :return: Defocused image
    """
    M, N = image.shape

    x = np.linspace(0, M - 1, M) / M
    x0 = x - 1/2

    if i == 1:
        # Sinusoïdal symétrique pi
        focusError = 120*np.sin(np.pi*x0)
    elif i == 2:
        # Polynomial non symétrique
        focusError = (9 * x0 ** 2 - 12 * x0 ** 3 - 6 * x0 ** 5) * 30
    elif i == 3:
        # Polynomial symétrique
        focusError = (3 * x0 ** 2 + 10 * x0 ** 3) * 60

    plt.figure()
    plt.plot(focusError)
    plt.title(f"Phase error Type {i}")
    plt.xlabel("Azimuth [sample]")
    plt.ylabel("Phase error [rad]")

    image = np.fft.ifft(np.fft.fft(image, axis=0)*np.exp(1j*focusError).reshape(-1, 1), axis=0)

    return image


def addNoise(image, SNR, powType="dB"):
    """
    Function that add complex noise to an image.

    :param image: Image on which noise is added
    :param SNR: Signal to noise ratio of the noised image
    :param powType: Type of SNR (linear or dB)
    :return: Noised image
    """
    M, N = image.shape

    SNR_lin = SNR

    if powType == "dB":
        SNR_lin = np.power(10, SNR/10)

    signalPower = np.sum(np.sum(np.abs(image)**2, axis=1), axis=0) /M /N
    bruitPower = signalPower / SNR_lin

    bruit = np.sqrt(bruitPower/2) * (np.random.randn(M, N) + 1j*np.random.randn(M, N))

    imageBruit = image + bruit

    return imageBruit


def deletePeak(image):
    """
    Function to remove excessive peaks in the images. Only used for displaying images.

    :param image: Image from which peaks are to be removed
    :return: Image without the peaks
    """
    image[np.where(np.abs(image) > np.quantile(np.abs(image), 0.99))] = np.quantile(np.abs(image), 0.99)
    return image


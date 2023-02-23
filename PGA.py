import matplotlib.pyplot as plt
import numpy as np
import scipy
import Util


def centringPoints(image):
    """
    Image that centres the important points in an image.

    :param image: Image with important points to be centred
    :return: Image with important points centred
    """
    M, N = image.shape
    imageCentre = np.zeros((M, N), dtype=complex)

    indiceMax = np.argmax(image, axis=0)
    indiceCentre = M//2

    for i in range(N):
        imageCentre[:, i] = np.roll(image[:, i], indiceCentre-indiceMax[i])

    return imageCentre


def windowingImage(centeredImage):
    """
    Function applying a window to an image to isolate the points centred on the image by the function centringPoints().

    :param centeredImage: Image with important points centred
    :return: Windowed image
    """
    M, N = centeredImage.shape
    indiceCentre = M//2

    imageCentreInt = np.real(centeredImage * np.conj(centeredImage))

    Sx = np.sum(imageCentreInt, axis=1)
    Sx_dB = 20*np.log10(np.abs(Sx))
    cutoff = np.max(Sx_dB)-10
    WinBool = Sx_dB >= cutoff

    W = np.sum(WinBool)
    W = 1.5*W

    x = np.linspace(0, M-1, M)

    window = np.logical_and(x > (indiceCentre-W/2), x < (indiceCentre+W/2))

    window = np.kron(window.reshape(-1, 1), np.ones((1, N)))

    imageWindow = centeredImage * window

    return imageWindow


def dGn_calcul(imageWindowed, version="v1"):
    """
    Function that tries to calculate the gradient of an image across the azimuths

    :param imageWindowed: Image for which the gradient is calculated
    :param version: Type of gradient calculation
    :return: Gradient of the image
    """
    Gn = np.fft.ifft(imageWindowed, axis=0)

    M, N = imageWindowed.shape
    indiceCentre = M//2

    x = np.linspace(0, M - 1, M)

    dGn = np.zeros((M, N))

    if version == "v1":
        dGn = np.fft.ifft(1j * np.kron((x - indiceCentre).reshape(-1, 1), np.ones((1, N))) * imageWindowed, axis=0)
    elif version == "v2":
        dGn = np.gradient(Gn, axis=0)
    elif version == "v3":
        dGn[:-1, :] = np.diff(Gn, 1, 0)
        dGn[-1, :] = dGn[-2, :]
    elif version == "v4":
        Nx = N
        Ny = M
        dx = 0.1
        dy = 0.1
        x = np.arange(0, (Nx - 1) * dx, dx)
        y = np.arange(0, (Ny - 1) * dx, dy)
        Nyq_kx = 1 / (2 * dx)
        Nyq_ky = 1 / (2 * dy)
        dkx = 1 / (Nx * dx)
        dky = 1 / (Ny * dy)
        kx = np.arange(-Nyq_kx, Nyq_kx, dkx)
        ky = np.arange(-Nyq_ky, Nyq_ky, dky)
        data_wavenumberdomain = np.fft.fft2(Gn)
        [KX, KY] = np.meshgrid(np.fft.ifftshift(kx), np.fft.ifftshift(ky))
        data_wavenumberdomain_differentiated = 2j * np.pi * KY * data_wavenumberdomain
        dGn = np.fft.ifft2(data_wavenumberdomain_differentiated)

    return dGn


def phaseGradientEstimation(imageWindowed):
    """
    Function that estimate the gradient of the phase error of a SAR image and then integrate it.

    The algorithm used here is based on the article Phase Gradient Autofocusing Technique (PGA).

    :param imageWindowed: Windowed image
    :return: Phase error in the initial SAR image
    """
    M, N = imageWindowed.shape

    dt = 3/900

    Gn = np.fft.ifft(imageWindowed, axis=0)
    dGn = np.gradient(Gn, axis=0)*M

    print(np.sum(np.sum(Gn, axis=1), axis=0))
    print(np.sum(np.sum(dGn, axis=1), axis=0))

    # dGn = dGn_calcul(imageCentreFenetre, "v4")
    num = np.sum(np.imag(np.conj(Gn)*dGn), axis=1)
    denom = np.sum(np.conj(Gn)*Gn, axis=1)

    print(dGn.shape)

    dPhi = num/denom

    print(dPhi.shape)

    phi = np.zeros((M,), dtype=complex)

    Fe = 1/dt
    df = Fe/M

    for i in range(M):
        phi[i] = np.sum(dPhi[0:i])*df

    # coefLin = np.polyfit(np.linspace(0, M-1, M), phi, 1)
    # phi = phi - np.polyval(coefLin, np.linspace(0, M-1, M))

    print(phi.shape)

    return phi


def PGA(image, NIteration):
    """
    Function that performs Phase Gradient Autofocus on a SAR image.

    :param image: SAR image that need to be focused
    :param NIteration: Number of iterations the algorithm will perform
    :return: Focused image
    """
    M, N = image.shape
    phi = np.zeros((M,), dtype=complex)

    for _ in range(NIteration):
        imageCentre = centringPoints(np.fft.fft(image, axis=0))
        imageFenetre = windowingImage(imageCentre)
        phi = phaseGradientEstimation(imageFenetre)

        image = phaseCorrectionPGA(image, -phi)

    plt.figure()
    plt.imshow(Util.deletePeak(np.abs(image)), cmap='gray', aspect='auto')
    plt.xlabel("Range [sample]")
    plt.ylabel("Azimut [sample]")
    plt.title("Focused image by the PGA algorithm")
    plt.colorbar()

    return image, -phi


def phaseCorrectionPGA(image, focusFiltre=None):
    """
    Function that reconstructs the complex image from a defocused image and a phase vector.

    :param image: Image to be reconstructed
    :param focusFiltre: Phase that will be applied for the reconstruction (the phase can be zero if it is None)
    :return: Reconstructed image
    """
    M, N = image.shape

    if focusFiltre is None:
        focusFiltre = np.ones((M, 1))

    g = np.fft.ifft(np.fft.fft(image, axis=0)*np.exp(1j*np.fft.fftshift(focusFiltre)).reshape(-1, 1), axis=0)

    return g


if __name__ == "__main__":
    matInit = scipy.io.loadmat('Data/simuData.mat')
    matInit = scipy.io.loadmat('Data/realData.mat')
    matInit = matInit['imag_f']

    PEType = 2
    Noise = 100

    matInitBruit = Util.addNoise(matInit, SNR=Noise)
    matDefocus = Util.defocus(matInitBruit, PEType)

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

    matRec = PGA(matDefocus, 20)

    print(f"Initial image entropy : {Util.entropyCalculation(matInit)}")
    print(f"Defocused image entropy : {Util.entropyCalculation(matDefocus)}")

    plt.show()

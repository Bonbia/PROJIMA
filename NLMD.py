import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

im=imread('img/pyramide.tif')


def gaussian_noise_image(image, sigma):
    """
    Add Gaussian noise to an image.

    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - sigma: Standard deviation of the Gaussian noise.

    Returns:
    - noisy_image: 2D numpy array of the noisy image.
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are valid
    return noisy_image

def salt_and_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    """
    Add salt and pepper noise to an image.

    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - amount: Proportion of image pixels to replace with noise.
    - salt_vs_pepper: Proportion of salt vs. pepper noise.

    Returns:
    - noisy_image: 2D numpy array of the noisy image.
    """
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))

    # Add salt (white) noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords] = 255

    # Add pepper (black) noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords] = 0

    return noisy_image


def nlm_denoising(image, patch_size=3, search_window=7, h=10.0):
    padded_image = np.pad(image, patch_size // 2, mode='reflect')
    denoised_image = np.zeros_like(image)
    half_patch = patch_size // 2
    half_window = search_window // 2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            i1,j1=i+patch_size,j+patch_size
            center=(i+half_patch,j+half_patch)
            #A modifier pour faire attention aux bords
            if i1>padded_image.shape[0] or j1>padded_image.shape[1]:
                i1,j1=padded_image.shape[0],padded_image.shape[1]
            patch1= padded_image[i:i1, j:j1]
            weights = []
            patches = []
            
    return denoised_image




gnimg=gaussian_noise_image(im, sigma=20)
spnimg=salt_and_pepper_noise(im, amount=0.05, salt_vs_pepper=0.5)
nlm_gnimg=nlm_denoising(gnimg, patch_size=3, search_window=7, h=10.0)
nlm_spnimg=nlm_denoising(spnimg, patch_size=3   , search_window=7, h=10.0)

plt.figure("Image originale")
plt.imshow(im, cmap='gray')
plt.figure("Image bruitée par du bruit gaussien")
plt.imshow(gnimg, cmap='gray')
plt.figure("Image bruitée par du bruit sel et poivre")
plt.imshow(spnimg, cmap='gray')
plt.figure("Image débruitée par NLMD du bruit gaussien")
plt.imshow(nlm_gnimg, cmap='gray')
plt.figure("Image débruitée par NLMD du bruit sel et poivre")
plt.imshow(nlm_spnimg, cmap='gray')
plt.show()

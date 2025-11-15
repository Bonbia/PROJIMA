#%% Imports
import numpy as np
from skimage.io import imread
import time
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage
from skimage import io as skio
import IPython
from skimage.transform import rescale
from skimage.io import imread

#%%

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

#Fonction de bruit gaussien cf tp 
def noisegauss(im,br):
    """ Cette fonction ajoute un bruit blanc gaussier d'ecart type br
       a l'image im et renvoie le resultat"""
    imt=np.float32(im.copy())
    sh=imt.shape
    bruit=br*np.random.randn(*sh)
    imt=imt+bruit
    return imt

#Fonction affichage
def viewimage(im, normalize=True,z=1,order=0,titre='',displayfilename=False):
    imin=im.copy().astype(np.float32)
    imin = rescale(imin, z, order=order)
    if normalize:
        imin-=imin.min()
        if imin.max()>0:
            imin/=imin.max()
    else:
        imin=imin.clip(0,255)/255 
    imin=(imin*255).astype(np.uint8)
    filename=tempfile.mktemp(titre+'.png')
    if displayfilename:
        print (filename)
    plt.imsave(filename, imin, cmap='gray')
    IPython.display.display(IPython.display.Image(filename))




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

#%%

#Pixelwise implementation of NLMD
def nlm_naif_piw(img,patch_size,search_window,h,sigma):
    #Sigma est le "standard deviation" du bruit
    denoised_img=np.zeros(img.shape)
    half_patch=patch_size//2
    half_window=search_window//2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i1,j1=i-half_patch,j-half_patch
            i2,j2=i+half_patch,j+half_patch
            #I et J restent nos centres de patchs et on regarde les patchs autour d'eux
            #On tronque les patchs si on est proche des bords
            #Probleme pour le calcul des distances
            if i1<0 :
                i1=0
            if j1<0 :
                j1=0
            if i2>img.shape[0] :
                i2=img.shape[0]
            if j2>img.shape[1] :
                j2=img.shape[1]
                        
            #Patch de référence
            patch1= img[i1:i2, j1:j2]
            weights = []
            i_start=max(0,i-half_window)
            i_end=min(img.shape[0],i+half_window)
            j_start=max(0,j-half_window)    
            j_end=min(img.shape[1],j+half_window)
            weighted_sum = 0.0
            weights_sum = 0.0
            for k in range(i_start, i_end):
                for l in range(j_start, j_end):
                    k1, l1 = k - half_patch, l - half_patch
                    k2, l2 = k + half_patch, l + half_patch
                    # Gestion des bords (sans padding)
                    k1, l1 = max(k1, 0), max(l1, 0)
                    k2, l2 = min(k2, img.shape[0]), min(l2, img.shape[1])
                    patch2 = img[k1:k2, l1:l2]
                    dist = np.sum((patch1 - patch2)**2)
                    dist = max(dist - 2*(sigma**2), 0)
                    weight = np.exp(-dist / (h*h))
                    weighted_sum += weight * img[k, l]
                    weights_sum += weight

            # Évite la division par zéro
            if weights_sum > 0:
                denoised_img[i, j] = weighted_sum / weights_sum
            else:
                denoised_img[i, j] = img[i, j]

    return denoised_img
            
            
def nlm_naif2_piw(img, patch_size, search_window, h, sigma):
    denoised_img = np.zeros(img.shape, dtype=np.float32)
    half_patch = patch_size // 2
    half_window = search_window // 2

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Patch de référence (avec gestion des bords)
            i1, j1 = max(i - half_patch, 0), max(j - half_patch, 0)
            i2, j2 = min(i + half_patch, img.shape[0] - 1), min(j + half_patch, img.shape[1] - 1)
            patch1 = img[i1:i2+1, j1:j2+1]

            # Fenêtre de recherche
            i_start, i_end = max(i - half_window, 0), min(i + half_window, img.shape[0] - 1)
            j_start, j_end = max(j - half_window, 0), min(j + half_window, img.shape[1] - 1)

            weighted_sum = 0.0
            weights_sum = 0.0
            for k in range(i_start, i_end + 1):
                for l in range(j_start, j_end + 1):
                    # Patch voisin (avec gestion des bords)
                    k1, l1 = max(k - half_patch, 0), max(l - half_patch, 0)
                    k2, l2 = min(k + half_patch, img.shape[0] - 1), min(l + half_patch, img.shape[1] - 1)
                    patch2 = img[k1:k2+1, l1:l2+1]
                    # Taille du patch de référence (patch1)
                    h1, w1 = patch1.shape
                    # Taille du patch voisin (patch2)
                    h2, w2 = patch2.shape
                    # Taille minimale pour normaliser la distance
                    min_size = min(h1, h2) * min(w1, w2)

                    # Calcul de la distance normalisée
                    diff = patch1[:min(h1, h2), :min(w1, w2)] - patch2[:min(h1, h2), :min(w1, w2)]
                    dist = np.sum(diff**2) / min_size  # Normalisation par la taille minimale
                    dist = max(dist - 2*(sigma**2), 0)
                    weight = np.exp(-dist / (h*h))
                    dist = max(dist - 2*(sigma**2), 0)
                    weight = np.exp(-dist / (h*h))

                    weighted_sum += weight * img[k, l]
                    weights_sum += weight

            # Évite la division par zéro
            if weights_sum > 0:
                denoised_img[i, j] = weighted_sum / weights_sum
            else:
                denoised_img[i, j] = img[i, j]

    return np.clip(denoised_img, 0, 255).astype(np.uint8)


def nlm_patchwise(img, patch_size=7, search_window=21, h=10, sigma=15):
    """
    Implémentation patchwise du Non-Local Means.

    Args:
        img: Image en niveaux de gris (numpy array).
        patch_size: Taille des patches (doit être impair).
        search_window: Taille de la fenêtre de recherche (doit être impair).
        h: Paramètre de filtrage (ex: h = 0.4 * sigma).
        sigma: Écart-type du bruit.

    Returns:
        Image débruitée (numpy array).
    """
    # Initialisation
    half_patch = patch_size // 2
    half_search = search_window // 2
    accumulated_weights = np.zeros_like(img, dtype=np.float32)
    accumulated_values = np.zeros_like(img, dtype=np.float32)

    # Parcourir chaque pixel central d'un patch
    for i in range(half_patch, img.shape[0] - half_patch):
        for j in range(half_patch, img.shape[1] - half_patch):
            # Extraire le patch de référence
            patch = img[i-half_patch:i+half_patch+1, j-half_patch:j+half_patch+1]

            # Fenêtre de recherche
            i_start = max(i - half_search, half_patch)
            i_end = min(i + half_search, img.shape[0] - half_patch)
            j_start = max(j - half_search, half_patch)
            j_end = min(j + half_search, img.shape[1] - half_patch)

            # Initialisation pour le patch débruité
            weighted_patch_sum = np.zeros_like(patch, dtype=np.float32)
            weights_sum_patch = 0.0

            # Parcourir les patches voisins
            for k in range(i_start, i_end + 1):
                for l in range(j_start, j_end + 1):
                    k1, l1 = k - half_patch, l - half_patch
                    k2, l2 = k + half_patch, l + half_patch
                    neighbor_patch = img[k1:k2+1, l1:l2+1]

                    # Taille minimale pour normaliser la distance
                    h1, w1 = patch.shape
                    h2, w2 = neighbor_patch.shape
                    min_h, min_w = min(h1, h2), min(w1, w2)
                    min_size = min_h * min_w

                    # Calcul de la distance normalisée
                    diff = patch[:min_h, :min_w] - neighbor_patch[:min_h, :min_w]
                    dist = np.sum(diff**2) / min_size
                    weight = np.exp(-max(dist - 2*(sigma**2), 0) / (h**2))

                    weighted_patch_sum[:min_h, :min_w] += weight * neighbor_patch[:min_h, :min_w]
                    weights_sum_patch += weight
            # Estimation du patch débruité
            if weights_sum_patch > 0:
                denoised_patch = weighted_patch_sum / weights_sum_patch
            else:
                denoised_patch = patch  # Garde le patch original si aucun poids valide

            # Agrégation des estimations pour chaque pixel du patch
            for di in range(patch_size):
                for dj in range(patch_size):
                    pixel_i = i - half_patch + di
                    pixel_j = j - half_patch + dj
                    accumulated_weights[pixel_i, pixel_j] += 1.0
                    accumulated_values[pixel_i, pixel_j] += denoised_patch[di, dj]

    # Normalisation finale
    denoised_img = np.divide(
        accumulated_values,
        accumulated_weights,
        out=np.zeros_like(accumulated_values),
        where=accumulated_weights != 0
    )

    return np.clip(denoised_img, 0, 255).astype(np.uint8)

            
            
def nlm_denoising(image, patch_size=3, search_window=21, h=10.0):
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


#%% Test des fonctions
gnimg=noisegauss(im,20)
viewimage(gnimg,normalize=False)

#spnimg=salt_and_pepper_noise(im, amount=0.05, salt_vs_pepper=0.5)
start = time.time()
nlm_gnimg=nlm_naif2_piw(gnimg, patch_size=3, search_window=7, h=10.0, sigma=20)
nlm_gnimg2=nlm_patchwise(gnimg, patch_size=3, search_window=7, h=10.0, sigma=20)
endg = time.time()
#nlm_spnimg=nlm_naif(spnimg, patch_size=3, search_window=7, h=10.0, sigma=20)
endsp = time.time()

plt.figure("Image originale")
plt.imshow(im, cmap='gray')
plt.figure("Image bruitée par du bruit gaussien")
plt.imshow(gnimg, cmap='gray')
plt.figure("Image débruitée par NLMD en pixelwise du bruit gaussien")
plt.imshow(nlm_gnimg, cmap='gray')
plt.figure("Image débruitée par NLMD en patchwise du bruit gaussien")
plt.imshow(nlm_gnimg2, cmap='gray')

plt.show()

#%% Affichage comparatif avec meme echelles de gris 
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes.ravel()
ax[0].imshow(im, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Image originale')
ax[1].imshow(gnimg, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Image bruitée (gaussien)')
ax[2].imshow(nlm_gnimg, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('NLMD pixelwise')
ax[3].imshow(nlm_gnimg2, cmap='gray', vmin=0, vmax=255)
ax[3].set_title('NLMD patchwise')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

#%% Autres Tests

nlm_gnimg=nlm_naif2_piw(gnimg, patch_size=3, search_window=7, h=8.0, sigma=20)
nlm_gnimg2=nlm_naif2_piw(gnimg, patch_size=5, search_window=7, h=8.0, sigma=20)
nlm_gnimg3=nlm_naif2_piw(gnimg, patch_size=7, search_window=7, h=8.0, sigma=20)
nlm_gnimg5=nlm_naif2_piw(gnimg, patch_size=3, search_window=9, h=8.0, sigma=20)
nlm_gnimg6=nlm_naif2_piw(gnimg, patch_size=5, search_window=9, h=8.0, sigma=20)







# %%

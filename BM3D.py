#%%

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
from scipy.spatial import distance
from scipy.fft import dct, idct, dctn, idctn

#%%
### Parametres BM3D Etape 1

lhard=20        #Lambda hard thresholding
br=20      #Ecart type du bruit
sigm=br     #ecart type du bruit
khard=8     #Taille des patchs
PatchSize=khard
Windowsearch=16  #Taille de la fenetre de recherche
Nhard=16
MaxMatch=Nhard    #Nombre max de patchs similaires
lamb2d = 2.0

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




### 

def sigma(patch,lhard,sigm):
    '''
    Seuillage dur (Hard Tresholding) sur un patch avec le seul lambda*sigma
    '''
    #Le patch (ressorti)est un vecteur en 1D
    for i in range(patch.shape[0]):
        if abs(patch[i])<lhard*sigm:
            patch[i]=0
    return patch

def distpatch(patch1,patch2,khard=8):
    '''
    Calcul distance entre deux patchs
    '''

    p1 = patch1.astype(np.float32)
    p2 = patch2.astype(np.float32)

    diff = p1 - p2
    dist = np.sqrt(np.sum(diff * diff)) / (khard ** 2)  # Normalization
    return dist
    

#Max Similar Patches = Nhard ?
Nhard=16


def grouping(im,patch_size=8,search_window=16,max_similar_patches=16,threshard=250):
    '''
    Etape de grouping des patchs similaires dans une image bruitée 'im'.
    Renvoie un bloc 3D de patches similaires.
    '''
    H, W = im.shape
    center_i = H // 2
    center_j = W // 2
    center_i = np.clip(center_i, patch_size // 2, H - patch_size // 2 - 1)
    center_j = np.clip(center_j, patch_size // 2, W - patch_size // 2 - 1)
    for i in range(patch_size//2, im.shape[0] - patch_size//2):
        for j in range(patch_size//2, im.shape[1] - patch_size//2):
            # Extraire le patch de référence
            ref_patch = im[i-patch_size//2:i+patch_size//2, j-patch_size//2:j+patch_size//2]

            # Définir la fenêtre de recherche
            window = im[max(0, i-search_window//2):min(im.shape[0], i+search_window//2),
                        max(0, j-search_window//2):min(im.shape[1], j+search_window//2)]

            # Liste pour stocker les patches similaires
            similar_patches = []
            distmin=float('inf')
            # Pour chaque patch dans la fenêtre de recherche
            for x in range(patch_size//2, window.shape[0] - patch_size//2):
                for y in range(patch_size//2, window.shape[1] - patch_size//2):
                    current_patch = window[x-patch_size//2:x+patch_size//2, y-patch_size//2:y+patch_size//2]
                    dist = distpatch(ref_patch.flatten(), current_patch.flatten(),khard)
                    if dist < distmin:
                        distmin=dist
                    if dist < threshard :
                        similar_patches.append((dist, current_patch,(i, j)))
            if len(similar_patches) == 0:
                similar_patches = [(0.0, ref_patch.copy(), (center_i, center_j))]
            # Trier les patches par distance et garder les 'max_similar_patches' plus proches
            similar_patches.sort(key=lambda x: x[0])
            grouped_patches = similar_patches[:max_similar_patches]
            coords  = [coord for (d, p, coord) in grouped_patches]
            # 'grouped_patches' est maintenant un bloc 3D de patches similaires
            block_3D = np.stack(grouped_patches, axis=2)
    return block_3D,coords

def bettergrouping(im, center_i=None, center_j=None,
             khard=khard, search_window=Windowsearch,
             Nhard=Nhard, threshard=250):
    """
    Find and group patches from the image *im*, return a 3D array of patches.

    Note that the distance computing is chosen from original paper rather than the analysis one.

    - im: 2D noised image
    - (center_i, center_j): Center of reference patch. If center = None, the center of the image will be used
    - patch_size: Size of patch (default = 8)
    - search_window: Size of the search window (in pixels)
    - max_similar_patches: Nhard
    - threshard: Distance threshold for accepting a patch
    """
    H, W = im.shape
    half_p = khard // 2
    half_w = search_window // 2

    # Select the center of the reference patch.
    if center_i is None:
        center_i = H // 2
    if center_j is None:
        center_j = W // 2
    center_i = np.clip(center_i, half_p, H - half_p - 1)
    center_j = np.clip(center_j, half_p, W - half_p - 1)

    # Build reference patch.
    ref_patch = im[center_i - half_p:center_i + half_p,
                   center_j - half_p:center_j + half_p]

    # Range of search window.
    i_min = max(center_i - half_w, half_p)
    i_max = min(center_i + half_w, H - half_p)
    j_min = max(center_j - half_w, half_p)
    j_max = min(center_j + half_w, W - half_p)

    similar_patches = []        # (dist, patch, (i, j))

    # Patch center.
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            current_patch = im[i - half_p:i + half_p,
                               j - half_p:j + half_p]
            dist = distpatch(ref_patch.flatten(),
                             current_patch.flatten(),
                             khard=khard)
            if dist < threshard:
                similar_patches.append((dist, current_patch.copy(), (i, j)))

    # Use *ref_patch* if no similar patches are found. 
    if len(similar_patches) == 0:
        similar_patches = [(0.0, ref_patch.copy(), (center_i, center_j))] # The only one.


    # 5. Sorting. Keep the smallest *max_similar_patches* patches.
    similar_patches.sort(key=lambda x: x[0])
    selected = similar_patches[:Nhard]

    patches = [p for (d, p, coord) in selected]
    coords  = [coord for (d, p, coord) in selected]

    block_3D = np.stack(patches, axis=2)

    return block_3D, coords



#Transfo lineaire en 3D

#Voir pb si prends pas bien les 3DBlock direct le faire patch par patch
#Voir si autre transfo 3D

def ondelet_3D(block_3D):
    '''
    Transfo ondelettes 3D sur un block 3D de patchs.
    '''
    block_3D = np.asarray(block_3D, dtype=np.float32)
    coeffs = dctn(block_3D, axes=(0, 1, 2), type=2, norm='ortho')
    return coeffs
def invondelet_3D(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float32)
    block_rec = idctn(coeffs, axes=(0, 1, 2), type=2, norm='ortho')    
    return block_rec

def hard_threshold(coeffs, l=lamb2d, sigm =sigma):
    thresh = l * sigm
    coeffs = coeffs.copy()
    coeffs[np.abs(coeffs) < thresh] = 0
    return coeffs

def collaborative_filtering(block_3D):
    Dtransfo=ondelet_3D(block_3D) #Transfo ondelette sur le block 3D
    Shrinked= hard_threshold(Dtransfo,lhard,sigm)    #Seuillage dur du block 
    Dinversed= invondelet_3D(Shrinked)  #Transfo inverse ondelette
    return Dinversed


def bm3d_step1(noisy, patch_size, search_window,
           max_similar_patches, step=3,lhard=2.7, sigma_noise=25):
    noisy = noisy.astype(np.float32)
    H, W = noisy.shape
    half_p = patch_size // 2
    
    v = np.zeros_like(noisy, dtype=np.float32)
    w = np.zeros_like(noisy, dtype=np.float32)

    for i in range(half_p, H - half_p, step):
        for j in range(half_p, W - half_p, step):
            # Grouping for (i, j).
            block_3D, coords = bettergrouping(
                noisy
            )

            block_filtered = collaborative_filtering(block_3D)
            num_patches = block_filtered.shape[2]

            weight = 1.0 # Can upgrade to more reasonable weights.

            for n in range(num_patches):
                pi, pj = coords[n] 
                top  = pi - half_p
                left = pj - half_p
                patch = block_filtered[:, :, n]
                v[top:top+patch_size, left:left+patch_size] += weight * patch
                w[top:top+patch_size, left:left+patch_size] += weight
    eps = 1e-12
    u_hat = v / (w + eps)

    # Special treatment for places where w == 0
    mask_zero = (w < eps)
    u_hat[mask_zero] = noisy[mask_zero]

    return u_hat

#%%#Test des fonctions
im=imread('img/pyramide.tif')
imb=noisegauss(im,br)

# viewimage(imb,titre='Image bruitée',displayfilename=True)
# block_3D,coords=grouping(imb,patch_size=khard,search_window=16,max_similar_patches=Nhard,threshard=250)
# print("Block 3D shape:", block_3D.shape)

# %%

testbmstep1=bm3d_step1(imb,patch_size=khard,search_window=21,max_similar_patches=Nhard,step=3,lhard=lhard,sigma_noise=sigm)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(im, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(imb, cmap='gray', vmin=0, vmax=255)
axes[1].set_title("Noisy")
axes[1].axis('off')

axes[2].imshow(testbmstep1, cmap='gray', vmin=0, vmax=255)
axes[2].set_title("BM3D step=3 window=21")
axes[2].axis('off')

plt.tight_layout()
plt.show()
# %%

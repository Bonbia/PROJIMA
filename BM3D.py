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


### 

im=imread('img/pyramide.tif')

br=20

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
l=20
sigm=br


def sigma(patch,l,sigm):
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            if abs(patch[i,j])<l*sigm:
                patch[i,j]=0
    return patch

def distpatch(patch1,patch2,khard=2):
    dist=distance.euclidean(sigma(patch1),sigma(patch2))/(khard**2)
    return dist
    

def grouping(patch_size=8,search_window=16,max_similar_patches=16,threshard=250):
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
                        if len(similar_patches) < max_similar_patches:
                            similar_patches.append((dist, current_patch))
                        else: 
                            if dist < distmin:
                                similar_patches.sort(key=lambda x: x[0], reverse=True)
                                similar_patches[-1] = (dist, current_patch)

            # Trier les patches par distance et garder les 'max_similar_patches' plus proches
            similar_patches.sort(key=lambda x: x[0])
            grouped_patches = [patch for (dist, patch) in similar_patches[:max_similar_patches]]

            # 'grouped_patches' est maintenant un bloc 3D de patches similaires
            block_3D = np.stack(grouped_patches, axis=2)
    return block_3D


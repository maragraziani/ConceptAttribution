import cv2
import skimage.color
import skimage.feature

def haralick(image, mask=None, mtype=None, verbose=False):
    """
    Compute Haralick's descriptors of the Grey-Level Co-occurences matrix
    """
    # we use standard parameters in the computation of the glcm
    glcm = skimage.feature.greycomatrix(
            skimage.img_as_ubyte(skimage.color.rgb2gray(image)),
            [1],
            [0],
            symmetric=True,
            normed=True)
    return skimage.feature.greycoprops(glcm, mtype)

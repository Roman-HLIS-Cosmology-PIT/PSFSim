from psfsim.polychrom import PolychromaticPSF
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def center_crop(img, crop_h=500, crop_w = 500):
    h, w = img.shape
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h+crop_h, start_w:start_w+crop_w]


scaNum = 18
wavelengths = np.array([1.40, 1.80, 100])
use_postage_stamp_size = 96
postage_stamp_size = 32

scaX = np.array([0, 10, -10, -10, 10], dtype=np.float64)
scaY = np.array([0, 10, 10, -10, -10], dtype=np.float64)

optical_psfs=[]
detector_images = []


for scax, scay in zip(scaX, scaY):
    poly_psf = PolychromaticPSF(scaNum, scax, scay, wavelengths)
    optical_psf = poly_psf.compute_poly_psf(optical_psf_only=True, use_postage_stamp_size=use_postage_stamp_size, postage_stamp_size=postage_stamp_size)
    cropped_optical_psf = center_crop(optical_psf)
    optical_psfs.append(cropped_optical_psf)

    detector_psf = poly_psf.compute_poly_psf(optical_psf_only=False, use_postage_stamp_size=use_postage_stamp_size, postage_stamp_size=postage_stamp_size)
    detector_images.append(detector_psf)






"""UNIVERSITY OF MANITOBA
COURSE: ASTR 3070
DATE: Fall 2025
INSTRUCTOR: Tyrone Woods
AUTHORS: Mark Pirgalin, Jade Yeung (attributed)
"""

import os
import numpy as np
import astroalign as align
from astropy.io import fits


def fetch_filenames(dir: str) -> list[str]:
    """Fetches a list of filenames from a directory
    Args:
        dir (str): path of the file directory
    Returns:
        list[str]: list of filenames
    """
    try:
        filenames = []  
        for item in os.scandir(dir):
            if item.is_file() and _check_truncated(item):
                filenames.append(item)                  
        print(f"{len(filenames)} files retrived from {dir}")
        return filenames
    except:
        print("Directory not found.")


def _check_truncated(file):
    
    # prevent corrupt header files
    hdr = fits.open(file)[0].header
    naxis1 = hdr.get('NAXIS1')
    naxis2 = hdr.get('NAXIS2')
    bitpix = hdr.get('BITPIX')
    
    expected_size = naxis1 * naxis2 * (bitpix/8)  # pixels * bytes
    actual_size = os.path.getsize(file)
    
    return actual_size >= expected_size   # the truncation issue only arises when the actual size is smaller than expected
    


def stack_master(files: list) -> np.array:
    """Stacks frames to create a master file. This does NOT align the frames when stacking.

    Args:
        files (list): list of filenames

    Returns:
        np.array: final image stack in the form of a 2D array
    """
    base = fits.open(files[0])
    data = base[0].data
    
    width = len(data[0])
    height = len(data)
    
    master_img = np.zeros((height, width))
    count = 0
    
    for file in files:
        count += 1
        imgs = fits.open(file)
        img_data = imgs[0].data
        master_img = master_img + img_data

    # average normalization
    master_img = master_img / count

    return master_img



def calibrate_lights(light: list[str], dark: np.array, flat_bias: np.array) -> list[np.array]:
    """Applies dark and bias/flat corrections to individual light frames. 

    Args:
        light (list[str]): list of light frames
        dark (np.array): master dark frame
        flat_bias (np.array): master flat/bias frame

    Returns:
        list[np.array]: list of corrected light frames
    """
    corr_HDU_list = []
    
    for file in light:
        light_imgs = fits.open(file)
        light_data = light_imgs[0].data
        light_corr = (light_data - dark) / flat_bias
        corr_HDU_list.append(light_corr)
        
    return corr_HDU_list



def align_images(lights: list[np.array], ref_image: np.array) -> np.array:
    """Aligns images from multiple nights. 
    
    Args:
        list_of_image_hdus (list[np.array]): list of calibrated lights from each night
        reference_image (np.array): image to use as reference for alignment

    Returns:
        np.array: aligned master stack array
    """
        
    master_image = np.zeros(ref_image.shape)
    count = 0
    
    for hdu_list in lights:
        
        for image in hdu_list:
            
            count+=1
            aligned_image, _ = align.register(image, ref_image)
            master_image += aligned_image
            
    return master_image/count, count
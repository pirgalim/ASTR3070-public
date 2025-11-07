"""UNIVERSITY OF MANITOBA
COURSE: ASTR 3070
DATE: Fall 2025
INSTRUCTOR: Tyrone Woods
AUTHORS: Mark Pirgalin, Jade Yeung (attributed)
"""

import os
import numpy as np
import astroalign as align
import pandas as pd
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



def make_header_data_lists(light: list[str], dark: np.array, flat_bias: np.array) -> list[np.array]:
    """Applies dark and bias/flat corrections to individual light frames 

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



def align_image(list_of_images1: np.array, list_of_images2: np.array, reference_image: np.array) -> np.array:
    """Aligns images from two nights. 
    - To use with one night, use the same list in both parameters and divide the returned array by 2
    - To use with multiple nights, use the returned array as one parameter and the next night of images as the second, then repeat as needed
    Args:
        list_of_images1 (np.array): images from first night
        list_of_images2 (np.array): images from second night
        reference_image (np.array): image to use as reference for alignment

    Returns:
        np.array: aligned master stack array
    """
    
    height=len(reference_image[:,0])
    width=len(reference_image[0,:])
    master_image=np.zeros((height, width))
    count=0

    image_list1=[]
    image_list2=[]

    for image in list_of_images1:
        count+=1
        aligned_image, _ = align.register(image, reference_image)
        master_image = master_image+aligned_image

    for image in list_of_images2:
        count+=1
        aligned_image, _ = align.register(image, reference_image)
        master_image = master_image+aligned_image

    return master_image/count, count



def create_table(pos_filename: str, pho_filename: str, sep_pos = '\s{2,}', sep_mag = '\s+') -> pd.DataFrame:
    """
    Create a table from .pos and .pho files

    Args:
        pos_filename (str): given .pos (coordinates) file
        pho_filename (str): given .pho (magnitudes) file
        sep_pos (str, optional): modify if the data is not separated correctly. Defaults to '\s{2,}'
        sep_mag (str, optional): modify if the data is not separated correctly. Defaults to '\s+'

    Returns:
        pd.DataFrame: Stetson table with Reference, RA, DEC, R mags, and B mags
    """

    # pos file needs to be cleaned
    df_pos = pd.read_csv(pos_filename, sep=sep_pos, comment="#", engine='python')   
    df_pos.columns = ['RA', 'DEC'] + ['' for i in range(3, df_pos.shape[1])] + ['Reference']
    
    # format negative values if applicable
    for col in df_pos.columns:
        try:
            df_pos[col] = df_pos[col].str.replace(" ", "").astype(float)    # convert negative values to floats
        except:
            print("Column formatting may have been unsuccessful, print the table to ensure data is displayed correctly.")
     
    # mag file
    df_mag = pd.read_csv(pho_filename, sep=sep_mag, comment="#", engine='python')
    
    # join tables
    stetson = pd.merge(df_pos[['Reference', 'RA', 'DEC']], df_mag[['Reference', 'R', 'B']], how='inner', on=['Reference'])
    
    return stetson
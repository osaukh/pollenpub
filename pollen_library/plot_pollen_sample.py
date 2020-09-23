#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:26:07 2020

@author: khoanam
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os

from collections import Counter
import cv2

#%% Helper functions
def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    try:
       return img[...,::-1]
    except:
       print('Error', path)        
       print('**********************************')
       
def pad_n_resize(im, desired_size):
    def create_blank(width, height, rgb_color=(0, 0, 0)):
        """Create new image(numpy array) filled with certain color in RGB"""
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)
    
        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = rgb_color
    
        return image

    square_size = max(im.shape)
    new_img = create_blank(square_size,square_size)
    left = (square_size - im.shape[1]) // 2
    top = (square_size - im.shape[0]) // 2
           
    new_img[top: top + im.shape[0], left:left+im.shape[1], :] = im
    new_img = cv2.resize(new_img, (desired_size,desired_size), interpolation = cv2.INTER_CUBIC)
    
    return new_img

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file
        self.embedding = None
        
    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 


#**********************************************************************************

def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


#%%  Load data and plot

img_dir = 'images_16_types'
metadata = load_metadata(img_dir)
targets = np.array([m.name for m in metadata])

classes = Counter(targets).keys()
classes = [*classes]

print(classes)


imgs_by_class = {}
for c in classes:
    file_idx = np.array([m.file for m in metadata if m.name == c])
    imgs_by_class[c] = file_idx

    
n_rows = 10
n_cols = 5
grid_size = n_rows* n_cols

for c in classes:
    imgs = []
    random_list = np.random.choice(len(imgs_by_class[c]), size = grid_size, replace=False)
    
    for i in range(grid_size):
        img = load_image(os.path.join(img_dir, c, imgs_by_class[c][random_list[i]]))
        img = pad_n_resize(img, 96)
            
        imgs.append(img)
        
    fig = plt.figure(figsize=(30, 30))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    
    
    for ax, im in zip(grid, imgs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    
    plt.suptitle(f'{c}', fontsize = 80)
    # plt.title(c, fontsize = 24)
    plt.show()
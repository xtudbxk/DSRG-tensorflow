import os
import glob
import shutil
import numpy as np
from scipy import io as scipy_io
from skimage import io as skimage_io

def rgb2label(label, colors=[], ignore_color=255):
    if len(colors) <= 0:
        colors = [(0, 0, 0), (128, 0, 0), (0, 128,0 ), (128, 128, 0),
                  (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                  (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                  (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                  (0, 64, 128)] # using palette for pascal voc
    rgb = ignore_color*np.ones(label.shape[0:2],dtype=np.uint8)
    for i,c in enumerate(colors):
        masks = label[:,:,0:3] == c
        mask = np.logical_and(masks[:,:,2],np.logical_and(masks[:,:,0],masks[:,:,1]))
        rgb[mask] = i
    return rgb.astype(np.uint8)

# paths
sbd_path = os.path.join("benchmark_RELEASE","dataset","cls")
pascal_path = os.path.join("VOCdevkit","VOC2012","SegmentationClass")
aug_path = os.path.join("VOCdevkit","VOC2012","SegmentationClassAug")

# create aug dataset path
if os.path.exists(aug_path) is not True:
    os.mkdir(aug_path)

# moving data from pascal to aug
pascal_filenames = glob.glob(os.path.join(pascal_path,"*.png"))
for filename_index,single_pascal_filename in enumerate(pascal_filenames):
    if filename_index % 500 == 0: print(f"processing pascal dataset - total:{len(pascal_filenames)},now:{filename_index},{filename_index/len(pascal_filenames)*100:.3}%")
    _,single_image_name = os.path.split(single_pascal_filename)
    single_image = skimage_io.imread(single_pascal_filename)
    if len(single_image.shape) >= 3 and single_image.shape[2] > 1:
        single_image = single_image[:,:,:3]
        single_image = rgb2label(single_image)
    skimage_io.imsave(os.path.join(aug_path,single_image_name),single_image)

# moving data from sbd to aug
sbd_filenames = glob.glob(os.path.join(sbd_path,"*.mat"))
for filename_index,single_sbd_filename in enumerate(sbd_filenames):
    if filename_index % 500 == 0: print(f"processing sbd dataset - total:{len(sbd_filenames)},now:{filename_index},{filename_index/len(sbd_filenames)*100:.3}%")
    single_id = os.path.split(single_sbd_filename)[1][:-4]
    sbd_data = scipy_io.loadmat(single_sbd_filename)
    skimage_io.imsave(os.path.join(aug_path,f"{single_id}.png"),sbd_data["GTcls"]["Segmentation"][0][0])

print("finished!")

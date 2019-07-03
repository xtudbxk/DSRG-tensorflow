import os
import glob
import shutil
from scipy import io as scipy_io
from skimage import io as skimage_io

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
    shutil.copy(single_pascal_filename,aug_path)

# moving data from sbd to aug
sbd_filenames = glob.glob(os.path.join(sbd_path,"*.mat"))
for filename_index,single_sbd_filename in enumerate(sbd_filenames):
    if filename_index % 500 == 0: print(f"processing sbd dataset - total:{len(sbd_filenames)},now:{filename_index},{filename_index/len(sbd_filenames)*100:.3}%")
    single_id = os.path.split(single_sbd_filename)[1][:-4]
    sbd_data = scipy_io.loadmat(single_sbd_filename)
    skimage_io.imsave(os.path.join(aug_path,f"{single_id}.png"),sbd_data["GTcls"]["Segmentation"][0][0])

print("finished!")

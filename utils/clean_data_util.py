import os
import shutil

subdirectories = ['0-50', '50-60', '60-70', '70-80','80-90','90-100']
current_loc = '/home/mohamed.imam/Downloads/Projects/CDTrans/'
def create_subdirectories(parent_dir):
    # Iterate over the subdirectory names and create each one
    for subdir_name in subdirectories:
        subdir_path = os.path.join(parent_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)


def copy_image_to_subdir(image_name, pred, parent_dir, data_dir, pred_class):
    if pred < 0.5:
        subdir_name = subdirectories[0]
    elif pred >= 0.5 and pred < 0.6:
        subdir_name = subdirectories[1]
    elif pred >= 0.6 and pred < 0.7:
        subdir_name = subdirectories[2]
    elif pred >= 0.7 and pred < 0.8:
        subdir_name = subdirectories[3]
    elif pred >= 0.8 and pred < 0.9:
        subdir_name = subdirectories[4]    
    else:
        subdir_name = subdirectories[5]
    
    subdir_path = os.path.join(parent_dir, subdir_name)
    
    image_path = os.path.join(data_dir, image_name)
    image_name=image_name.replace(".jpeg","_")+pred_class+".jpeg"
    dest_path = os.path.join(current_loc,subdir_path, image_name)
    shutil.copyfile(image_path, dest_path)    


# import torch
# import string
# # torch.random.seed()
# a = torch.tensor([[ 0.8548, -0.9946, -0.8302, -0.2046],
#         [ 0.1133,  0.7916, -0.0083, -0.5238],
#         [ 0.9271, -0.6097, -2.1425,  0.7769],
#         [ 0.0148,  2.2342, -1.0224,  1.1227]])
# argmax = torch.argmax(a).item()
# argmax_str = str(argmax)

# print(argmax_str)
# # print (b)
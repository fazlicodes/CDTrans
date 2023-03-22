import os
import shutil
import random
from tqdm import tqdm
import pandas as pd

# Set the path to the directory containing the subdirectories of classes
data_dir = "../test/sgada_data/"

# Set the path to the directory where you want to save the subset dataset
subset_dir = "../test/sample_coco/"

# Create the subset directory if it doesn't exist
if not os.path.exists(subset_dir):
    os.mkdir(subset_dir)
     
# Set the size of the subset dataset
subset_size = 100



cat = os.listdir(data_dir)

for i in cat:

    # Annotations
    ann = pd.DataFrame(columns=['path','label'])
    labels = ['person','bicycle','car']

    # Get the list of subdirectories (classes) in the data directory
    cat_dir = os.path.join(data_dir,i)

 
    cat_sub_dir = os.path.join(subset_dir,i)
    if not os.path.exists(cat_sub_dir):
        os.mkdir(cat_sub_dir)
    
    if  not os.path.isdir(cat_dir):
                print('not dir')
                continue
    classes = os.listdir(cat_dir)

    # Loop over each class subdirectory
    for cls in classes:
        
        # Create a subdirectory in the subset directory for this class
        cls_subset_dir = os.path.join(cat_sub_dir, cls)
        if not os.path.exists(cls_subset_dir):
            os.mkdir(cls_subset_dir)

        # Get a list of all the files in the class subdirectory
        cls_dir = os.path.join(cat_dir, cls)
        
        if  not os.path.isdir(cls_dir):
            print('not dir')
            continue
        print(cls_dir)
        files = os.listdir(cls_dir)

        # Choose a random subset of files from the class subdirectory
        subset = random.sample(files, min(subset_size, len(files)))

        # Copy the chosen files to the class's subset directory
        for file in tqdm(subset):
            file_name = os.path.basename(file)
            old_file_path = os.path.join(cls_dir, file_name)
            new_file_path = os.path.join(cls_subset_dir, file_name)
            # print(old_file_path)
            shutil.copy(old_file_path, new_file_path)
            ann = ann.append(pd.DataFrame([['{}/{}/{}'.format(i,cls,file), labels.index(cls)]],columns=['path','label']),ignore_index=True)

    # print(ann.tail(200))
    ann.to_csv(os.path.join(subset_dir, '{}.txt'.format(i)), header=None, index=None, sep=' ', mode='a')
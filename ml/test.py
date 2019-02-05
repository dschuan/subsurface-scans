import itertools
import os, os.path, shutil
import argparse
import os
import shutil
import tensorboard
print(tensorboard.__file__)



#
# space = [5,10,20,30]
# layer = 4
# output = []
# for layer in range(layers):
#     combi = list(itertools.combinations_with_replacement(space,layers))
#     output = output + combi
# print(output)
# print(len(output))


# folder_path = "logs/1channelgrid_norm_drop1/"
#
# images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#
# for image in images:
#     folder_name = image.split('.')[0]
#
#     new_path = os.path.join(folder_path, folder_name)
#     if not os.path.exists(new_path):
#         os.makedirs(new_path)
#
#     old_image_path = os.path.join(folder_path, image)
#     new_image_path = os.path.join(new_path, image)
#     shutil.move(old_image_path, new_image_path)




def move_files(abs_dirname):
    """Move files into subdirectories."""

    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]



    for f in files:
        print('looking at folder',f)

        subfolders = [f+'/train',f+'/test']

        for subfolder in subfolders:
            print('looking type',subfolder)
            experiments = [os.path.join(subfolder, x) for x in os.listdir(subfolder)]

            i = 0
            for experiment in experiments:
                print('looking experiment',experiment)
                # create new subdir if necessary

                subdir_name = subfolder + str(i)
                print('make dir',subdir_name)
                os.mkdir(subdir_name)

                # move file to current dir
                f_base = os.path.basename(experiment)
                print('moving file',f_base)
                print('to',os.path.join(subdir_name, f_base))
                shutil.move(experiment, os.path.join(subdir_name, f_base))

                i += 1

# move_files(os.path.abspath('./logs/1channelgrid_norm_drop1'))

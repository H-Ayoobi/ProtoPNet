import Augmentor
import os
import sys
import sys
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

datasets_root_dir = '/Users/hayoobi/PycharmProjects/ProtoArgNet/CUB_200_2011/cub200_cropped/'
dir = datasets_root_dir + 'train_cropped/'
target_dir = datasets_root_dir + 'train_cropped_augmented/'

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

progress_interval = 5
for j in range(len(folders)):
    fd = folders[j]
    tfd = target_folders[j]
    # rotation
    flag = True
    while(flag):
        try:
            p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
            p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
            p.flip_left_right(probability=0.5)
            for i in range(10):
                p.process()
            del p
            # skew
            p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
            p.skew(probability=1, magnitude=0.2)  # max 45 degrees
            p.flip_left_right(probability=0.5)
            for i in range(10):
                p.process()
            del p
            # shear
            p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
            p.shear(probability=1, max_shear_left=10, max_shear_right=10)
            p.flip_left_right(probability=0.5)
            for i in range(10):
                p.process()
            del p
            flag = False
        except:
            print(f"Failed at this directory: folder: {fd}, target folder: {tfd}")


    if (j + 1) % progress_interval == 0:
        progress = ((j + 1) / len(folders)) * 100
        sys.stdout.write(f"\rProgress: {int(progress)}%")
        sys.stdout.flush()
    # random_distortion
    #p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    #p.flip_left_right(probability=0.5)
    #for i in range(10):
    #    p.process()
    #del p

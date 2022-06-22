from copy import copy
from operator import length_hint
import shutil
import os
import tensorflow as tf
import pathlib

def copy_pictures(directory, worker_index, worker_count, single_classification_mode, debug_mode):
    print("Setting up training data...")
    classes = []
    copy_path = directory[0:directory.rindex('\\')] + "\{}".format(worker_index)

    #21.06 disabled -> TODO: needs to return false if TOTAL_CLIENTS increased
    # return if data is already split
    # if(check_if_already_split(copy_path, single_classification_mode)):
    #     return copy_path

    if(debug_mode):
        return copy_path

    #clear
    if(os.path.isdir(copy_path)):
        shutil.rmtree(copy_path)

    # download pictures if not existing yet
    if(not os.path.isdir(directory)):
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
        data_dir = pathlib.Path(data_dir)
        image_count = len(list(data_dir.glob('*/*.jpg')))
        print("images found: ", image_count)

    # iterate over files in directory
    for class_name in os.listdir(directory):
        f = os.path.join(directory, class_name)
        # checking if it is a file
        if not os.path.isfile(f):
            classes.append(class_name)

    if(single_classification_mode):
        #assign every worker one class
        if(worker_count > len(classes)):
            raise "Number of clients can not exceed number of classes (single_classification_mode=true)"


        #split every class in even parts
        for class_name in classes:
            count = 0

            #count images in dir
            for picture in os.listdir(directory + "\{}".format(class_name)):
                f = os.path.join(f"{directory}\\{class_name}", picture)
                # checking if it is a file
                if os.path.isfile(f):
                    count += 1

            start_index = 0
            end_index = (int)(count * 0.8)
            count = 0

            for picture in os.listdir(directory + "\{}".format(class_name)):
                f = os.path.join(f"{directory}\\{class_name}", picture)
                # checking if it is a file
                if os.path.isfile(f):
                    count += 1
                if(class_name != classes[worker_index - 1]):
                    src_fpath = f
                    dest_fpath = f"{copy_path}\\test\\not_{classes[worker_index - 1]}\\{picture}"
                    try:
                        shutil.copy(src_fpath, dest_fpath)
                    except IOError as io_err:
                        os.makedirs(os.path.dirname(dest_fpath))
                        shutil.copy(src_fpath, dest_fpath)
                        pass

                    src_fpath = f
                    dest_fpath = f"{copy_path}\\train\\not_{classes[worker_index - 1]}\\{picture}"
                    try:
                        shutil.copy(src_fpath, dest_fpath)
                    except IOError as io_err:
                        os.makedirs(os.path.dirname(dest_fpath))
                        shutil.copy(src_fpath, dest_fpath)
                        pass
                else:
                    if (count in range(start_index, end_index)):
                        src_fpath = f
                        dest_fpath = f"{copy_path}\\train\\{class_name}\\{picture}"
                        try:
                            shutil.copy(src_fpath, dest_fpath)
                        except IOError as io_err:
                            os.makedirs(os.path.dirname(dest_fpath))
                            shutil.copy(src_fpath, dest_fpath)
                            pass
                    else:
                        src_fpath = f
                        dest_fpath = f"{copy_path}\\test\\{class_name}\\{picture}"
                        try:
                            shutil.copy(src_fpath, dest_fpath)
                        except IOError as io_err:
                            os.makedirs(os.path.dirname(dest_fpath))
                            shutil.copy(src_fpath, dest_fpath)
                            pass


        # shutil.copytree(directory + "\{}".format(classes[worker_index - 1]), copy_path + "\\train\{}".format(classes[worker_index - 1]))
        # shutil.copytree(directory, copy_path + "\\test")
    else:
        #split every class in even parts
        for class_name in classes:
            count = 0

            #count images in dir
            for picture in os.listdir(directory + "\{}".format(class_name)):
                f = os.path.join(f"{directory}\\{class_name}", picture)
                # checking if it is a file
                if os.path.isfile(f):
                    count += 1

            pic_amount_per_worker = (int)(count / worker_count)  
            start_index = (worker_index - 1) * pic_amount_per_worker
            end_index = start_index + pic_amount_per_worker  
            count = 0

            for picture in os.listdir(directory + "\{}".format(class_name)):
                f = os.path.join(f"{directory}\\{class_name}", picture)
                # checking if it is a file
                if os.path.isfile(f):
                    count += 1
                if os.path.isfile(f) and count in range(start_index, end_index):
                    src_fpath = f
                    dest_fpath = f"{copy_path}\\{class_name}\\{picture}"
                    try:
                        shutil.copy(src_fpath, dest_fpath)
                    except IOError as io_err:
                        os.makedirs(os.path.dirname(dest_fpath))
                        shutil.copy(src_fpath, dest_fpath)
                        pass

    return copy_path

def check_if_already_split(copy_path, single_classification_mode):
    if(os.path.isdir(copy_path)):
        for base, dirs, files in os.walk(copy_path):
            if(single_classification_mode and len(dirs) == 1):
                return True
            if(not single_classification_mode and len(dirs) != 1):
                return True
            return False
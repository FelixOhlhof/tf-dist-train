from ast import While
from copy import copy
from operator import length_hint
from csv import writer
import shutil
import os
from time import sleep
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import random as rn
from datetime import datetime



def copy_pictures(directory, worker_index, worker_count, binary_classification_mode, debug_mode):
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

    if(binary_classification_mode):
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
                    if (count not in range(start_index, end_index)):
                        src_fpath = f
                        dest_fpath = f"{copy_path}\\test\\not_{classes[worker_index - 1]}\\{picture}"
                        try:
                            shutil.copy(src_fpath, dest_fpath)
                        except IOError as io_err:
                            os.makedirs(os.path.dirname(dest_fpath))
                            shutil.copy(src_fpath, dest_fpath)
                            pass
                    else:
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
        for _, dirs, _ in os.walk(copy_path):
            if(single_classification_mode and len(dirs) == 1):
                return True
            if(not single_classification_mode and len(dirs) != 1):
                return True
            return False

            
def append_list_as_row(file_name, list_of_elem):
    tmp = 0
    while(True):
        try:
            # Open file in append mode
            with open(file_name, 'a', newline='') as write_obj:
                for e in localize_floats(list_of_elem):
                    if e == '\n':
                        write_obj.write('\n')
                    else:
                        write_obj.write(str(e) + ';')
                # # Create a writer object from csv module
                # csv_writer = writer(write_obj, delimiter=';')
                # # Add contents of list as last row in the csv file
                # csv_writer.writerow(localize_floats(list_of_elem))
            return
        except Exception as ex:
            if(tmp > 3):
                raise Exception('Report Error file blocked...', ex)
            sleep(1)
            tmp+=1

def localize_floats(row):
    return [
        str(el).replace('.', ',') if isinstance(el, float) else el 
        for el in row
    ]

def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    return "{0}:{1}:{2}".format(int(hours), int(mins), int(sec))

def report(values):
    append_list_as_row('stats.csv', values)

def show_plot(history, epochs, client_id):
    #visualize training results
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    now = datetime.now().strftime("%H.%M.%S")
    graph = f"results\\{client_id}_{now}.png"
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(graph, dpi=600)
    plt.close()
    return graph
    # plt.show()
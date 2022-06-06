import shutil
import os

def copy_pictures(directory, worker_index, worker_count):
    classes = []
    copy_path = directory[0:directory.rindex('\\')] + "\{}".format(worker_index)

    #return if data is already split
    if(os.path.isdir(copy_path)):
        return copy_path

    # iterate over files in
    # that directory
    for class_name in os.listdir(directory):
        f = os.path.join(directory, class_name)
        # checking if it is a file
        if not os.path.isfile(f):
            classes.append(class_name)

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
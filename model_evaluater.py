import sqlite3
import os, time
import numpy as np

import tensorflow as tf
from tensorflow import keras

class Model():
  def __init__(self, TEST_ID, MODEL_PATH, STRATEGY, CLASS_1, CLASS_2, WORKER_ID):
    self.TEST_ID = TEST_ID
    self.MODEL_PATH = MODEL_PATH
    self.STRATEGY = STRATEGY
    self.CLASS_1 = CLASS_1
    self.CLASS_2 = CLASS_2
    self.WORKER_ID = WORKER_ID
    self.MODEL = tf.keras.models.load_model(MODEL_PATH)
    self.correct_pred = 0

def load_image(img_path):
    img = keras.preprocessing.image.load_img(
                  img_path, target_size=(180, 180)
              )
    img_tensor = keras.preprocessing.image.img_to_array(img)                   # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def get_classes(ds):
    classes = []
    for filename in os.listdir(ds): # for each class in directory
        classes.append(filename.split('\\')[-1])
    return classes

def get_last_test_id():
    con = sqlite3.connect(db)
    cur = con.cursor()
    last_test_id = cur.execute(f"SELECT Count(*) from Results").fetchone()[0]
    con.close()
    return last_test_id

def get_strategy(test_id):
    con = sqlite3.connect(db)
    cur = con.cursor()
    strategy = cur.execute(f"SELECT Strategy FROM Results WHERE TEST_ID = {test_id}").fetchone()[0]
    con.close()
    return strategy

def get_models():
    models = []

    con = sqlite3.connect(db)
    cur = con.cursor()
    m = cur.execute(f"SELECT * from Models where TEST_ID = {last_test_id}").fetchall()
    con.close()

    for s in m:
        models.append(Model(s[0], s[1], s[2], s[3], s[4], s[5]))

    return models

def test(ds):
    models = get_models()
    classes = get_classes(ds)

    correct_predicted = 0
    total_predicted = 0

    for filename in os.listdir(ds): # for each class in directory
        x = os.path.join(ds, filename)
        true_class = filename.split('\\')[-1]

        for f in os.listdir(x): # for each file picture in class 
            predictions = []
            image = load_image(os.path.join(x, f))

            for m in models: # run picture through every model
                predictions.append(1 - m.MODEL.predict(image, verbose='0'))

            predicted_class = classes[np.argmax(predictions)]

            if(true_class == predicted_class):
                correct_predicted += 1
                models[np.argmax(predictions)].correct_pred += 1
            total_predicted +=1
            models[np.argmax(predictions)].total_pred += 1

    accuracy = correct_predicted/total_predicted
    print(accuracy)


def get_validation_ovo(ds):
    correct_predicted = 0
    correct_predicted_classes = [0 for i in range(len(classes))]
    total_predicted = 0

    y_true = []
    y_pred = []

    images = []

    for filename in os.listdir(ds): # for each class in directory
        x = os.path.join(ds, filename)
        true_class = filename.split('\\')[-1]

        for f in os.listdir(x): # for each file picture in class 
            predictions = []
            image = load_image(os.path.join(x, f))
            images.append(image)

            y_true.append(classes.index(true_class))
    
            
    images = np.vstack(images)
    start_time = time.time()

    for m in models: # run picture through every model
        predictions.append(m.MODEL.predict(images, verbose='0'))

    end_time = time.time()
    evaluation_time = end_time - start_time

    for i in range(len(predictions[0])):
        pred_dan = 0
        pred_ros = 0
        pred_sun = 0
        pred_tup = 0

        for m in range(len(models)):
            if(models[m].CLASS_1 == 'dandelion'):
                pred_dan += 1 - predictions[m][i][0]
            if(models[m].CLASS_1 == 'roses'):
                pred_ros += 1 - predictions[m][i][0]
            if(models[m].CLASS_1 == 'sunflowers'):
                pred_sun += 1 - predictions[m][i][0]
            if(models[m].CLASS_1 == 'tulips'):
                pred_tup += 1 - predictions[m][i][0]
            if(models[m].CLASS_2 == 'dandelion'):
                pred_dan += predictions[m][i][0]
            if(models[m].CLASS_2 == 'roses'):
                pred_ros += predictions[m][i][0]
            if(models[m].CLASS_2 == 'sunflowers'):
                pred_sun += predictions[m][i][0]
            if(models[m].CLASS_2 == 'tulips'):
                pred_tup += predictions[m][i][0]

        y_pred.append([pred_dan, pred_ros, pred_sun, pred_tup])

        predicted_class = classes[np.argmax(y_pred[i])]

        if(classes[y_true[i]] == predicted_class):
            correct_predicted += 1
            correct_predicted_classes[np.argmax(y_pred[i])] += 1
        total_predicted +=1


        # pred_dan = []
        # pred_ros = []
        # pred_sun = []
        # pred_tup = []

        # for m in range(len(models)):
        #     if(models[m].CLASS_1 == 'dandelion'):
        #         pred_dan.append(1 - predictions[m][i])
        #     if(models[m].CLASS_1 == 'roses'):
        #         pred_ros.append(1 - predictions[m][i])
        #     if(models[m].CLASS_1 == 'sunflowers'):
        #         pred_sun.append(1 - predictions[m][i])
        #     if(models[m].CLASS_1 == 'tulips'):
        #         pred_tup.append(1 - predictions[m][i])
        #     if(models[m].CLASS_2 == 'dandelion'):
        #         pred_dan.append(predictions[m][i])
        #     if(models[m].CLASS_2 == 'roses'):
        #         pred_ros.append(predictions[m][i])
        #     if(models[m].CLASS_2 == 'sunflowers'):
        #         pred_sun.append(predictions[m][i])
        #     if(models[m].CLASS_2 == 'tulips'):
        #         pred_tup.append(predictions[m][i])

        # y_pred.append([pred_dan[np.argmax(pred_dan)][0], pred_ros[np.argmax(pred_dan)][0], pred_sun[np.argmax(pred_dan)][0], pred_tup[np.argmax(pred_dan)][0]])

    # for i in range(len(y_pred)):
    #     predicted_class = classes[np.argmax(y_pred[i])]

    #     if(classes[y_true[i]] == predicted_class):
    #         correct_predicted += 1
    #     total_predicted +=1

    accuracy = round(correct_predicted/total_predicted, 4)
    correct_predicted_classes_accuracy = [round(x*len(classes)/total_predicted, 4) for x in correct_predicted_classes]
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = round(float(scce(y_true, y_pred).numpy()), 4)
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    print(f"Class Accuracy: {correct_predicted_classes_accuracy}")
    return accuracy, loss, round(evaluation_time, 2),  correct_predicted_classes_accuracy


def get_validation_ovr(ds):
    correct_predicted = 0
    correct_predicted_classes = [0 for i in range(len(classes))]
    total_predicted = 0

    y_true = []
    y_pred = []

    images = []

    for filename in os.listdir(ds): # for each class in directory
        x = os.path.join(ds, filename)
        true_class = filename.split('\\')[-1]

        for f in os.listdir(x): # for each file picture in class 
            predictions = []
            image = load_image(os.path.join(x, f))
            images.append(image)

            y_true.append(classes.index(true_class))
    
            
    images = np.vstack(images)
    start_time = time.time()

    for m in models: # run picture through every model
        predictions.append(1 - m.MODEL.predict(images, verbose='0'))

    end_time = time.time()
    evaluation_time = end_time - start_time

    for i in range(len(predictions[0])):
        y_pred.append([predictions[0][i][0], predictions[1][i][0], predictions[2][i][0], predictions[3][i][0]])

    for i in range(len(y_pred)):
        predicted_class = classes[np.argmax(y_pred[i])]

        if(classes[y_true[i]] == predicted_class):
            correct_predicted += 1
            correct_predicted_classes[np.argmax(y_pred[i])] += 1
        total_predicted +=1
        
    accuracy = round(correct_predicted/total_predicted, 4)
    correct_predicted_classes_accuracy = [round(x*len(classes)/total_predicted, 4) for x in correct_predicted_classes]
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = round(float(scce(y_true, y_pred).numpy()), 4)
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    print(f"Class Accuracy: {correct_predicted_classes_accuracy}")
    return accuracy, loss, round(evaluation_time, 2),  correct_predicted_classes_accuracy

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    db = r"results\tf_dist_train_results.db"
    ds_test = r"C:\Users\felix\.keras\datasets\Validation\animals_small"
    ds_train = r"C:\Users\felix\.keras\datasets\animals_small"
    ds_tmp = r"C:\Users\felix\.keras\datasets\tmp"


    last_test_id = get_last_test_id()
    strategy = get_strategy(last_test_id)
    models = get_models()
    classes = get_classes(ds_test)


    if(strategy == 'ONE_VS_REST'):
        val_accuracy, val_loss, evaluation_time, correct_predicted_classes_accuracy  = get_validation_ovr(ds_test)
        accuracy, loss,_,_ = get_validation_ovr(ds_train)
    else:
        val_accuracy, val_loss, evaluation_time, correct_predicted_classes_accuracy = get_validation_ovo(ds_test)
        accuracy, loss,_,_ = get_validation_ovo(ds_train)
        
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    hlp = ";".join(map(str, correct_predicted_classes_accuracy))
    sql = f"UPDATE Results SET VAL_ACCURACY = {val_accuracy}, VAL_LOSS = {val_loss}, EVALUATION_TIME = {evaluation_time}, ACCURACY = {accuracy}, LOSS = {loss} , CLASSES_VAL_ACCURACY = '{hlp}' WHERE TEST_ID = {last_test_id}"
    cur.execute(sql)
    con.commit()
    con.close()
    print(f"Updated result of Test {last_test_id}")
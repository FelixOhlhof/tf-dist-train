import argparse
import configparser
from unittest import case
# Parameters
parser = argparse.ArgumentParser(description='Client for tensorflow distributed training')
parser.add_argument("-i", "--id", help="The id of the client", type=int, required=True)
args = parser.parse_args()
config = configparser.ConfigParser()
config.read("config.ini")
from struct import pack
from time import sleep
import pickle, sys, os
import socket, util
import time
from classifier import Classifier
from binary_classifier import BinaryClassifier as BinaryClassifier
from http import client
from pathlib import Path


class Client():
    def __init__(self):
        self.use_gpu = config.getboolean("CLIENT","USE_GPU")
        if(not self.use_gpu):
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # nessesary for the seed
        self.seed = (int)(config["CLIENT"]["SEED"]) # get the seed
        self.epochs=(int)(config["CLIENT"]["EPOCHES"]) # Number of epoches to be trained
        self.hostname = config["SERVER"]["HOST"] # The server's hostname or IP address
        self.port = (int)(config["SERVER"]["PORT"]) # The port used by the server
        self.client_id = args.id # The ID of the worker/client
        self.client_count = (int)(config["CLIENT"]["TOTAL_CLIENTS"])
        self.batch_size = (int)(config["CLIENT"]["BATCH_SIZE"])
        self.dataset_name = config["CLIENT"]["DATASET_NAME"]
        self.shuffle_data_mode = config.getboolean("CLIENT","SHUFFLE_DATA_MODE")
        self.one_vs_rest = config.getboolean("CLIENT","ONE_VS_REST")
        self.one_vs_one = config.getboolean("CLIENT","ONE_VS_ONE")
        self.send_weights_without_improvement = config.getboolean("CLIENT","SEND_WEIGHTS_WITHOUT_IMPROVEMENT")
        self.save_checkpoint = config.getboolean("CLIENT","SAVE_CHECKPOINT")
        self.load_checkpoint = config.getboolean("CLIENT","LOAD_CHECKPOINT")
        self.debug_mode = config.getboolean("CLIENT","DEBUG_MODE")
        self.data_dir = util.copy_pictures(f"{Path.home()}\\.keras\\datasets\\{self.dataset_name}", self.client_id, self.client_count, self.one_vs_rest, self.one_vs_one, self.debug_mode)

    def start_dist_training(self):
        if(self.one_vs_rest):
            self.train_one_vs_rest()
        elif(self.one_vs_rest):
            self.train_one_vs_one()
        else:
            self.train_multi_class()
        
    def train_one_vs_rest(self):
        classifier = BinaryClassifier(self.data_dir, self.seed)
        classifier.train_epoch(self.epochs)

    def train_one_vs_one(self):
        # TODO:impl
        pass

    def train_multi_class(self):
        best_accuracy = 0.0
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        classifier = Classifier(self.client_id, self.client_count, self.data_dir, self.seed, self.save_checkpoint, self.load_checkpoint, self.batch_size, self.shuffle_data_mode)

        start_time = time.time()

        for i in range(self.epochs):
            print(f"************* EPOCH {i+1} *************")
            if(self.shuffle_data_mode):
                hist = classifier.train_epoch_in_shuffle_mode(inner_epoch=1, current_epoch=i)
            else:
                hist = classifier.train_epoch(inner_epoch=1)

            history = self.updateHistory(hist, history)
            new_accuracy = hist.history['accuracy'][0]
            print(hist.history)
            
            if new_accuracy > best_accuracy or self.send_weights_without_improvement:
                print(f" -> best_accuracy: {best_accuracy} new_accuracy: {new_accuracy}")
                # send weights and get new weights
                updated_weights = self.retrieve_new_weights(old_weights=classifier.model.get_weights())
            else:
                # send skip to server
                print(f"Accuracy did not improve -> best_accuracy: {best_accuracy} new_accuracy: {new_accuracy}")
                updated_weights = self.send_skip(old_weights=classifier.model.get_weights())

            # set updated weights    
            classifier.model.set_weights(updated_weights)

            if(new_accuracy > best_accuracy):
                best_accuracy = new_accuracy
            print()

        print(classifier.model.history.history)
        score = classifier.calculate_score()

        # report
        end_time = time.time()
        time_lapsed = end_time - start_time
        sleep((self.client_id - 1) * 4)
        graph_path = util.save_validation_loss_plot(history, self.epochs, self.client_id)
        self.report(time_lapsed, classifier, hist, score, graph_path)

        

    def report(self, time_lapsed, classifier, hist, score, graph_path):
        base_infos = [self.client_count, self.epochs, classifier.batch_size, self.one_vs_rest,self.shuffle_data_mode, self.send_weights_without_improvement, self.seed, self.use_gpu, util.time_convert(time_lapsed), round(score, 2)]
        training_stats = [round(hist.history['accuracy'][0], 2), round(hist.history['loss'][0], 2), round(hist.history['val_accuracy'][0], 2), round(hist.history['val_loss'][0], 2), graph_path]

        if(self.client_id == 1):
            util.report(base_infos + training_stats)
            return
        else:
            sleep(self.client_id * 3)
            util.report(training_stats)
        if(self.client_id == self.client_count):
            sleep(self.client_id * 2)
            util.report(['\n'])
            return

    def send_skip(self, old_weights):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.hostname, self.port))

            #send skip flag
            s.send(pickle.dumps(-1))
            print(pickle.loads(s.recv(1024)))

            #get size of new weights
            msg_len = pickle.loads(s.recv(1024))
            print(f"Recieved size from Server {msg_len}")

            if(msg_len == -1):
                # no worker improved, server sent skip flag
                print("No worker improved, continuing with old weights!")
                return old_weights

            # avg weights from other workers
            new_weights = pickle.loads(s.recv(msg_len))
            print("Recieved new weights")
            return new_weights


    def retrieve_new_weights(self, old_weights):
        print("First weight before merging: ", old_weights[0][0][0][0][0])
        b_weights = pickle.dumps(old_weights)
        print(sys.getsizeof(b_weights), " bytes will be transfered to server...")

        # Send weights and get new new calculated weights
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.hostname, self.port))
            
            #send size of weights
            s.send(pickle.dumps(sys.getsizeof(b_weights)))
            print(pickle.loads(s.recv(1024)))

            #send weights
            s.sendall(b_weights)
            #print(pickle.loads(s.recv(1024)))

            #get size of new weights
            msg_len = pickle.loads(s.recv(1024))

            #rec new weights
            print(f"Recieved length of new weights {msg_len}")
            data = s.recv(msg_len)

            new_weights = pickle.loads(data)
            print("Recieved new weights! First weight after merging:", new_weights[0][0][0][0][0])
            return new_weights

    def updateHistory(self, hist, history):
        history['loss'].append(hist.history['loss'][0])
        history['accuracy'].append(hist.history['accuracy'][0])
        history['val_loss'].append(hist.history['val_loss'][0])
        history['val_accuracy'].append(hist.history['val_accuracy'][0])
        return history

if __name__ == "__main__":
    client = Client()    
    client.start_dist_training()
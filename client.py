import argparse
import configparser
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
from flowerclassifier import Flowerclassifier
from binary_flowerclassifier import Flowerclassifier as BinaryClassifier
from http import client
from pathlib import Path


class Client():
    def __init__(self):
        if(config.getboolean("CLIENT","USE_SEED")):
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # nessesary for the seed
        self.seed = (int)(config["CLIENT"]["SEED"]) # get the seed
        self.epochs=(int)(config["CLIENT"]["EPOCHES"]) # Number of epoches to be trained
        self.hostname = config["SERVER"]["HOST"] # The server's hostname or IP address
        self.port = (int)(config["SERVER"]["PORT"]) # The port used by the server
        self.client_id = args.id # The ID of the worker/client
        self.client_count = (int)(config["CLIENT"]["TOTAL_CLIENTS"]) # Total worker/client count TODO: clients register -> server starts training (by command) 
        self.dataset_name = config["CLIENT"]["DATASET_NAME"]
        self.single_classification_mode = config.getboolean("CLIENT","SINGLE_CLASSIFICATION_MODE")
        self.send_weights_without_improvement = config.getboolean("CLIENT","SEND_WEIGHTS_WITHOUT_IMPROVEMENT")
        self.save_checkpoint = config.getboolean("CLIENT","SAVE_CHECKPOINT")
        self.load_checkpoint = config.getboolean("CLIENT","LOAD_CHECKPOINT")
        self.debug_mode = config.getboolean("CLIENT","DEBUG_MODE")
        self.data_dir = util.copy_pictures(f"{Path.home()}\\.keras\\datasets\\{self.dataset_name}", self.client_id, self.client_count, self.single_classification_mode, self.debug_mode)

    def start_dist_training(self):
        if(self.single_classification_mode):
            self.train_binary()
        else:
            self.train_multi_class()


    def train_binary(self):
        classifier = BinaryClassifier(self.data_dir, self.seed)
        classifier.train_epoch(self.epochs)

    def train_multi_class(self):
        best_accuracy = 0.0
        validation = None
        classifier = Flowerclassifier(self.client_id, self.data_dir, self.seed, self.save_checkpoint, self.load_checkpoint)

        for i in range(self.epochs):
            print(f"************* EPOCH {i+1} *************")
            validation = classifier.train_epoch(inner_epoch=1)
            new_accuracy = validation.history['accuracy'][0]
            print(validation.history)
            
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
        classifier.classify_single_picture()
        #classifier.show_plot(validation, self.epochs)

    def send_skip(self, old_weights):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, proto=6) as s:
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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, proto=6) as s:
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



if __name__ == "__main__":
    client = Client()    
    client.start_dist_training()
import argparse
from struct import pack
from time import sleep
parser = argparse.ArgumentParser(description='Client for tensorflow distributed training')
parser.add_argument("-i", "--id", help="The id of the client", type=int, required=True)
args = parser.parse_args()
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
import pickle, sys, os
import socket, util
from flowerclassifier import Flowerclassifier
from http import client
from pathlib import Path

class Client():
    def __init__(self, hostname, port, client_id, client_count, dataset_name, single_classification_mode):
        self.hostname = hostname # The server's hostname or IP address
        self.port = port # The port used by the server
        self.client_id = client_id # The ID of the worker/client
        self.client_count = client_count # Total worker/client count TODO: clients register -> server starts training (by command) 
        self.data_dir = util.copy_pictures(f"{Path.home()}\\.keras\\datasets\\{dataset_name}", self.client_id, self.client_count, single_classification_mode)

    def start_dist_training(self, epochs):
        classifier = Flowerclassifier(self.data_dir)

        for i in range(epochs):
            validation = classifier.train_epoch(inner_epoch=1)
            print(validation.history)
            #TODO: check if accuracy improved otherwise don't mean => send skip to server or similar
            updated_weights = self.retrieve_new_weights(old_weights=classifier.model.get_weights())
            classifier.model.set_weights(updated_weights)

        print(classifier.model.history.history)
        classifier.classify_picture()


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



if __name__ == "__main__":
    client = Client(config["SERVER"]["HOST"], (int)(config["SERVER"]["PORT"]), client_id=args.id , client_count=(int)(config["CLIENT"]["TOTAL_CLIENTS"]), dataset_name=config["CLIENT"]["DATASET_NAME"], single_classification_mode=config.getboolean("CLIENT","SINGLE_CLASSIFICATION_MODE"))    
    client.start_dist_training(epochs=(int)(config["CLIENT"]["EPOCHES"]))
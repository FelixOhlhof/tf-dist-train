import argparse
parser = argparse.ArgumentParser(description='Client for tensorflow distributed training')
parser.add_argument("-i", "--id", help="The id of the client", type=int, required=True)
args = parser.parse_args()
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
import pickle, sys
import socket, util
from flowerclassifier import Flowerclassifier
from http import client
from pathlib import Path

class Client():
    def __init__(self, hostname, port, client_id, client_count, dataset_name):
        self.hostname = hostname # The server's hostname or IP address
        self.port = port # The port used by the server
        self.client_id = client_id # The ID of the worker/client
        self.client_count = client_count # Total worker/client count TODO: clients register -> server starts training (by command) 
        self.data_dir = util.copy_pictures(f"{Path.home()}\\.keras\\datasets\\{dataset_name}", self.client_id, self.client_count)

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
            s.sendall(b_weights)
            print("weights sent to server")
            #wait here until server calculated mean???
            msg_len = 31914812 #31914802 #TODO: auto
            arr = bytearray()
            pos = 0      
            max_msg_size = 4096
            try:
                while pos < msg_len: #can you not just do while true and break out when no packets are coming anymore?
                    packet = s.recv(max_msg_size) #receive data from server
                    #if not packet: break
                    pos += max_msg_size
                    arr.extend(packet)
                    print("received: ", len(arr), "von: ", msg_len)
            finally:
                byteObj = bytes(arr)
                print(byteObj[-20:])

                new_weights = pickle.loads(byteObj)
                print("Recieved new weights! First weight after merging:", new_weights[0][0][0][0][0])
                return new_weights



if __name__ == "__main__":
    client = Client(config["SERVER"]["HOST"], (int)(config["SERVER"]["PORT"]), client_id=args.id , client_count=(int)(config["CLIENT"]["TOTAL_CLIENTS"]), dataset_name=config["CLIENT"]["DATASET_NAME"])       
    client.start_dist_training(epochs=(int)(config["CLIENT"]["EPOCHES"]))
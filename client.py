import argparse
parser = argparse.ArgumentParser(description='Client for tensorflow distributed training')
parser.add_argument("-i", "--id", help="The id of the client", type=int, required=True)
parser.add_argument("-c", "--count", help="Total count of clients", type=int, required=True)
args = parser.parse_args()
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

        pass

    def retrieve_new_weights(self, old_weights):
        print("First weight before merging: ", old_weights[0][0][0][0][0])
        b_weights = pickle.dumps(old_weights)
        print(sys.getsizeof(b_weights), " bytes will be transfered to server...")

        # Send weights and get new new calculated weights
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.hostname, self.port))
            s.sendall(b_weights)
            
            msg_len = 31914802 #TODO: auto
            arr = bytearray()
            pos = 0      
            max_msg_size = 4096

            while pos < msg_len:
                packet = s.recv(max_msg_size)
                pos += max_msg_size
                arr.extend(packet)

            byteObj = bytes(arr)

            new_weights = pickle.loads(byteObj)
            print("Recieved new weights! First weight after merging:", new_weights[0][0][0][0][0])
            return new_weights



if __name__ == "__main__":
    client = Client("127.0.0.1", 65433, client_id=args.id , client_count=args.count, dataset_name="flower_photos")       
    client.start_dist_training(epochs=1)


# TODO: epoche......https://machinelearningmastery.com/polyak-neural-network-model-weight-ensemble/
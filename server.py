from ast import With
import configparser
from time import sleep
config = configparser.ConfigParser()
config.read('config.ini')
import multiprocessing
import socket, sys, pickle
from numpy import average
from numpy import array
import random

WORKERS_COUNT = (int)(config["CLIENT"]["TOTAL_CLIENTS"])

def get_average_weights(members):
    # prepare an array of equal weights
    n_models = len(members)
    weights = [1/n_models for i in range(1, n_models+1)]

    new_weights = members[0]
    
    # determine how many layers need to be averaged
    n_layers = len(members[0])

    for layer in range(n_layers):
        # collect this layer from each model
        layer_weights = array([model[layer] for model in members])
        # weighted average of weights for this layer
        avg_layer_weights = average(layer_weights, axis=0, weights=weights)
        # store average layer weights
        new_weights[layer] = avg_layer_weights
    return new_weights

def calc_new_weights(worker_weights_queue, new_weights):
    while True:
        sleep(0.1)
        #Every worker send its weights?
        if(worker_weights_queue.full()):
            #Mean, Set new_weights
            print("Calculating new weights")
            l = list()

            while(worker_weights_queue.empty()):
                print("some extra time") # nessesary because qsize()/qfull() is not relyable
                sleep(0.1)
            while not worker_weights_queue.empty():
                data = worker_weights_queue.get()
                if(data != -1):
                    l.append(data)

            if(len(l) == 0):
                for i in range(WORKERS_COUNT):
                    new_weights.put(-1)
            else:
                # calc weights
                nw = get_average_weights(l)
                #put cal weights in queue for every worker
                for i in range(WORKERS_COUNT):
                    new_weights.put(nw)
            

def update_sync(data, worker_weights_queue, new_weights): 
    b = pickle.loads(data)
    worker_weights_queue.put(b) 
    #Sync/wait for the other worker
    print("Worker waiting") 
    return pickle.dumps(new_weights.get())

def handle(conn, address, worker_weights_queue, new_weights):
    try:
        print(f"Client {address} connected")
        # rec size of msg length
        msg_len = pickle.loads(conn.recv(1024))
        conn.send(pickle.dumps(f"SERVER: msg_len {msg_len} recieved"))

        if(msg_len == -1): # -1 = skip flag
            # client weights didn't improve, set skip flag for queue
            print(f"Client {address} sent skip flag")
            worker_weights_queue.put(-1)
            #wait for calculation
            data = new_weights.get()

            if(data == -1):
                #send skip flag
                print(f"No client improved, sending skip flag to client {address}")
                conn.send(pickle.dumps(-1))
            else:
                # send size of weights
                data = pickle.dumps(data)
                conn.send(pickle.dumps(sys.getsizeof(data)))
                # send weights
                print(f"Sending new weights to client {address}")
                conn.send(data)
            return

        # rec weights
        data = conn.recv(msg_len)
        # conn.send(pickle.dumps(f"SERVER: Recieved weights"))

        print(sys.getsizeof(data), f" bytes recieved from {address}")
        data = update_sync(data, worker_weights_queue, new_weights)

        print(f"sending {sys.getsizeof(data)} bytes to {address}")
        if(sys.getsizeof(data) == 50):
            print("oh no")
        # send size of weights
        conn.send(pickle.dumps(sys.getsizeof(data)))
        # send weights
        conn.send(data)
    finally:
        conn.close()

class Server():
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        manager = multiprocessing.Manager()
        self.worker_weights_queue = multiprocessing.Queue(WORKERS_COUNT)
        self.new_weights = multiprocessing.Queue()

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)

        #start calculating process
        process = multiprocessing.Process(target=calc_new_weights, args=(self.worker_weights_queue, self.new_weights))
        process.daemon = True
        process.start()
        
        while True:
            conn, address = self.socket.accept()
            process = multiprocessing.Process(target=handle, args=(conn, address, self.worker_weights_queue, self.new_weights))
            process.daemon = True
            process.start()

if __name__ == "__main__":
    server = Server(config["SERVER"]["HOST"], (int)(config["SERVER"]["PORT"]))
    try:
        print("Server Start")
        server.start()
    finally:
        for process in multiprocessing.active_children():
            process.terminate()
            process.join()
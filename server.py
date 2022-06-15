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

# Number of workers
workers_count = (int)(config["CLIENT"]["TOTAL_CLIENTS"])

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

		# f = open("sample_org.txt", "wb")
		# f.write(pickle.dumps(new_weights))
		# f.close()
	return new_weights

def calc_new_weights(worker_weights_queue, new_weights, mutex):
    while True:
        sleep(0.1)
        #Every worker send its weights?
        if(worker_weights_queue.qsize() == workers_count):
            #Mean, Set new_weights
            print("Calculating new weights")
            l = list()
            while not worker_weights_queue.empty():
                l.append(worker_weights_queue.get())
            new_weights.append(get_average_weights(l))
            

def update_sync(data, worker_weights_queue, mutex, new_weights): 
    #Next epoch
    # if(len(worker_weights) == 0 and len(new_weights) != 0):
    #     new_weights[:] = []

    b = pickle.loads(data)
    worker_weights_queue.put(b)   

    print("Worker waiting") 

    #Sync/wait for the other worker
    while(len(new_weights) == 0):
        sleep(0.05)
    return pickle.dumps(new_weights[0])

def handle(conn, address, worker_weights_queue, mutex, new_weights):
    print(f"Client {address} connected")
    #rec size of msg length
    msg_len = pickle.loads(conn.recv(1024))
    conn.send(pickle.dumps(f"SERVER: Size {msg_len} recieved"))

    #rec weights
    data = conn.recv(msg_len)
    #conn.send(pickle.dumps(f"SERVER: Recieved weights"))

    print(sys.getsizeof(data), f" bytes recieved from {address}")
    data = update_sync(data, worker_weights_queue, mutex, new_weights)

    print(f"sending {sys.getsizeof(data)} bytes to {address}")
    
    #send size of weights
    conn.send(pickle.dumps(sys.getsizeof(data)))
    conn.send(data)
    conn.close()

class Server():
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        manager = multiprocessing.Manager()
        self.worker_weights_queue = multiprocessing.Queue()
        self.new_weights = manager.list() #TODO: Array
        self.mutex = multiprocessing.RLock()
        #self.mutex.acquire()

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)

        #start calculating process
        process = multiprocessing.Process(target=calc_new_weights, args=(self.worker_weights_queue, self.new_weights, self.mutex))
        process.daemon = True
        process.start()
        
        while True:
            conn, address = self.socket.accept()
            process = multiprocessing.Process(target=handle, args=(conn, address, self.worker_weights_queue, self.mutex, self.new_weights))
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
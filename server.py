import ctypes
import multiprocessing
import socket, sys, time, pickle
from numpy import average
from numpy import array

# Number of workers
workers_count = 2

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

def update_sync(data, worker_weights, mutex, new_weights): 
    #Next epoch
    if(len(worker_weights) == 0 and len(new_weights) != 0):
        new_weights[:] = []

    b = pickle.loads(data)
    worker_weights.append(b)   

    #Every worker send its weights?
    if(len(worker_weights) == workers_count):
        #Mean, Set new_weights
        new_weights.append(get_average_weights(worker_weights))
        #Reset
        worker_weights[:]=[]
        #Every thread can send the updated weights to the client
        mutex.release()
        return pickle.dumps(new_weights[0])
            
    #Sync/wait for the other worker
    mutex.acquire()

    return pickle.dumps(new_weights[0])

def handle(conn, address, worker_weights, mutex, new_weights):
    print(f"Client {address} connected")
    msg_len = 15957659 #TODO: auto
    arr = bytearray()
    pos = 0      
    max_msg_size = 4096

    while pos < msg_len:
        packet = conn.recv(max_msg_size)
        pos += max_msg_size
        arr.extend(packet)

    byteObj = bytes(arr) #TODO: check if byte string is faster

    print(sys.getsizeof(byteObj))

    data = update_sync(byteObj, worker_weights, mutex, new_weights)
    print(f"sending data ({sys.getsizeof(data)}) to {address}")

    conn.send(data)
    conn.close()

class Server():
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.manager = multiprocessing.Manager()
        self.worker_weights = self.manager.list()
        self.new_weights = self.manager.list() #TODO: Array
        self.mutex = self.manager.Lock()
        self.mutex.acquire()

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)
        
        while True:
            conn, address = self.socket.accept()
            process = multiprocessing.Process(target=handle, args=(conn, address, self.worker_weights, self.mutex, self.new_weights))
            process.daemon = True
            process.start()

if __name__ == "__main__":
    server = Server("127.0.0.1", 65433)
    try:
        print("Server Start")
        server.start()
    finally:
        for process in multiprocessing.active_children():
            process.terminate()
            process.join()
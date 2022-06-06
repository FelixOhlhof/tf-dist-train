import ctypes
import multiprocessing
import socket, sys, time, pickle, util

# Number of workers
workers_count = 2


def update_sync(data, worker_weights, mutex, new_weights): 
    #Next epoch
    if(len(worker_weights) == 0 and len(new_weights) != 0):
        new_weights[:] = []

    b = pickle.loads(data)
    worker_weights.append(b)   

    #Every worker send its weights?
    if(len(worker_weights) == workers_count):
        #Mean, Set new_weights
        new_weights.append(util.get_average_weights(worker_weights))
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

    conn.sendall(data)
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
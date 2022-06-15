import configparser

config = configparser.ConfigParser()
config.read('config.ini')
import multiprocessing
import socket, sys, pickle
from numpy import average
from numpy import array

# Number of workers
workers_count = (int)(config["CLIENT"]["TOTAL_CLIENTS"])


def get_average_weights(members):
    # prepare an array of equal weights
    n_models = len(members)
    weights = [1 / n_models for i in range(1, n_models + 1)]
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
    # Next epoch
    # if(len(worker_weights) == 0 and len(new_weights) != 0):
    new_weights[:] = []  # why only reset in case and not always?

    b = pickle.loads(data)

    if (
            len(worker_weights) < workers_count - 1):  # could all clients check if statement at the same time and all go waiting???
        worker_weights.append(b)
        print("worker waiting")
        mutex.acquire()
    # Every worker send its weights?
    else:  # error occurs if weights are appended too fast before mutex.acquire()
        worker_weights.append(b)
        print("last workers weights received!")
        # Mean, Set new_weights
        new_weights.append(get_average_weights(worker_weights))
        print("new weights calculated!")
        # Reset
        worker_weights[:] = []
        # Every thread can send the updated weights to the client
        for i in range(workers_count - 1):
            mutex.release()
        print("return data to last worker")
        return pickle.dumps(new_weights[0])

    # Sync/wait for the other worker
    print("return data to waiting worker")
    return pickle.dumps(new_weights[0])


def handle(conn, address, worker_weights, mutex, new_weights):
    print(f"Client {address} connected")
    msg_len = 15957659  # TODO: auto
    arr = bytearray()
    pos = 0
    max_msg_size = 4096

    try:
        while pos < msg_len:  # does not work with while True ... but WHY?!?!
            packet = conn.recv(max_msg_size)
            # if not packet: break
            pos += max_msg_size
            arr.extend(packet)
    finally:
        byteObj = bytes(arr)  # TODO: check if byte string is faster
        print(byteObj[-20:])
        print(sys.getsizeof(byteObj), f" bytes recieved from {address}")

        data = update_sync(byteObj, worker_weights, mutex, new_weights)
        print(f"sending {sys.getsizeof(data)} bytes to {address}")

        conn.send(data)
        conn.close()


class Server():
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.manager = multiprocessing.Manager()
        self.worker_weights = self.manager.list()
        self.new_weights = self.manager.list()  # TODO: Array
        self.mutex = self.manager.Lock()
        self.mutex.acquire()

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(3)

        while True:
            conn, address = self.socket.accept()
            process = multiprocessing.Process(target=handle,
                                              args=(conn, address, self.worker_weights, self.mutex, self.new_weights))
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

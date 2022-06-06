import tensorflow as tf
import numpy as np
import pickle, sys, json
import socket, util
from tensorflow import keras

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65433  # The port used by the server

m_1 = tf.keras.models.load_model('./bestmodel.hdf5')
m_1.summary()

weights = m_1.get_weights()
b_weights = pickle.dumps(weights)
print(sys.getsizeof(b_weights))

# Send weights and get new new calculated weights
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b_weights)
    
    msg_len = 31914802 #TODO: auto
    arr = bytearray()
    pos = 0      
    max_msg_size = 4096

    ######## Fehler hier -> es kommen nicht alle Daten an oder zumindest nicht korrekt
    ######## vor dem senden im Server sind die Bytes definitiv noch in Ordnung (habe die data-variable serialisiert und im 
    ######## HxD-Editor verglichen) 
    ######## die unterschiedliche Größen (print(sys.getsizeof())) kommen vom hin- und her dumpen/laden bei pickle -> dies ######## macht aber nichts (habe ich ebenfalls in HxD überprüft)
    while pos < msg_len:
        packet = s.recv(max_msg_size)
        pos += max_msg_size
        arr.extend(packet)

    byteObj = bytes(arr)

    new_weights = pickle.loads(byteObj)
    print(sys.getsizeof(new_weights))
    m_1.set_weights(new_weights)

# TODO: epoche......https://machinelearningmastery.com/polyak-neural-network-model-weight-ensemble/
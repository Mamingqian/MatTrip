from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, Conv1D, Dropout, Flatten, MaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
import tensorflow as tf
import numpy as np
import data
import time

'''set hyper parameters'''
batch_size = 64
epochs = 100
latent_dim = 100  
embed_dim = 20  

'''get raw data'''
user_interest = data.get_interest() #3000*6, 3000 users, 6 categories
user_routes = data.get_routes()     #3000*? , -1 -> start, -2 -> end
input_vectors = data.combine(user_interest, user_routes) #3000*8, treat start POI and end POI as one of the inputs

def get_data():
    '''
    return "en-i":encoder_input_data,"de-i":decoder_input_data:,'de-o':decoder_target_data
    type -> numpy.ndarray
    encoder shape --- (3000,8,1)
    decoder shape --- (3000,22,31)
    number of POIs -> 31
    maximum route length -> 22
    '''

    target_pois = set() # set of all POI numbers in the city
    for route in user_routes:
        for poi in route:
            if poi not in target_pois:
                target_pois.add(poi)
    target_pois = sorted(list(target_pois))
    num_pois = len(target_pois) # 31

    max_route_length = max([len(route) for route in user_routes]) #22
    poi_to_index = dict([(poi, i) for i, poi in enumerate(target_pois)])
    category_num = 6

    # Prepare data for network
    encoder_input_data = np.zeros((len(input_vectors), category_num + 2, 1), dtype=np.float32)
    decoder_input_data = np.zeros((len(input_vectors), max_route_length, num_pois), dtype=np.float32)
    decoder_target_data = np.zeros((len(input_vectors), max_route_length, num_pois), dtype=np.float32)

    # (3000,8,1)
    for i, user_interest in enumerate(input_vectors):
        for t, category in enumerate(user_interest):
            encoder_input_data[i, t, 0] = category
    
    # (3000,22,31)
    for i, target_route in enumerate(user_routes):
        for t, poi in enumerate(target_route):
            decoder_input_data[i, t, poi_to_index[poi]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, poi_to_index[poi]] = 1.0
    
    Dict={"en-i":encoder_input_data,"de-i":decoder_input_data,"de-o":decoder_target_data}
    return Dict



def get_parameter():
    '''
    return {"dm_input": num_pois, "Ty": decoder_length, "Tx": encoder_length}
    dm_input = 31
    Tx = 8
    Ty = 22
    '''
    target_pois = set()
    for route in user_routes:
        for poi in route:
            if poi not in user_pois:
                target_pois.add(poi)
    target_pois = sorted(list(target_pois))

    num_pois = len(target_pois) #31
    decoder_length = max([len(route) for route in target_routes]) #22
    encoder_length = len(user_interest[0])+2 #8

    Dict = {"dm_input": num_pois, "Ty": decoder_length, "Tx": encoder_length}
    return Dict



def model(latent_dim):
    '''
    return a seq2seq model
    '''
    parameters = get_parameter()
    dm_input, Tx, Ty = parameters["dm_input"], parameters["Tx"],  parameters["Ty"]

    encoder_inputs = Input(shape=(Tx, 1))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, h, c = encoder(encoder_inputs)
    encoder_states = [h,c]

    decoder_inputs = Input(shape = (None, dm_input))
    decoder = LSTM(latent_dim, return_sequences = True, return_state = True, dropout=0.2, recurrent_dropout=0.5)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state = encoder_states)

    decoder_dense = Dense(dm_input, activation = "softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def train_model(latent_dim, batch_size, epochs):
    '''
    train the model using preprocessed data
    '''
    data = get_data()
    en_i, de_i, de_o = data["en-i"], data["de-i"], data["de-o"]
    model = model(latent_dim)
    model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy")
    model.fit([en_i, de_i], de_o, batch_size = batch_size, epochs = epochs, validation_split = 0.2)
    model.save('s2s.h5')
    return model

if __name__ == "__main__":
    pass



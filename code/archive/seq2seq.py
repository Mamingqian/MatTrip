from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, Conv1D, Dropout, Flatten, MaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
import numpy as np
import data
import time

#------------------------------------------BEGIN-------------------------------------------
batch_size = 64
epochs = 100
latent_dim = 100
category_num = 6
embed_dim = 20

input_interest = data.get_interest()
target_routes = data.get_routes()
input_vectors = data.combine(input_interest, target_routes)

'''all poi in consideration'''
target_pois = set()
for route in target_routes:
    for poi in route:
        if poi not in target_pois:
            target_pois.add(poi)
target_pois = sorted(list(target_pois))
num_decoder_tokens = len(target_pois) #31

max_route_length = max([len(route) for route in target_routes]) #22

'''mapping from poi number to index'''
target_poi_index = dict([(poi, i) for i, poi in enumerate(target_pois)])

encoder_input_data = np.zeros((len(input_vectors), category_num + 2, 1), dtype=np.float32)
decoder_input_data = np.zeros((len(input_vectors), max_route_length, num_decoder_tokens), dtype=np.float32)
decoder_target_data = np.zeros((len(input_vectors), max_route_length, num_decoder_tokens), dtype=np.float32)





# print(encoder_input_data.shape)
# print(decoder_input_data.shape)
# print(decoder_target_data.shape)

'''(3000, 8, 1)'''
for i, user_interest in enumerate(input_vectors):
    for t, category in enumerate(user_interest):
        encoder_input_data[i, t, 0] = category

'''turn to one_hot vector'''
'''(3000,22,31)'''
for i, target_route in enumerate(target_routes):
    for t, poi in enumerate(target_route):
        decoder_input_data[i, t, target_poi_index[poi]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_poi_index[poi]] = 1.0
#-------------------------------------------END--------------------------------------------


#------------------------------------------BEGIN-------------------------------------------
'''
batch_size = 64
epochs = 100
latent_dim = 100
category_num = 6
embed_dim = 20
'''

encoder_inputs = Input(shape=(category_num + 2, 1))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape = (None, num_decoder_tokens))

decoder = LSTM(latent_dim, return_sequences = True, return_state = True, dropout=0.2, recurrent_dropout=0.5)

# decoder_outputs, _, _ = decoder(decoder_inputs, initial_state = encoder_states)
decoder_outputs = decoder(decoder_inputs, initial_state= encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation = "softmax")
decoder_outputs = decoder_dense(decoder_outputs)
#-------------------------------------------END--------------------------------------------


#------------------------------------------BEGIN-------------------------------------------
encoder_inputs = Input(shape=(category_num + 2, 1))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape = (None, num_decoder_tokens))
decoder = LSTM(latent_dim, return_sequences = True, return_state = True, dropout=0.2, recurrent_dropout=0.5)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state = encoder_states)

decoder_outputs = decoder(decoder_inputs, initial_state= encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation = "softmax")
decoder_outputs = decoder_dense(decoder_outputs)
#-------------------------------------------END--------------------------------------------
inputs = Input(shape = (None))
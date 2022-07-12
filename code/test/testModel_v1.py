from data import *
from model import *
from train import *
from generation import *

# # GRU version
# # Training the Model
# hidden_size = 128
# encoder = EncoderRNN(1, hidden_size)
# decoder = DecoderRNN(hidden_size, 30)
# trainIters(encoder, decoder, 10000)

# # Test a sample
# index, en_i, de_i, de_o = trainingExample(35)
# originalRoute = routesData[index]
# sampleTargetRoute = BSRoute(encoder, decoder, en_i)
# print("Original Route: ", originalRoute)
# print("Generated Route: ", sampleTargetRoute)

# # Test time and length
# testLength(encoder, decoder)
# testTime(encoder, decoder)

# # Test constraint BSRoute
# outDoorPOIs = fakeOutDoorPOIs(0.3)
# sampleTargetRoute = ConsBSRoute(encoder, decoder, en_i, outDoorPOIs, isRain=True)
# print("Original Route: ", originalRoute)
# print("OutDoorPOIs: ", outDoorPOIs)
# print("Generated Route: ", sampleTargetRoute)

# # Test Recalculation
# path = [6, 29, 15, 23, 27, 13, 22]
# sampleTargetRoute = ReBSRoute(encoder, decoder, en_i, path)
# print("Original Route: ", originalRoute)
# print("Generated Route: ", sampleTargetRoute)





# Bi-LSTM version
# Training the Model
encoder = BiEncoderRNN(input_size = 1, hidden_size = 128, seq_len = 8)

index, en_i, de_i, de_o = trainingExample(35)
originalRoute = routesData[index]


# encoder_output = encoder_output.view(-1, 2 * hidden_size)
# seq_len = encoder_output.shape[0]
# hidden = init_hidden.view(-1, hidden_size)
# hidden = hidden.repeat(seq_len, 1)
# cat_hidden = torch.cat([hidden, encoder_output], dim = 1)
# fc = nn.Linear(3*hidden_size, 1)
# cat_hidden = fc(cat_hidden)
# cat_hidden = cat_hidden.flatten()
# softmax = nn.Softmax(dim=0)
# weights = softmax(cat_hidden)

decoder = AttentionDecoderRNN(hidden_size = 128, output_size = 30)

# forward
init_hidden = decoder.initHidden()
init_cell = decoder.initCell()
encoder_output = encoder(en_i) # 8*1*256
input = de_o[0]

hidden_size = 128
output_size = 30
encoder_output = encoder_output.view(-1, 2 * hidden_size)
seq_len = encoder_output.shape[0]

hidden = init_hidden
hidden = hidden.view(-1, hidden_size)
rep_hidden = hidden.repeat(seq_len, 1)
cat_hidden = torch.cat([rep_hidden, encoder_output], dim = 1)

fc = nn.Linear(3*hidden_size, 1)
softmax = nn.Softmax(dim=0)
fc2 = nn.Linear(hidden_size, output_size)

cat_hidden = fc(cat_hidden).flatten()
weights = softmax(cat_hidden).view(1,-1)

context = torch.mm(weights, encoder_output)
input = torch.cat([input.view(1,-1), context], dim = 1)

lstm = nn.LSTMCell(2*hidden_size+output_size, hidden_size, bias=True)
(next_hidden, next_cell) = lstm(input, (hidden, init_cell.view(1,-1)))
output = fc2(next_hidden).flatten()
output = softmax(output)

input = de_o[0]
output2, (next_hidden, next_cell) = decoder(input, init_hidden, init_cell, encoder_output)
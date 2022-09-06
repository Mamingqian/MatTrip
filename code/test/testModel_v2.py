from data import *
from model import *
from train import *
from generation import *
from evaluation import *

# Attention version
# Training the Model
"""
hidden_size = 128
encoder = BiEncoderRNN(input_size = 1, hidden_size = 128, seq_len = 8)
decoder = AttentionDecoderRNN(hidden_size = 128, output_size = 30)
BiTrainIters(encoder, decoder, 5000)

torch.save(encoder, "encoder.pkl")
torch.save(decoder, "decoder.pkl")
"""

# straightly load model
encoder = torch.load("result/encoder.pkl")
decoder = torch.load("result/decoder.pkl")

index = 45
_, en_i, _, _ = trainingExample(45)
route = routesData[index]
recRoute = greedyRoute(encoder, decoder, en_i, isBi=True)
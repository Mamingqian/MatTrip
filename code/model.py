import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN_GRU(nn.Module):
    '''
    普通的encoder, 使用gru单元, 作为baseline使用
    init: input_size->1, hidden_size->一般128
    input : input -> (1,1,input_size), hidden -> (1,1,hidden_size)
    output: output, hidden -> (1,1,hidden_size)
    '''
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        input = input.view(1,1,-1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size)


class EncoderRNN_LSTM(nn.Module):
    '''
    普通的encoder, 使用lstm单元, 作为baseline使用
    init: input_size->1, hidden_size->一般128
    input : input -> (1,1,input_size), hidden -> ((1,1,hidden_size), (1,1,hidden_size))
    output: output -> (1,1,hidden_size), hidden -> ((1,1,hidden_size), (1,1,hidden_size))
    '''
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input.view(1,1,-1), hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size),torch.zeros(1, 1, self.hidden_size))


class DecoderRNN_GRU(nn.Module):
    '''
    普通的decoder, gru-fc-logSoftmax, 作为baseline使用
    init: hidden_size->一般128, output_size->numPOI
    input : input -> (1,1,output_size), hidden -> (1,1,hidden_size)
    output: output -> (1,1,output_size), hidden -> (1,1,hidden_size)
    '''
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)
        
    def forward(self, input, hidden):
        output, hidden = self.gru(input.view(1,1,-1), hidden.view(1,1,-1))
        output = self.fc(output)
        output = self.softmax(output.flatten())
        return output.view(1,1,-1), hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN_LSTM(nn.Module):
    '''
    普通的decoder, lstm-fc-logSoftmax, 作为baseline使用
    init: hidden_size->一般128, output_size->numPOI
    input : input -> (1,1,output_size), hidden -> ((1,1,hidden_size), (1,1,hidden_size))
    output: output -> (1,1,output_size), hidden -> ((1,1,hidden_size), (1,1,hidden_size))
    '''
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)
        
    def forward(self, input, hidden, cell):
        output, (next_hidden, next_cell) = self.lstm(input.view(1,1,-1), (hidden,cell))
        output = self.fc(output)
        output = self.softmax(output.flatten())
        return output.view(1,1,-1), (next_hidden, next_cell)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size)





class BiEncoderRNN(nn.Module):
    """
    改进Encoder, 双向Bi-LSTM, 文章采取这种方法
    init: input_size -> 1, hidden_size -> 一般128, seq_len -> numCate+2
    input: input -> (seq_len, 1, input_size - 1)
    output -> (seq_len, 1, 2*hidden_size)
    """
    def __init__(self, input_size, hidden_size, seq_len, num_layers = 3):
        super(BiEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, bias=True)

    def forward(self, input):
        h0 = self.__initHidden()
        c0 = self.__initCell()
        input = input.view(self.seq_len,1,-1)
        output, _= self.lstm(input, (h0,c0))
        return output   # seq_len * 1 * (2 * hidden_size)

    def __initHidden(self):
        return torch.zeros(2*self.lstm.num_layers,1,self.hidden_size)

    def __initCell(self):
        return torch.zeros(2*self.lstm.num_layers,1,self.hidden_size)

class BiEncoderRNNwithFC(nn.Module):
    """
    改进Encoder, 双向Bi-LSTM, 文章采取这种方法
    init: input_size -> 1, hidden_size -> 一般128, seq_len -> numCate+2
    input: input -> (seq_len, 1, input_size - 1)
    output -> (seq_len, 1, 2*hidden_size)
    """
    def __init__(self, input_size, hidden_size,seq_len, num_layers = 3):
        super(BiEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, bias=True)

    def forward(self, input):
        h0 = self.__initHidden()
        c0 = self.__initCell()
        input = input.view(self.seq_len,1,-1)
        output, _= self.lstm(input, (h0,c0))
        return output   # seq_len * 1 * (2 * hidden_size)

    def __initHidden(self):
        return torch.zeros(2*self.lstm.num_layers,1,self.hidden_size)

    def __initCell(self):
        return torch.zeros(2*self.lstm.num_layers,1,self.hidden_size)

class AttentionDecoderRNN(nn.Module):
    """
    改进Decoder, 添加Attention机制, 文章采取这种方法
    init: hidden_size->一般128, output_size->numPOI, num_layers
    input: input -> (1, output_size - 30), hidden/cell -> (num_layers, 1, hidden_size)
           encoder_output -> (seq_len, 1, 2*hidden_size)
    output: output -> (output_size)
    
    参考:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    https://github.com/NELSONZHAO/zhihu/tree/master/mt_attention_birnn
    """
    def __init__(self, hidden_size, output_size, num_layers = 1):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(2*hidden_size+output_size, hidden_size, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * (num_layers+2), 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden, cell, encoder_output):
        # reshape encoder_ouput, input, hidden
        encoder_output = encoder_output.view(-1, 2 * self.hidden_size) #[seq_len, 2*hidden_size]
        reHidden = hidden.view(-1, self.hidden_size * self.num_layers)
        seq_len = encoder_output.shape[0]

        # repeat prev hidden, concat prev hidden and encoder ouput
        rep_hidden = reHidden.repeat(seq_len, 1) # rep_hidden shape: (seq_len, hidden_size * num_layers)
        cat_hidden = torch.cat([rep_hidden, encoder_output], dim = 1) # [seq_len, hidden_size * (num_layers+2)]

        # dense and softmax layer to calculate weight alpha
        weights = self.mlp(cat_hidden)
        weights = F.softmax(weights.flatten(), dim=0).view(1,-1) # [1, seq_len]
        
        # get context vector
        context = torch.mm(weights, encoder_output) # matrix mul, [1, 2*hidden_size]

        # concat input and context vector to get input vector
        # input shape: 1, 2*hidden + output_size
        input = torch.cat([input.view(1,-1), context], dim = 1)
        # input = F.relu(input)

        # put into lstm cell
        output, (next_hidden, next_cell) = self.lstm(input.view(1,1,-1), (hidden.view(self.num_layers,1,-1), cell.view(self.num_layers,1,-1)))
        output = self.fc(output)
        output = F.log_softmax(output.flatten(), dim = 0)

        return output, (next_hidden, next_cell)

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

    def initCell(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


class MLP(nn.Module):
    '''
    基本的MLP模型
    input -> (1, 1, inputSize)
    output -> (outputSize)
    '''
    def __init__(self, inputSize, outputSize, layers = [64,32,16]):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential()

        # inputSize -> layers[0]
        self.mlp.add_module("Linear_0", nn.Linear(inputSize, layers[0]))
        self.mlp.add_module("Relu_0", nn.ReLU(inplace=True))

        # layers[0] -> layers[-1]
        for li in range(1, len(layers)):
            self.mlp.add_module("Linear_%d" % li, nn.Linear(layers[li - 1], layers[li]))
            self.mlp.add_module("Relu_%d" % li, nn.ReLU(inplace=True))

        # layers[-1] -> outputSize
        self.mlp.add_module("Linear_output", nn.Linear(layers[-1], outputSize))
        self.softmax = nn.LogSoftmax(dim = 0)


    def forward(self, Input):
        output = self.mlp(Input.view(1,1,-1))
        output = self.softmax(output.flatten())
        return output




class MLP_Linear(nn.Module):
    '''
    基本的MLP模型
    input -> (1, 1, inputSize)
    output -> (outputSize)
    '''
    def __init__(self, inputSize, outputSize, layers = [64,32,16]):
        super(MLP_Linear, self).__init__()
        self.mlp = nn.Sequential()

        # inputSize -> layers[0]
        self.mlp.add_module("Linear_0", nn.Linear(inputSize, outputSize))


    def forward(self, Input):
        output = self.mlp(Input.view(1,1,-1))
        return output



if __name__ == "__main__":
    # gruEncoder
    gruEncoderUnit = EncoderRNN_GRU(input_size = 8, hidden_size = 128)
    input = torch.randn([1,1,8])
    output, hidden = gruEncoderUnit(input, gruEncoderUnit.initHidden())

    # lstmEncoder
    lstmEncoderUnit = EncoderRNN_LSTM(input_size = 8, hidden_size = 128)
    input = torch.randn([1,1,8])
    output, hidden = lstmEncoderUnit(input, lstmEncoderUnit.initHidden())

    # gruDecoder
    gruDecoderUnit = DecoderRNN_GRU(hidden_size = 128, output_size = 30)
    input = torch.randn([1,1,30])
    output, hidden = gruDecoderUnit(input, gruDecoderUnit.initHidden())    

    # lstmDecoder
    lstmDecoderUnit = DecoderRNN_LSTM(hidden_size = 128, output_size = 30)
    input = torch.randn([1,1,30])
    output, hidden = lstmDecoderUnit(input, lstmDecoderUnit.initHidden())    

    # biEncoder
    biEncoder = BiEncoderRNN(input_size = 1, hidden_size = 128, seq_len = 8)
    input = torch.randn([8,1,1])
    encoderOutput = biEncoder(input)

    # attnDecoder
    attnDecoder = AttentionDecoderRNN(hidden_size = 128, output_size = 30)
    input = torch.randn([1,1,30])
    output, (nextHidden, nextCell) = attnDecoder(input, attnDecoder.initHidden(), attnDecoder.initCell(), encoderOutput)
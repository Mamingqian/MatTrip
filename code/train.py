import torch
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import time
import math

from data import *
from model import *

class LT_Trainer():
    def __init__(self, encoder, decoder, cityData):
        self.encoder = encoder
        self.decoder = decoder
        self.cityData = cityData

    def train(self, n_iters, print_every=1000, plot_every=200, learning_rate = 0.01):
        '''
        训练LT模型, 作为baseline
        '''
        current_loss_print = 0
        current_loss_plot = 0
        all_losses = []

        en_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        de_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        start = time.time()
        for iter in range(1, n_iters + 1):
            index, en_i, de_i, de_o = self.cityData.trainingExample()
            loss = self.__train(en_i, de_i, de_o, en_optimizer, de_optimizer)

            current_loss_print += loss
            if iter % print_every == 0:
                print('%d %d%%'%(iter, iter / n_iters * 100), ", Time: ", self.__timeSince(start), ", Loss: ", current_loss_print/print_every)
                current_loss_print = 0

            current_loss_plot += loss
            if iter % plot_every == 0:
                all_losses.append(current_loss_plot / plot_every)
                current_loss_plot = 0
        
        plt.figure()
        plt.plot(all_losses)

    def __train(self, en_i, de_i, de_o, en_optimizer, de_optimizer):
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()

        input_length = en_i.shape[0] # 8
        target_length = de_o.shape[0] # ?
        loss = 0

        # encoder forward
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(input_length, self.encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(en_i[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output.reshape(-1)

        # decoder forward
        
        use_teacher_forcing = True if random.random() < 0.5 else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(de_i[di], decoder_hidden)
                loss += self.__Loss(decoder_output, de_o[di])
        else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_input = de_i[0]
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # Create new decoder input
                poi = self.cityData.tensor2poi(decoder_output)
                decoder_input = self.cityData.poi2tensor(poi)
                decoder_input = decoder_input.detach()

                loss += self.__Loss(decoder_output, de_o[di])
                # if poi == self.cityData.tensor2poi(de_o[-1]):
                #     break

        loss.backward()
        en_optimizer.step()
        de_optimizer.step()

        return loss.item() / target_length

    # helper functions
    def __timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def __Loss(self, prediction, target):
        '''
        损失函数采取nllloss, 对应logsoftmax
        '''
        criterion = nn.NLLLoss()
        prediction = prediction.view(1,-1)
        _, targetNum = target.squeeze().topk(1)
        return criterion(prediction, targetNum)

class Trainer():
    def __init__(self, encoder, decoder, cityData):
        self.encoder = encoder
        self.decoder = decoder
        self.cityData = cityData

    def train(self, n_iters, print_every=1000, plot_every=200, learning_rate = 0.01):
        current_loss_print = 0
        current_loss_plot = 0
        all_losses = []

        en_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        de_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        start = time.time()
        for iter in range(1, n_iters + 1):
            index, en_i, de_i, de_o = self.cityData.trainingExample()
            loss = self.__train(en_i, de_i, de_o, en_optimizer, de_optimizer)
            
            current_loss_print += loss
            if iter % print_every == 0:
                print('%d %d%%'%(iter, iter / n_iters * 100), ", Time: ", self.__timeSince(start), ", Loss: ", current_loss_print / print_every)
                current_loss_print = 0

            current_loss_plot += loss
            if iter % plot_every == 0:
                all_losses.append(current_loss_plot / plot_every)
                current_loss_plot = 0
        
        plt.figure()
        plt.plot(all_losses)

    # Training the network, BiLSTM and Attention version
    def __train(self, en_i, de_i, de_o, en_optimizer, de_optimizer):
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()

        input_length = en_i.shape[0] # 8
        target_length = de_o.shape[0] # ?
        loss = 0

        # encoder forward
        encoder_output = self.encoder(en_i)  # seq_len * 1 * (2 * hidden_size)

        # decoder forward
        decoder_hidden = self.decoder.initHidden()
        decoder_cell = self.decoder.initCell()
        use_teacher_forcing = True if random.random() < 0.5 else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(de_i[di], decoder_hidden, decoder_cell, encoder_output)
                loss += self.__Loss(decoder_output, de_o[di])
        else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_input = de_i[0]
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
                # Create new decoder input
                poi = self.cityData.tensor2poi(decoder_output)
                decoder_input = self.cityData.poi2tensor(poi)
                decoder_input = decoder_input.detach()

                loss += self.__Loss(decoder_output, de_o[di])
                if poi == self.cityData.tensor2poi(de_o[-1]):
                    break

        loss.backward()
        en_optimizer.step()
        de_optimizer.step()

        return loss.item() / target_length

    # helper functions
    def __timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def __Loss(self, prediction, target):
        '''
        损失函数采取nllloss, 对应logsoftmax
        '''
        criterion = nn.NLLLoss()
        prediction = prediction.view(1,-1)
        _, targetNum = target.squeeze().topk(1)
        return criterion(prediction, targetNum)

class Trainer_v2:
    def __init__(self, prefEncoder, posEncoder, decoder, cityData):
        self.prefEncoder = prefEncoder
        self.posEncoder = posEncoder
        self.decoder = decoder
        # self.mlp = mlp

        self.cityData = cityData

    def train(self, n_iters, print_every=1000, plot_every=200, learning_rate = 0.01):
        current_loss_print = 0
        current_loss_plot = 0
        all_losses = []

        pref_en_optimizer = optim.Adam(self.prefEncoder.parameters(), lr=learning_rate)
        pos_en_optimizer = optim.Adam(self.posEncoder.parameters(), lr=learning_rate)
        # mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)
        de_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        start = time.time()
        for iter in range(1, n_iters + 1):
            index, pref_i, pos_i, de_i, de_o = self.cityData.trainingExample_v2()
            loss = self.__train(pref_i, pos_i, de_i, de_o, [pref_en_optimizer, pos_en_optimizer, de_optimizer])
            
            current_loss_print += loss
            if iter % print_every == 0:
                print('%d %d%%'%(iter, iter / n_iters * 100), ", Time: ", self.__timeSince(start), ", Loss: ", current_loss_print / print_every)
                current_loss_print = 0

            current_loss_plot += loss
            if iter % plot_every == 0:
                all_losses.append(current_loss_plot / plot_every)
                current_loss_plot = 0
        
        plt.figure()
        plt.plot(all_losses)

    # Training the network, BiLSTM and Attention version
    def __train(self, pref_i, pos_i, de_i, de_o, optimizers):
        for opt in optimizers:
            opt.zero_grad()

        fref_length = pref_i.shape[0] # 6
        pos_length = pos_i.shape[0] # 6
        target_length = de_o.shape[0] # ?

        loss = 0

        # encoder forward
        pref_encoder_output = self.prefEncoder(pref_i)  # seq_len * 1 * (2 * hidden_size)
        pos_encoder_output = self.posEncoder(pos_i)  # seq_len * 1 * (2 * hidden_size)

        # fusion
        encoder_output = torch.cat([pref_encoder_output, pos_encoder_output], dim=0)

        # decoder forward
        decoder_hidden = self.decoder.initHidden()
        decoder_cell = self.decoder.initCell()
        use_teacher_forcing = True if random.random() < 0.5 else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(de_i[di], decoder_hidden, decoder_cell, encoder_output)
                loss += self.__Loss(decoder_output, de_o[di])
        else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_input = de_i[0]
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
                # Create new decoder input
                poi = self.cityData.tensor2poi(decoder_output)
                decoder_input = self.cityData.poi2tensor(poi)
                decoder_input = decoder_input.detach()

                loss += self.__Loss(decoder_output, de_o[di])
                if poi == self.cityData.tensor2poi(de_o[-1]):
                    break

        loss.backward()
        for opt in optimizers:
            opt.step()

        return loss.item() / target_length

    # helper functions
    def __timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def __Loss(self, prediction, target):
        '''
        损失函数采取nllloss, 对应logsoftmax
        '''
        criterion = nn.NLLLoss()
        prediction = prediction.view(1,-1)
        _, targetNum = target.squeeze().topk(1)
        return criterion(prediction, targetNum)

class Trainer_pref:
    def __init__(self, prefEncoder, decoder, cityData):
        self.prefEncoder = prefEncoder
        self.decoder = decoder
        # self.mlp = mlp

        self.cityData = cityData

    def train(self, n_iters, print_every=1000, plot_every=200, learning_rate = 0.01):
        current_loss_print = 0
        current_loss_plot = 0
        all_losses = []

        pref_en_optimizer = optim.Adam(self.prefEncoder.parameters(), lr=learning_rate)
        # mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)
        de_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        start = time.time()
        for iter in range(1, n_iters + 1):
            index, pref_i, pos_i, de_i, de_o = self.cityData.trainingExample_v2()
            loss = self.__train(pref_i, pos_i, de_i, de_o, [pref_en_optimizer, de_optimizer])
            
            current_loss_print += loss
            if iter % print_every == 0:
                print('%d %d%%'%(iter, iter / n_iters * 100), ", Time: ", self.__timeSince(start), ", Loss: ", current_loss_print / print_every)
                current_loss_print = 0

            current_loss_plot += loss
            if iter % plot_every == 0:
                all_losses.append(current_loss_plot / plot_every)
                current_loss_plot = 0
        
        plt.figure()
        plt.plot(all_losses)

    # Training the network, BiLSTM and Attention version
    def __train(self, pref_i, pos_i, de_i, de_o, optimizers):
        for opt in optimizers:
            opt.zero_grad()

        fref_length = pref_i.shape[0] # 6
        target_length = de_o.shape[0] # ?

        loss = 0

        # encoder forward
        encoder_output = self.prefEncoder(pref_i)  # seq_len * 1 * (2 * hidden_size)


        # decoder forward
        decoder_hidden = self.decoder.initHidden()
        decoder_cell = self.decoder.initCell()
        use_teacher_forcing = True if random.random() < 0.5 else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(de_i[di], decoder_hidden, decoder_cell, encoder_output)
                loss += self.__Loss(decoder_output, de_o[di])
        else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_input = de_i[0]
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
                # Create new decoder input
                poi = self.cityData.tensor2poi(decoder_output)
                decoder_input = self.cityData.poi2tensor(poi)
                decoder_input = decoder_input.detach()

                loss += self.__Loss(decoder_output, de_o[di])
                if poi == self.cityData.tensor2poi(de_o[-1]):
                    break

        loss.backward()
        for opt in optimizers:
            opt.step()

        return loss.item() / target_length

    # helper functions
    def __timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def __Loss(self, prediction, target):
        '''
        损失函数采取nllloss, 对应logsoftmax
        '''
        criterion = nn.NLLLoss()
        prediction = prediction.view(1,-1)
        _, targetNum = target.squeeze().topk(1)
        return criterion(prediction, targetNum)

class Trainer_pos:
    def __init__(self, posEncoder, decoder, cityData):
        self.posEncoder = posEncoder
        self.decoder = decoder
        # self.mlp = mlp

        self.cityData = cityData

    def train(self, n_iters, print_every=1000, plot_every=200, learning_rate = 0.01):
        current_loss_print = 0
        current_loss_plot = 0
        all_losses = []

        pos_en_optimizer = optim.Adam(self.posEncoder.parameters(), lr=learning_rate)
        # mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)
        de_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        start = time.time()
        for iter in range(1, n_iters + 1):
            index, pref_i, pos_i, de_i, de_o = self.cityData.trainingExample_v2()
            loss = self.__train(pref_i, pos_i, de_i, de_o, [pos_en_optimizer, de_optimizer])
            
            current_loss_print += loss
            if iter % print_every == 0:
                print('%d %d%%'%(iter, iter / n_iters * 100), ", Time: ", self.__timeSince(start), ", Loss: ", current_loss_print / print_every)
                current_loss_print = 0

            current_loss_plot += loss
            if iter % plot_every == 0:
                all_losses.append(current_loss_plot / plot_every)
                current_loss_plot = 0
        
        plt.figure()
        plt.plot(all_losses)

    # Training the network, BiLSTM and Attention version
    def __train(self, pref_i, pos_i, de_i, de_o, optimizers):
        for opt in optimizers:
            opt.zero_grad()

        target_length = de_o.shape[0] # ?

        loss = 0

        # encoder forward
        encoder_output = self.posEncoder(pos_i)  # seq_len * 1 * (2 * hidden_size)

        # decoder forward
        decoder_hidden = self.decoder.initHidden()
        decoder_cell = self.decoder.initCell()
        use_teacher_forcing = True if random.random() < 0.5 else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(de_i[di], decoder_hidden, decoder_cell, encoder_output)
                loss += self.__Loss(decoder_output, de_o[di])
        else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_input = de_i[0]
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
                # Create new decoder input
                poi = self.cityData.tensor2poi(decoder_output)
                decoder_input = self.cityData.poi2tensor(poi)
                decoder_input = decoder_input.detach()

                loss += self.__Loss(decoder_output, de_o[di])
                if poi == self.cityData.tensor2poi(de_o[-1]):
                    break

        loss.backward()
        for opt in optimizers:
            opt.step()

        return loss.item() / target_length

    # helper functions
    def __timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def __Loss(self, prediction, target):
        '''
        损失函数采取nllloss, 对应logsoftmax
        '''
        criterion = nn.NLLLoss()
        prediction = prediction.view(1,-1)
        _, targetNum = target.squeeze().topk(1)
        return criterion(prediction, targetNum)

class Trainer_join:
    def __init__(self, encoder, decoder, cityData):
        self.encoder = encoder
        self.decoder = decoder
        # self.mlp = mlp

        self.cityData = cityData

    def train(self, n_iters, print_every=1000, plot_every=200, learning_rate = 0.01):
        current_loss_print = 0
        current_loss_plot = 0
        all_losses = []

        pos_en_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        # mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)
        de_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        start = time.time()
        for iter in range(1, n_iters + 1):
            index, pref_i, pos_i, de_i, de_o = self.cityData.trainingExample_v2()
            en_i = torch.cat([pref_i, pos_i], dim=0)
            loss = self.__train(en_i, de_i, de_o, [pos_en_optimizer, de_optimizer])
            
            current_loss_print += loss
            if iter % print_every == 0:
                print('%d %d%%'%(iter, iter / n_iters * 100), ", Time: ", self.__timeSince(start), ", Loss: ", current_loss_print / print_every)
                current_loss_print = 0

            current_loss_plot += loss
            if iter % plot_every == 0:
                all_losses.append(current_loss_plot / plot_every)
                current_loss_plot = 0
        
        plt.figure()
        plt.plot(all_losses)

    # Training the network, BiLSTM and Attention version
    def __train(self, en_i, de_i, de_o, optimizers):
        for opt in optimizers:
            opt.zero_grad()

        target_length = de_o.shape[0] # ?
        loss = 0

        # encoder forward
        encoder_output = self.encoder(en_i)  # seq_len * 1 * (2 * hidden_size)

        # decoder forward
        decoder_hidden = self.decoder.initHidden()
        decoder_cell = self.decoder.initCell()
        use_teacher_forcing = True if random.random() < 0.5 else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(de_i[di], decoder_hidden, decoder_cell, encoder_output)
                loss += self.__Loss(decoder_output, de_o[di])
        else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_input = de_i[0]
            for di in range(target_length):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
                # Create new decoder input
                poi = self.cityData.tensor2poi(decoder_output)
                decoder_input = self.cityData.poi2tensor(poi)
                decoder_input = decoder_input.detach()

                loss += self.__Loss(decoder_output, de_o[di])
                if poi == self.cityData.tensor2poi(de_o[-1]):
                    break

        loss.backward()
        for opt in optimizers:
            opt.step()

        return loss.item() / target_length

    # helper functions
    def __timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def __Loss(self, prediction, target):
        '''
        损失函数采取nllloss, 对应logsoftmax
        '''
        criterion = nn.NLLLoss()
        prediction = prediction.view(1,-1)
        _, targetNum = target.squeeze().topk(1)
        return criterion(prediction, targetNum)

class Trainer_na:
    def __init__(self, prefEncoder, posEncoder, mlp, decoder, cityData):
        self.prefEncoder = prefEncoder
        self.posEncoder = posEncoder
        self.decoder = decoder
        self.mlp = mlp

        self.cityData = cityData

    def train(self, n_iters, print_every=1000, plot_every=200, learning_rate = 0.01):
        current_loss_print = 0
        current_loss_plot = 0
        all_losses = []

        pref_en_optimizer = optim.Adam(self.prefEncoder.parameters(), lr=learning_rate)
        pos_en_optimizer = optim.Adam(self.posEncoder.parameters(), lr=learning_rate)
        mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)
        de_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        start = time.time()
        for iter in range(1, n_iters + 1):
            index, pref_i, pos_i, de_i, de_o = self.cityData.trainingExample_v2()
            loss = self.__train(pref_i, pos_i, de_i, de_o, [pref_en_optimizer, pos_en_optimizer, mlp_optimizer, de_optimizer])
            
            current_loss_print += loss
            if iter % print_every == 0:
                print('%d %d%%'%(iter, iter / n_iters * 100), ", Time: ", self.__timeSince(start), ", Loss: ", current_loss_print / print_every)
                current_loss_print = 0

            current_loss_plot += loss
            if iter % plot_every == 0:
                all_losses.append(current_loss_plot / plot_every)
                current_loss_plot = 0
        
        plt.figure()
        plt.plot(all_losses)

    # Training the network, BiLSTM and Attention version
    def __train(self, pref_i, pos_i, de_i, de_o, optimizers):
        for opt in optimizers:
            opt.zero_grad()

        fref_length = pref_i.shape[0] # 6
        pos_length = pos_i.shape[0] # 6
        target_length = de_o.shape[0] # ?

        loss = 0

        # encoder forward
        pref_encoder_output = self.prefEncoder(pref_i)  # seq_len * 1 * (2 * hidden_size)
        pos_encoder_output = self.posEncoder(pos_i)  # seq_len * 1 * (2 * hidden_size)

        # fusion
        encoder_output = torch.cat([pref_encoder_output, pos_encoder_output], dim=0) #12*1*(2*hidden_size)

        # decoder forward
        encoder_output = self.mlp(encoder_output)

        decoder_h = self.decoder.initHidden()
        decoder_c = encoder_output.view(1,1,-1)

        use_teacher_forcing = True if random.random() < 0.5 else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, (decoder_h, decoder_c) = self.decoder(de_i[di], decoder_h, decoder_c)
                loss += self.__Loss(decoder_output, de_o[di])
        else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_input = de_i[0]
            for di in range(target_length):
                decoder_output, (decoder_h, decoder_c) = self.decoder(decoder_input, decoder_h, decoder_c)
                # Create new decoder input
                poi = self.cityData.tensor2poi(decoder_output)
                decoder_input = self.cityData.poi2tensor(poi)
                decoder_input = decoder_input.detach()

                loss += self.__Loss(decoder_output, de_o[di])
                # if poi == self.cityData.tensor2poi(de_o[-1]):
                #     break

        loss.backward()
        for opt in optimizers:
            opt.step()

        return loss.item() / target_length

    # helper functions
    def __timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def __Loss(self, prediction, target):
        '''
        损失函数采取nllloss, 对应logsoftmax
        '''
        criterion = nn.NLLLoss()
        prediction = prediction.view(1,-1)
        _, targetNum = target.squeeze().topk(1)
        return criterion(prediction, targetNum)





if __name__ == "__main__":
    trt = CityData("Toronto")
    osak = CityData("Osaka")

    # 1.1. Learning Tour
    LTEncoder_trt = EncoderRNN_LSTM(input_size = 1, hidden_size = 128)
    LTDecoder_trt = DecoderRNN_LSTM(hidden_size = 128, output_size = trt.numPOI)
    lt_trainer_trt = LT_Trainer(LTEncoder_trt, LTDecoder_trt, trt)
    lt_trainer_trt.train(30000, learning_rate=0.005)
    torch.save(lt_trainer_trt.encoder, "result/LT_Encoder_trt.pkl")
    torch.save(lt_trainer_trt.decoder, "result/LT_Decoder_trt.pkl")

    # # 1.2. v1
    # Encoder_trt = BiEncoderRNN(input_size = 1, hidden_size = 128, seq_len=trt.numCategory+2, num_layers=2)
    # Decoder_trt = AttentionDecoderRNN(hidden_size = 128, output_size = trt.numPOI, num_layers=1)
    # trainer_trt = Trainer(Encoder_trt, Decoder_trt, trt)
    # trainer_trt.train(20000, learning_rate=0.01)

    # 1.3. v2
    Pref_encoder_trt = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=trt.numCategory, num_layers=1)
    Pos_encoder_trt = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=6, num_layers=1)
    Decoder_trt_v2 = AttentionDecoderRNN(hidden_size = 128, output_size = trt.numPOI, num_layers=1)
    trainer_trt_v2 = Trainer_v2(Pref_encoder_trt, Pos_encoder_trt, Decoder_trt_v2, trt)
    trainer_trt_v2.train(30000, learning_rate=0.005)
    torch.save(trainer_trt_v2.prefEncoder, "result/v2_PrefEncoder_trt.pkl")
    torch.save(trainer_trt_v2.posEncoder, "result/v2_PosEncoder_trt.pkl")
    torch.save(trainer_trt_v2.decoder, "result/v2_Decoder_trt.pkl")

    # 1.4. pref
    Pref_encoder_trt = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=trt.numCategory, num_layers=1)
    Decoder_trt_v2 = AttentionDecoderRNN(hidden_size = 128, output_size = trt.numPOI, num_layers=1)
    trainer_trt_v2 = Trainer_pref(Pref_encoder_trt, Decoder_trt_v2, trt)
    trainer_trt_v2.train(30000, learning_rate=0.005)
    torch.save(trainer_trt_v2.prefEncoder, "result/pref_PrefEncoder_trt.pkl")
    torch.save(trainer_trt_v2.decoder, "result/pref_Decoder_trt.pkl")

    # 1.5. pos
    pos_encoder_trt = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=6, num_layers=1)
    Decoder_trt_v2 = AttentionDecoderRNN(hidden_size = 128, output_size = trt.numPOI, num_layers=1)
    trainer_trt_v2 = Trainer_pos(pos_encoder_trt, Decoder_trt_v2, trt)
    trainer_trt_v2.train(30000, learning_rate=0.005)
    torch.save(trainer_trt_v2.posEncoder, "result/pos_posEncoder_trt.pkl")
    torch.save(trainer_trt_v2.decoder, "result/pos_Decoder_trt.pkl")

    # 1.6. join
    encoder_trt = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=trt.numCategory+6, num_layers=1)
    Decoder_trt_v2 = AttentionDecoderRNN(hidden_size = 128, output_size = trt.numPOI, num_layers=1)
    trainer_trt_v2 = Trainer_join(encoder_trt, Decoder_trt_v2, trt)
    trainer_trt_v2.train(30000, learning_rate=0.005)
    torch.save(trainer_trt_v2.encoder, "result/join_posEncoder_trt.pkl")
    torch.save(trainer_trt_v2.decoder, "result/join_Decoder_trt.pkl")
    

    # 1.7. na
    encoder_pref = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=trt.numCategory, num_layers=1)
    encoder_pos = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=6, num_layers=1)
    mlp = MLP_Linear(inputSize = (trt.numCategory+6)*2*128, outputSize=128)
    decoder = DecoderRNN_LSTM(hidden_size=128, output_size=trt.numPOI)
    trainer = Trainer_na(encoder_pref, encoder_pos, mlp, decoder,trt)
    trainer.train(30000, learning_rate=0.005)
    torch.save(trainer.posEncoder, "result/na_posEncoder_trt.pkl")
    torch.save(trainer.prefEncoder, "result/na_prefEncoder_trt.pkl")
    torch.save(trainer.mlp, "result/na_mlp_trt.pkl")
    torch.save(trainer.decoder, "result/na_decoder_trt.pkl")






    # Osak


    # 1.1. Learning Tour
    LTEncoder_osak = EncoderRNN_LSTM(input_size = 1, hidden_size = 128)
    LTDecoder_osak = DecoderRNN_LSTM(hidden_size = 128, output_size = osak.numPOI)
    lt_trainer_osak = LT_Trainer(LTEncoder_osak, LTDecoder_osak, osak)
    lt_trainer_osak.train(20000, learning_rate=0.01)
    torch.save(lt_trainer_osak.encoder, "result/LT_Encoder_osak.pkl")
    torch.save(lt_trainer_osak.decoder, "result/LT_Decoder_osak.pkl")

    # # 1.2. v1
    # Encoder_osak = BiEncoderRNN(input_size = 1, hidden_size = 128, seq_len=osak.numCategory+2, num_layers=2)
    # Decoder_osak = AttentionDecoderRNN(hidden_size = 128, output_size = osak.numPOI, num_layers=1)
    # trainer_osak = Trainer(Encoder_osak, Decoder_osak, osak)
    # trainer_osak.train(20000, learning_rate=0.01)

    # 1.3. v2
    Pref_encoder_osak = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=osak.numCategory, num_layers=1)
    Pos_encoder_osak = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=6, num_layers=1)
    Decoder_osak_v2 = AttentionDecoderRNN(hidden_size = 128, output_size = osak.numPOI, num_layers=1)
    trainer_osak_v2 = Trainer_v2(Pref_encoder_osak, Pos_encoder_osak, Decoder_osak_v2, osak)

    trainer_osak_v2.train(30000, learning_rate=0.005)
    torch.save(trainer_osak_v2.prefEncoder, "result/v2_PrefEncoder_osak.pkl")
    torch.save(trainer_osak_v2.posEncoder, "result/v2_PosEncoder_osak.pkl")
    torch.save(trainer_osak_v2.decoder, "result/v2_Decoder_osak.pkl")


    # 1.4. pref
    Pref_encoder_osak = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=osak.numCategory, num_layers=1)
    Decoder_osak_v2 = AttentionDecoderRNN(hidden_size = 128, output_size = osak.numPOI, num_layers=1)
    trainer_osak_v2 = Trainer_pref(Pref_encoder_osak, Decoder_osak_v2, osak)
    trainer_osak_v2.train(30000, learning_rate=0.005)
    torch.save(trainer_osak_v2.prefEncoder, "result/pref_PrefEncoder_osak.pkl")
    torch.save(trainer_osak_v2.decoder, "result/pref_Decoder_osak.pkl")

    # 1.5. pos
    pos_encoder_osak = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=6, num_layers=1)
    Decoder_osak_v2 = AttentionDecoderRNN(hidden_size = 128, output_size = osak.numPOI, num_layers=1)
    trainer_osak_v2 = Trainer_pos(pos_encoder_osak, Decoder_osak_v2, osak)
    trainer_osak_v2.train(30000, learning_rate=0.005)
    torch.save(trainer_osak_v2.posEncoder, "result/pos_posEncoder_osak.pkl")
    torch.save(trainer_osak_v2.decoder, "result/pos_Decoder_osak.pkl")


    # 1.6. join
    encoder_osak = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=osak.numCategory+6, num_layers=1)
    Decoder_osak_v2 = AttentionDecoderRNN(hidden_size = 128, output_size = osak.numPOI, num_layers=1)
    trainer_osak_v2 = Trainer_join(encoder_osak, Decoder_osak_v2, osak)
    trainer_osak_v2.train(30000, learning_rate=0.005)
    torch.save(trainer_osak_v2.encoder, "result/join_posEncoder_osak.pkl")
    torch.save(trainer_osak_v2.decoder, "result/join_Decoder_osak.pkl")

    # 1.7. na
    encoder_pref = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=osak.numCategory, num_layers=1)
    encoder_pos = BiEncoderRNN(input_size=1, hidden_size=128, seq_len=6, num_layers=1)
    mlp = MLP_Linear(inputSize = (osak.numCategory+6)*2*128, outputSize=128)
    decoder = DecoderRNN_LSTM(hidden_size=128, output_size=osak.numPOI)
    trainer = Trainer_na(encoder_pref, encoder_pos, mlp, decoder,osak)
    trainer.train(30000, learning_rate=0.005)
    torch.save(trainer.posEncoder, "result/na_posEncoder_osak.pkl")
    torch.save(trainer.prefEncoder, "result/na_prefEncoder_osak.pkl")
    torch.save(trainer.mlp, "result/na_mlp_osak.pkl")
    torch.save(trainer.decoder, "result/na_decoder_osak.pkl")
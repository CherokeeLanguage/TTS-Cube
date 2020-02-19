import torch
import sys

sys.path.append('')
import numpy as np
from cube2.networks.g2p import G2P
from cube2.networks.modules import Attention

import torch.nn as nn


class Seq2SeqSimplified(nn.Module):
    def __init__(self, s2s):
        super(Seq2SeqSimplified, self).__init__()
        self.emb_size = s2s.emb_size
        self.input_emb = s2s.input_emb  # nn.Embedding(num_input_tokens, embedding_size, padding_idx=pad_index)
        self.output_emb = s2s.output_emb  # nn.Embedding(num_output_tokens, embedding_size, padding_idx=pad_index)
        self.encoder = s2s.encoder  # nn.LSTM(embedding_size, encoder_size, encoder_layers, dropout=0.33, bidirectional=True)
        self.decoder = s2s.decoder  # nn.LSTM(encoder_size * 2 + embedding_size, decoder_size, decoder_layers, dropout=0.33)
        self.attention = s2s.attention  # Attention(encoder_size, decoder_size, att_proj_size=decoder_size)
        self.output = s2s.output  # nn.Linear(decoder_size, num_output_tokens)
        self._PAD = s2s._PAD  # pad_index
        self._UNK = s2s._UNK  # unk_index
        self._EOS = s2s._EOS  # stop_index
        self._dec_input_size = s2s._dec_input_size  # encoder_size * 2 + embedding_size

    def forward(self, x):
        x = self.input_emb(x)
        encoder_output, encoder_hidden = self.encoder(x.permute(1, 0, 2))
        encoder_output = encoder_output.permute(1, 0, 2)
        count = 0

        _, decoder_hidden = self.decoder(torch.zeros((1, x.shape[0], self._dec_input_size)))
        last_output_emb = torch.zeros((x.shape[0], self.emb_size))
        output_list = []
        index = 0
        reached_end = [False for ii in range(x.shape[0])]
        while True:
            _, encoder_att = self.attention(decoder_hidden[-1][-1].unsqueeze(0), encoder_output)
            decoder_input = torch.cat([encoder_att, last_output_emb], dim=1)
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), decoder_hidden)
            output = self.output(decoder_output.squeeze(0))
            output_list.append(output.unsqueeze(1))

            outp = torch.argmax(output, dim=1)
            last_output_emb = self.output_emb(outp)
            for ii in range(outp.shape[0]):
                if outp[ii] == self._EOS:
                    reached_end[ii] = True

            all_done = True
            for done in reached_end:
                if not done:
                    all_done = False
                    break
            if all_done:
                break
            # if np.all(reached_end):
            #    break
            index += 1

            if index > x.shape[1] * 10:
                break

        return torch.cat(output_list, dim=1)


g2p = G2P()
g2p.load('data/models/en-g2p')
g2p.eval()

s2s = Seq2SeqSimplified(g2p.seq2seq)

script_model = torch.jit.script(s2s, torch.tensor([[1,2,3,4],[5,6,7,8]]))
from ipdb import set_trace
set_trace()
script_model.save("data/models/en-g2p.pth")


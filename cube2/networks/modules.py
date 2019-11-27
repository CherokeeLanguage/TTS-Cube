#
# Author: Tiberiu Boros
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn


class UpsampleNet(nn.Module):
    def __init__(self, input_size, output_size, upsample_scales):
        super(UpsampleNet, self).__init__()
        self.upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.ReLU())
            # self.upsample_conv.append(nn.Dropout(0.5))
        self.output_transform = nn.Linear(input_size, output_size)

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x)
            noise = noise * 0.01
            # noise += 1
            x = x + noise
        x = torch.clamp(x, min=0, max=1)
        c = x.permute(0, 2, 1)
        if self.upsample_conv is not None:
            # B x 1 x C x T'
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)
        return self.output_transform(c.permute(0, 2, 1))


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.softmax(torch.bmm(v, energy).squeeze(1), dim=1)

        a = attention.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.squeeze(1)

        return attention, weighted


class Seq2Seq(nn.Module):
    def __init__(self, num_input_tokens, num_output_tokens, embedding_size=100, encoder_size=100, encoder_layers=2,
                 decoder_size=200, decoder_layers=2, pad_index=0, unk_index=1, stop_index=2):
        super(Seq2Seq, self).__init__()
        self.emb_size = embedding_size
        self.input_emb = nn.Embedding(num_input_tokens, embedding_size, padding_idx=pad_index)
        self.output_emb = nn.Embedding(num_output_tokens, embedding_size, padding_idx=pad_index)
        self.encoder = nn.LSTM(embedding_size, encoder_size, encoder_layers, dropout=0.33, bidirectional=True)
        self.decoder = nn.LSTM(encoder_size * 2 + embedding_size, decoder_size, decoder_layers, dropout=0.33)
        self.attention = Attention(encoder_size, decoder_size)
        self.output = nn.Linear(decoder_size, num_output_tokens)
        self._PAD = pad_index
        self._UNK = unk_index
        self._EOS = stop_index
        self._dec_input_size = encoder_size * 2 + embedding_size

    def forward(self, x, gs_output=None):
        # x, y = self._make_batches(input, gs_output)
        x = self.input_emb(x)
        encoder_output, encoder_hidden = self.encoder(x.permute(1, 0, 2))
        encoder_output = encoder_output.permute(1, 0, 2)
        count = 0
        if gs_output is not None:
            batch_output_emb = self.output_emb(gs_output)

        _, decoder_hidden = self.decoder(torch.zeros((1, x.shape[0], self._dec_input_size), device=self._get_device()))
        last_output_emb = torch.zeros((x.shape[0], self.emb_size), device=self._get_device())
        output_list = []
        index = 0
        reached_end = [False for _ in range(x.shape[0])]
        while True:
            _, encoder_att = self.attention(decoder_hidden[-1][-1].unsqueeze(0), encoder_output)
            decoder_input = torch.cat([encoder_att, last_output_emb], dim=1)
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), decoder_hidden)
            output = self.output(decoder_output.squeeze(0))
            output_list.append(output.unsqueeze(1))

            if gs_output is not None:
                last_output_emb = batch_output_emb[:, index, :]
                index += 1
                if index == gs_output.shape[1]:
                    break
            else:
                outp = torch.argmax(output, dim=1)
                last_output_emb = self.output_emb(outp)
                for ii in range(outp.shape[0]):
                    if outp[ii] == self._EOS:
                        reached_end[ii] = True
                import numpy as np
                if np.all(reached_end):
                    break
                index += 1

                if index > x.shape[1] * 10:
                    break

        return torch.cat(output_list, dim=1)

    def _get_device(self):
        if self.input_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.input_emb.weight.device.type, str(self.input_emb.weight.device.index))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


class PostNet(nn.Module):
    def __init__(self, num_mels=80, kernel_size=5, filter_size=512):
        super(PostNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(num_mels, filter_size, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(filter_size, filter_size, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(filter_size, filter_size, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(filter_size, filter_size, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(filter_size, num_mels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_mels)
        )
        for ii in range(len(self.network) // 4):
            torch.nn.init.xavier_uniform_(
                self.network[ii * 4].weight, gain=torch.nn.init.calculate_gain('linear'))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.network(x).permute(0, 2, 1)
        return y

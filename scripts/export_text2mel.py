import torch
import torch.nn as nn
import sys

sys.path.append('')
from cube2.io_modules.dataset import DatasetIO, Encodings
from cube2.networks.text2mel import Text2Mel


class Text2MelSimplified(nn.Module):
    def __init__(self, text2mel):
        super(Text2MelSimplified, self).__init__()
        self.MGC_PROJ_SIZE = text2mel.MGC_PROJ_SIZE
        self.pframes = text2mel.pframes
        self.mgc_order = text2mel.mgc_order
        self.char_emb = text2mel.char_emb  # nn.Embedding(len(self.encodings.char2int), char_emb_size, padding_idx=0)
        self.case_emb = text2mel.case_emb  # nn.Embedding(4, 16, padding_idx=0)

        self.mgc_proj = text2mel.mgc_proj  # LinearNorm(mgc_size, self.MGC_PROJ_SIZE)
        self.encoder = text2mel.encoder
        self.decoder = text2mel.decoder

        self.dec2hid = text2mel.dec2hid

        self.output_mgc = text2mel.output_mgc  # nn.Sequential(LinearNorm(500, mgc_size * pframes, w_init_gain='sigmoid'))
        self.output_stop = text2mel.output_stop  # nn.Sequential(LinearNorm(500, self.pframes, w_init_gain='sigmoid'),
        #              nn.Sigmoid())
        # self.mel2style = text2mel.mel2style  # Mel2Style(num_mgc=mgc_size, gst_dim=self.STYLE_EMB_SIZE, num_gst=self.NUM_GST)
        self.num_gst = text2mel.mel2style.num_gst
        self.gst = text2mel.mel2style.gst
        self.att = text2mel.att  # Attention(encoder_size + self.STYLE_EMB_SIZE // 2, decoder_size)

        self.speaker_emb = text2mel.speaker_emb
        self.postnet = text2mel.postnet

    def forward(self, x, x_case, spearker_id, style_probs):
        batch_size = 1
        unfolded_gst = torch.tensor([[i for i in range(self.num_gst)] for ii in range(batch_size)],
                                    dtype=torch.long)
        unfolded_gst = torch.tanh(self.gst(unfolded_gst))

        a = style_probs.unsqueeze(1)
        style = torch.bmm(a, unfolded_gst).squeeze(1)
        # x, x_speaker = self._make_input(input)

        lstm_input = torch.cat((self.char_emb(x), self.case_emb(x_case)), dim=-1)
        x_speaker = self.speaker_emb(spearker_id)

        encoder_output, encoder_hidden = self.encoder(lstm_input.permute(1, 0, 2))
        encoder_output = encoder_output.permute(1, 0, 2)
        style = style.unsqueeze(1).repeat(1, encoder_output.shape[1], 1)
        x_speaker = x_speaker.unsqueeze(1).repeat(1, encoder_output.shape[1], 1)

        encoder_output = torch.cat((encoder_output, x_speaker + style), dim=-1)

        _, decoder_hidden = self.decoder(torch.zeros((1,
                                                      encoder_output.shape[0],
                                                      encoder_output.shape[2] + self.MGC_PROJ_SIZE)
                                                     ))
        last_mgc = torch.zeros((1, self.mgc_order))
        lst_output = []
        lst_stop = []
        lst_att = []
        index = 0
        stationary = 0
        last_index = 0
        delta_att = 7
        wait_count = 0
        while True:
            start = last_index - delta_att
            stop = last_index + delta_att
            if start < 0:
                stop += -start
                start = 0
            if stop > encoder_output.shape[1] - 1:
                start -= stop - encoder_output.shape[1] - 1
                stop = encoder_output.shape[1] - 1

            if start < 0 or stop > encoder_output.shape[1] - 1:
                start = 0
                stop = encoder_output.shape[1]

            att_vec, att = self.att(decoder_hidden[-1][-1].unsqueeze(0), encoder_output[:, start:stop, :])
            new_index = torch.argmax(att_vec).detach().squeeze().cpu() + start
            if new_index == last_index:
                wait_count += 1
                if wait_count == 10:
                    att = encoder_output[:, last_index + 1, :]
            else:
                wait_count = 0

            last_index = new_index

            if last_index >= encoder_output.shape[1] - 2:
                stationary += 1
            if stationary == 4:
                break
            lst_att.append(att_vec.unsqueeze(1))
            m_proj = torch.tanh(self.mgc_proj(last_mgc))

            decoder_input = torch.cat((att, m_proj), dim=1)
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), hx=decoder_hidden)
            # attn_output, attn_hidden = self.attention_rnn(decoder_input.unsqueeze(0), hx=attn_hidden)
            decoder_output = decoder_output.permute(1, 0, 2)

            decoder_output = self.dec2hid(decoder_output)
            out_mgc = torch.sigmoid(self.output_mgc(decoder_output))
            out_stop = self.output_stop(decoder_output)
            for iFrame in range(self.pframes):
                lst_output.append(out_mgc[:, :, iFrame * self.mgc_order:iFrame * self.mgc_order + self.mgc_order])
                lst_stop.append(out_stop[:, :, iFrame])

            last_mgc = out_mgc.squeeze(1)[:, -self.mgc_order:]
            index += self.pframes
            stop_probs = out_stop[0][-1].detach()
            for sp in stop_probs:
                if sp > 0.5:
                    break
            # failsafe
            if index > x.shape[1] * 25:
                break
        mgc = torch.cat(lst_output, dim=1)  # .view(len(input), -1, self.mgc_order)

        return mgc + self.postnet(mgc)


encodings = Encodings()
encodings.load('data/text2mel.encodings')
text2mel = Text2Mel(encodings, pframes=3)
text2mel.load('data/text2mel.best')
text2mel.eval()

st2m = Text2MelSimplified(text2mel)

script_model = torch.jit.script(st2m)  # , (torch.tensor([0, 1, 2, 3]),
#  torch.tensor([0]),
#  torch.tensor([0 for _ in range(10)])))

script_model.save("data/text2mel.pth")

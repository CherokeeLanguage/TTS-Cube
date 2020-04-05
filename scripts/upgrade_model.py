import optparse
import sys

sys.path.append('')
import torch
import torch.nn as nn
from cube2.networks.modules import Attention, AttentionCNN, PostNet, Mel2Style, ConvNorm, LinearNorm


class Text2Mel(nn.Module):
    def __init__(self, encodings, char_emb_size=100, encoder_size=256, encoder_layers=1, decoder_size=1024,
                 decoder_layers=2, mgc_size=80, pframes=3, teacher_forcing=0.0):
        super(Text2Mel, self).__init__()
        self.MGC_PROJ_SIZE = 100
        self.STYLE_EMB_SIZE = 100
        self.NUM_GST = 10
        # self.ATT_RNN_SIZE = 1000

        self.encodings = encodings
        self.teacher_forcing = teacher_forcing
        self.pframes = pframes
        self.mgc_order = mgc_size
        self.char_emb = nn.Embedding(len(self.encodings.char2int), char_emb_size, padding_idx=0)
        self.case_emb = nn.Embedding(4, 16, padding_idx=0)

        self.mgc_proj = LinearNorm(mgc_size, self.MGC_PROJ_SIZE)
        self.encoder = nn.LSTM(char_emb_size + 16, encoder_size, encoder_layers, bias=True,
                               dropout=0 if encoder_layers == 1 else 0, bidirectional=True)

        self.decoder = nn.LSTM(encoder_size * 2 + self.MGC_PROJ_SIZE + self.STYLE_EMB_SIZE, decoder_size,
                               decoder_layers,
                               bias=True,
                               dropout=0 if decoder_layers == 1 else 0,
                               bidirectional=False)

        self.dec2hid = nn.Sequential(LinearNorm(decoder_size, 500, w_init_gain='tanh'), nn.Tanh())
        self.dropout = nn.Dropout(0.1)
        self.output_mgc = nn.Sequential(LinearNorm(500, mgc_size * pframes, w_init_gain='sigmoid'))
        self.output_stop = nn.Sequential(LinearNorm(500, self.pframes, w_init_gain='sigmoid'),
                                         nn.Sigmoid())
        self.mel2style = Mel2Style(num_mgc=mgc_size, gst_dim=self.STYLE_EMB_SIZE, num_gst=self.NUM_GST)

        self.att = Attention(encoder_size + self.STYLE_EMB_SIZE // 2, decoder_size)
        self.speaker_emb = nn.Embedding(len(encodings.speaker2int), self.STYLE_EMB_SIZE)
        self.postnet = PostNet(num_mels=mgc_size)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--source', action='store', dest='input_file',
                      help='Source model (must end with .last or .best)')
    parser.add_option('--destination', action='store', dest='output_file',
                      help='Destination model file')

    (params, _) = parser.parse_args(sys.argv)
    from cube2.io_modules.dataset import Encodings

    encodings = Encodings()
    encodings.load(params.input_file.replace('.best', '.encodings').replace('.last', '.encodings'))
    text2mel = Text2Mel(encodings, teacher_forcing=False, pframes=3)
    text2mel.load(params.input_file)
    del text2mel.att
    text2mel.att = AttentionCNN(256 + 100 // 2, 1024)
    text2mel.save(params.output_file)

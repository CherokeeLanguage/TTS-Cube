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
import sys
import torch.nn as nn
import numpy as np
import json
import random

sys.path.append('')
from cube2.networks.modules import Attention, PostNet, Mel2Style
from os.path import exists


class Text2Mel(nn.Module):
    def __init__(self, encodings, char_emb_size=100, encoder_size=128, encoder_layers=1, decoder_size=1000,
                 decoder_layers=1, mgc_size=80, pframes=5, teacher_forcing=0.0):
        super(Text2Mel, self).__init__()
        self.MGC_PROJ_SIZE = 256
        self.ATT_RNN_SIZE = 1000

        self.encodings = encodings
        self.teacher_forcing = teacher_forcing
        self.pframes = pframes
        self.mgc_order = mgc_size
        self.char_emb = nn.Embedding(len(self.encodings.char2int), char_emb_size, padding_idx=0)
        self.case_emb = nn.Embedding(4, 16, padding_idx=0)
        self.char_conv = nn.Sequential(nn.Conv1d(char_emb_size + 16, 512, 5, padding=2),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 5, padding=2),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 5, padding=2),
                                       nn.ReLU(),
                                       nn.Dropout(0.5)
                                       )
        self.mgc_proj = nn.Sequential(nn.Linear(mgc_size * self.pframes, self.MGC_PROJ_SIZE), nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(self.MGC_PROJ_SIZE, self.MGC_PROJ_SIZE), nn.ReLU(), nn.Dropout(0.5))
        self.encoder = nn.LSTM(512, encoder_size, encoder_layers, bias=True,
                               dropout=0 if encoder_layers == 1 else 0, bidirectional=True)
        self.decoder = nn.LSTM(encoder_size * 2 + self.MGC_PROJ_SIZE, decoder_size, decoder_layers,
                               bias=True,
                               dropout=0 if decoder_layers == 1 else 0,
                               bidirectional=False)
        self.attention_rnn = nn.LSTM(encoder_size * 2 + self.MGC_PROJ_SIZE, self.ATT_RNN_SIZE, decoder_layers,
                                     bias=True,
                                     dropout=0 if decoder_layers == 1 else 0,
                                     bidirectional=False)

        self.dec2hid = nn.Sequential(nn.Linear(decoder_size + encoder_size * 2, 500), nn.ReLU(), nn.Dropout(0.5))
        self.dropout = nn.Dropout(0.1)
        self.output_mgc = nn.Sequential(nn.Linear(500, mgc_size * pframes))
        self.output_stop = nn.Sequential(nn.Linear(self.ATT_RNN_SIZE, self.pframes), nn.Sigmoid())
        self.att = Attention(encoder_size, self.ATT_RNN_SIZE)
        self.postnet = PostNet(mgc_size)
        self.mel2style = Mel2Style(num_mgc=mgc_size, gst_dim=encoder_size * 2)
        self.mel2fft = PostNet(mgc_size, output_size=513)

    def forward(self, input, gs_mgc=None, token=None):
        if gs_mgc is not None:
            max_len = max([mgc.shape[0] for mgc in gs_mgc])
            # gs_mgc = torch.tensor(gs_mgc, dtype=self._get_device())
            tmp = np.zeros((len(gs_mgc), (max_len // self.pframes) * self.pframes, self.mgc_order))
            for iFrame in range((max_len // self.pframes) * self.pframes):
                index = iFrame
                for iB in range(len(gs_mgc)):
                    if index < gs_mgc[iB].shape[0]:
                        for zz in range(self.mgc_order):
                            tmp[iB, iFrame, zz] = gs_mgc[iB][index, zz]

            gs_mgc = torch.tensor(tmp, device=self._get_device(), dtype=torch.float)
            gsts, style = self.mel2style(gs_mgc)
        else:  # uniform distribution of style tokens
            batch_size = len(input)
            unfolded_gst = torch.tensor([[i for i in range(self.mel2style.num_gst)] for _ in range(batch_size)],
                                        device=self.dec2hid[0].weight.device.type, dtype=torch.long)
            unfolded_gst = torch.tanh(self.mel2style.gst(unfolded_gst))
            # a = [(1.0 - 0.8) / (self.mel2style.num_gst - 1) for _ in range(self.mel2style.num_gst)]
            # a[2] = 0.8
            if token is None:
                a = [1.0 / self.mel2style.num_gst for _ in range(self.mel2style.num_gst)]
            else:
                a = [0, 0, 0, 0, 0, 0, 0, 0]
                a[token] = 1
            a = torch.tensor(a, device=self.dec2hid[0].weight.device.type, dtype=torch.float).unsqueeze(0).unsqueeze(
                0).repeat(batch_size, 1, 1)
            style = torch.bmm(a, unfolded_gst).squeeze(1)
        index = 0
        # input
        x = self._make_input(input)
        lstm_input = self.char_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_output, encoder_hidden = self.encoder(lstm_input.permute(1, 0, 2))
        encoder_output = encoder_output.permute(1, 0, 2)
        style = style.unsqueeze(1).expand_as(encoder_output)
        encoder_output = encoder_output  # + style

        _, decoder_hidden = self.decoder(
            torch.zeros((1, encoder_output.shape[0], encoder_output.shape[2] + self.MGC_PROJ_SIZE),
                        device=self._get_device()))
        attn_output, attn_hidden = self.attention_rnn(
            torch.zeros((1, encoder_output.shape[0], encoder_output.shape[2] + self.MGC_PROJ_SIZE),
                        device=self._get_device()))
        last_mgc = torch.zeros((lstm_input.shape[0], self.mgc_order * self.pframes), device=self._get_device())
        lst_output = []
        lst_stop = []
        lst_att = []
        index = 0
        prev_att_vec = None
        while True:
            # from ipdb import set_trace
            # set_trace()
            att_vec, att = self.att(attn_output, encoder_output)

            lst_att.append(att_vec.unsqueeze(1))
            m_proj = self.mgc_proj(last_mgc)

            decoder_input = torch.cat((att, m_proj), dim=1)
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), hx=decoder_hidden)
            attn_output, attn_hidden = self.attention_rnn(decoder_input.unsqueeze(0), hx=attn_hidden)
            decoder_output = decoder_output.permute(1, 0, 2)
            pre_hid = torch.cat((att.unsqueeze(1), decoder_output), dim=2)
            decoder_output = self.dec2hid(pre_hid)
            out_mgc = self.output_mgc(decoder_output)
            out_stop = self.output_stop(attn_output.permute(1, 0, 2))
            for iFrame in range(self.pframes):
                lst_output.append(out_mgc[:, :, iFrame * self.mgc_order:iFrame * self.mgc_order + self.mgc_order])
                lst_stop.append(out_stop[:, :, iFrame])
            if gs_mgc is not None:
                prob = random.random()
                if prob >= self.teacher_forcing:
                    mgc_list = []
                    for ii in range(self.pframes):
                        mgc_list.append(gs_mgc[:, index + ii, :])
                    last_mgc = torch.cat(mgc_list, dim=1)
                else:

                    last_mgc = out_mgc.detach().squeeze(1)
            else:
                last_mgc = out_mgc.squeeze(1)
            index += self.pframes
            if gs_mgc is not None and index == gs_mgc.shape[1]:
                break
            elif gs_mgc is None:
                if any(out_stop[0][-1].detach().cpu().numpy() > 0.5):
                    break
                # failsafe
                if index > x.shape[1] * 25:
                    break
        mgc = torch.cat(lst_output, dim=1)  # .view(len(input), -1, self.mgc_order)
        stop = torch.cat(lst_stop, dim=1)  # .view(len(input), -1)
        att = torch.cat(lst_att, dim=1)
        mgc_post = mgc + self.postnet(mgc)
        fft = self.mel2fft(mgc)
        return mgc_post, mgc, fft, stop, att

    def _correct_attention(self, att, att_vec, prev_att_vec, encoder_outputs):
        # TODO: add method for correcting skipped and reversed encoder_outputs
        if prev_att_vec is None:
            return att_vec, att
        return att_vec, att

    def _make_input(self, input):
        max_len = max([len(seq) for seq in input])
        x_char = np.zeros((len(input), max_len), dtype=np.int32)
        x_case = np.zeros((len(input), max_len), dtype=np.int32)
        for iBatch in range(x_char.shape[0]):
            for iToken in range(x_char.shape[1]):
                if iToken < len(input[iBatch]):
                    char = input[iBatch][iToken]
                    case = 0
                    if char.lower() == char.upper():
                        case = 1  # symbol
                    elif char.lower() != char:
                        case = 2  # upper
                    else:
                        case = 3  # lower
                    char = char.lower()
                    if char in self.encodings.char2int:
                        char = self.encodings.char2int[char]
                    else:
                        char = 1  # UNK
                    x_char[iBatch, iToken] = char
                    x_case[iBatch, iToken] = case

        x_char = torch.tensor(x_char, device=self._get_device(), dtype=torch.long)
        x_case = torch.tensor(x_case, device=self._get_device(), dtype=torch.long)
        x_char = self.char_emb(x_char)
        x_case = self.case_emb(x_case)
        return torch.cat((x_char, x_case), dim=2)

    def _get_device(self):
        if self.case_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.case_emb.weight.device.type, str(self.case_emb.weight.device.index))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


class DataLoader:
    def __init__(self, dataset):
        from cube2.io_modules.dataset import DatasetIO
        self._dio = DatasetIO()
        self._dataset = dataset
        self._file_index = 0
        self._frame_index = 0
        self._cur_x = []
        self._cur_mgc = []

    def _read_next(self):
        if self._file_index == len(self._dataset.files):
            self._file_index = 0
            import random
            random.shuffle(self._dataset.files)
        file = self._dataset.files[self._file_index]
        mgc_file = file + ".mgc.npy"
        fft_file = file + ".fft.npy"
        mgc = np.load(mgc_file)
        fft = np.load(fft_file)
        txt_file = file + ".txt"
        lab_file = file + ".lab"
        if exists(lab_file):
            json_obj = json.load(open(lab_file))
            trans = json_obj['transcription']
        else:
            txt = open(txt_file).read().strip()
            trans = [c for c in txt]
        self._file_index += 1
        return trans, mgc, np.abs(fft)

    def get_batch(self, batch_size, mini_batch_size=16, device='cuda:0'):
        batch_mgc = []
        batch_fft = []
        batch_x = []
        while len(batch_x) < batch_size:
            x, mgc, fft = self._read_next()
            batch_x.append(x)
            batch_mgc.append(mgc)
            batch_fft.append(fft)
        return batch_x, batch_mgc, batch_fft
        # return torch.tensor(batch_x, device=device, dtype=torch.float32), \
        #       torch.tensor(batch_mgc, device=device, dtype=torch.float32)


def _eval(text2mel, dataset, params, mse_loss):
    import tqdm
    text2mel.eval()
    test_steps = len(dataset._dataset.files) // params.batch_size
    if len(dataset._dataset.files) % params.batch_size != 0:
        test_steps += 1
    with torch.no_grad():
        total_loss = 0.0
        progress = tqdm.tqdm(range(test_steps))
        for step in progress:
            sys.stdout.flush()
            sys.stderr.flush()
            x, mgc, fft = dataset.get_batch(batch_size=params.batch_size)
            pred_mgc, pred_pre, pred_fft, pred_stop, pred_att = text2mel(x, gs_mgc=mgc)
            target_mgc, target_fft, target_stop, target_size = _make_batch(mgc, fft, params.pframes,
                                                                           device=params.device)

            target_mgc.requires_grad = False
            target_stop.requires_grad = False

            num_tokens = [len(seq) for seq in x]
            num_mgcs = [m.shape[0] // params.pframes for m in mgc]
            if not params.disable_guided_attention:
                target_att = _compute_guided_attention(num_tokens, num_mgcs, device=params.device)
                target_att.requires_grad = False
            loss_comb = mse_loss(pred_mgc.view(-1), target_mgc.view(-1)) + \
                        mse_loss(pred_pre.view(-1), target_mgc.view(-1)) + \
                        mse_loss(pred_fft.reshape(-1), target_fft.view(-1))
            if not params.disable_guided_attention:
                loss_comb = loss_comb + (pred_att * target_att).mean()
            lss_comb = loss_comb.item()
            total_loss += lss_comb

            progress.set_description('LOSS={0:.4}'.format(lss_comb))
        return total_loss / test_steps


def _update_encodings(encodings, dataset):
    import tqdm
    for train_file in tqdm.tqdm(dataset._dataset.files):
        txt_file = train_file + ".txt"
        lab_file = train_file + ".lab"
        if exists(lab_file):
            json_obj = json.load(open(lab_file))
            trans = json_obj['transcription']
        else:
            txt = open(txt_file).read().strip()
            trans = [c for c in trans]
        for char in trans:
            from cube2.io_modules.dataset import PhoneInfo
            pi = PhoneInfo(char, [], -1, -1)
            encodings.update(pi)


def _make_batch(gs_mgc, gs_fft, pframes, device='cpu'):
    max_len = max([mgc.shape[0] for mgc in gs_mgc])
    if max_len % pframes != 0:
        max_len = (max_len // pframes) * pframes
    # gs_mgc = torch.tensor(gs_mgc, dtype=self._get_device())
    tmp_mgc = np.zeros((len(gs_mgc), max_len, gs_mgc[0].shape[1]))
    tmp_fft = np.zeros((len(gs_fft), max_len, gs_fft[0].shape[1]))
    tmp_stop = np.zeros((len(gs_mgc), max_len))
    gs_size = [mgc.shape[0] for mgc in gs_mgc]

    for ii in range(max_len):
        index = ii
        for iB in range(len(gs_mgc)):
            if index < gs_mgc[iB].shape[0]:
                tmp_stop[iB, ii] = 0.0
                for zz in range(tmp_mgc.shape[2]):
                    tmp_mgc[iB, ii, zz] = gs_mgc[iB][index, zz]
                for zz in range(tmp_fft.shape[2]):
                    tmp_fft[iB, ii, zz] = gs_fft[iB][index, zz]
            else:
                tmp_stop[iB, ii] == 1.0
    gs_mgc = torch.tensor(tmp_mgc, device=device, dtype=torch.float)
    gs_fft = torch.tensor(tmp_fft, device=device, dtype=torch.float)
    gs_stop = torch.tensor(tmp_stop, device=device, dtype=torch.float)
    return gs_mgc, gs_fft, gs_stop, gs_size


def _compute_guided_attention(num_tokens, num_mgc, device='cpu'):
    max_num_toks = max(num_tokens)
    max_num_mgc = max(num_mgc)
    target_probs = np.zeros((len(num_tokens), max_num_mgc, max_num_toks))

    for iBatch in range(len(num_tokens)):
        for iDecStep in range(max_num_mgc):
            for iAttIndex in range(max_num_toks):
                cChars = num_tokens[iBatch]
                cDecSteps = num_mgc[iBatch]
                t1 = iDecStep / cDecSteps
                value = 1.0 - np.exp(-((float(iAttIndex) / cChars - t1) ** 2) / 0.1)
                target_probs[iBatch, iDecStep, iAttIndex] = value

    return torch.tensor(target_probs, device=device)


def weight_reset(m):
    if isinstance(m, nn.LSTM) or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        m.reset_parameters()


def _start_train(params):
    from cube2.io_modules.dataset import Dataset, Encodings
    import tqdm

    trainset = Dataset("data/processed/train", max_wav_size=16000 * 12)
    import random
    random.shuffle(trainset.files)
    devset = Dataset("data/processed/dev")
    sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
        len(devset.files)) + ' development files\n')
    epoch = 1
    patience_left = params.patience
    trainset = DataLoader(trainset)
    devset = DataLoader(devset)
    encodings = Encodings()
    if params.resume:
        encodings.load('data/text2mel.encodings')
    else:
        _update_encodings(encodings, trainset)
        encodings.store('data/text2mel.encodings')
    text2mel = Text2Mel(encodings, teacher_forcing=params.teacher_forcing, pframes=params.pframes)
    if params.resume:
        text2mel.load('data/text2mel.last')
    text2mel.to(params.device)
    optimizer_gen = torch.optim.Adam(text2mel.parameters(), lr=params.lr)
    text2mel.save('data/text2mel.last')

    # text2mel.mel2style.apply(weight_reset)

    test_steps = 500
    global_step = 0

    bce_loss = torch.nn.BCELoss()
    abs_loss = torch.nn.L1Loss(reduction='mean')
    mse_loss = torch.nn.MSELoss(reduction='mean')
    best_gloss = _eval(text2mel, devset, params, mse_loss)
    sys.stdout.write('Devset loss={0}\n'.format(best_gloss))
    while patience_left > 0:
        text2mel.train()
        total_loss = 0.0
        progress = tqdm.tqdm(range(test_steps))
        for step in progress:
            sys.stdout.flush()
            sys.stderr.flush()
            global_step += 1
            x, mgc, fft = trainset.get_batch(batch_size=params.batch_size)
            pred_mgc, pred_pre, pred_fft, pred_stop, pred_att = text2mel(x, gs_mgc=mgc)
            target_mgc, target_fft, target_stop, target_size = _make_batch(mgc,
                                                                           fft,
                                                                           params.pframes,
                                                                           device=params.device)
            target_mgc.requires_grad = False
            target_stop.requires_grad = False

            num_tokens = [len(seq) for seq in x]
            num_mgcs = [m.shape[0] // params.pframes for m in mgc]
            if not params.disable_guided_attention:
                target_att = _compute_guided_attention(num_tokens, num_mgcs, device=params.device)
                target_att.requires_grad = False

            lst_gs = []
            lst_pre_mgc = []
            lst_post_mgc = []
            lst_fft = []
            lst_gs_fft = []

            for sz in target_size:
                lst_gs.append(target_mgc[:, :sz, :])
                lst_gs_fft.append(target_fft[:, :sz, :])
                lst_pre_mgc.append(pred_pre[:, :sz, :])
                lst_post_mgc.append(pred_mgc[:, :sz, :])
                lst_fft.append(pred_fft[:, :sz, :])

            l_tar_mgc = torch.cat(lst_gs, dim=1)
            l_pre_mgc = torch.cat(lst_pre_mgc, dim=1)
            l_post_mgc = torch.cat(lst_post_mgc, dim=1)
            l_fft = torch.cat(lst_fft, dim=1)
            l_tar_fft = torch.cat(lst_gs_fft, dim=1)
            loss_comb = mse_loss(l_post_mgc.view(-1), l_tar_mgc.view(-1)) + \
                        mse_loss(l_pre_mgc.view(-1), l_tar_mgc.view(-1)) + \
                        mse_loss(l_fft.view(-1), l_tar_fft.view(-1))

            loss_comb = loss_comb + bce_loss(pred_stop.view(-1), target_stop.view(-1))
            if not params.disable_guided_attention:
                loss_comb = loss_comb + (pred_att * target_att).mean()
            loss = loss_comb
            optimizer_gen.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text2mel.parameters(), 1.)
            optimizer_gen.step()
            lss_comb = loss_comb.item()
            total_loss += lss_comb

            progress.set_description('LOSS={0:.4}'.format(lss_comb))

        g_loss = _eval(text2mel, devset, params, mse_loss)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.write(
            '\tGlobal step {0} LOSS={1:.4}\n'.format(global_step, total_loss / test_steps))
        sys.stdout.write('\tDevset evaluation: {0}\n'.format(g_loss))
        if g_loss < best_gloss:
            best_gloss = g_loss
            sys.stdout.write('\tStoring data/text2mel.best\n')
            text2mel.save('data/text2mel.best')

        if not np.isnan(total_loss):
            sys.stdout.write('\tStoring data/text2mel.last\n')
            text2mel.save('data/text2mel.last')
        else:
            sys.stdout.write('exiting because of nan loss')
            sys.exit(0)


def _test_synth(params):
    from cube2.io_modules.dataset import Dataset, Encodings

    encodings = Encodings()
    encodings.load('data/text2mel.encodings')
    text2mel = Text2Mel(encodings)
    text2mel.load('data/text2mel.last')
    text2mel.to(params.device)
    text2mel.eval()
    mgc, stop, att = text2mel(['This is a simple test'])
    # from ipdb import set_trace
    # set_trace()
    mgc = mgc[0].detach().cpu().numpy()
    bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
    for x in range(mgc.shape[0]):
        for y in range(mgc.shape[1]):
            val = mgc[x, y]
            color = np.clip(val * 255, 0, 255)
            bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]  # bitmap[y, x] = [color, color, color]
    from PIL import Image
    img = Image.fromarray(bitmap)
    img.save('test.png')

    att = att[0].detach().cpu().numpy()
    new_att = np.zeros((att.shape[1], att.shape[0], 3), dtype=np.uint8)
    for ii in range(att.shape[1]):
        for jj in range(att.shape[0]):
            val = np.clip(int(att[jj, ii] * 255), 0, 255)
            new_att[ii, jj, 0] = val
            new_att[ii, jj, 1] = val
            new_att[ii, jj, 2] = val

    img = Image.fromarray(new_att)
    img.save('test.att.png')


if __name__ == '__main__':
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('--patience', action='store', dest='patience', default=20, type='int',
                      help='Num epochs without improvement (default=20)')
    parser.add_option("--batch-size", action='store', dest='batch_size', default='12', type='int',
                      help='number of samples in a single batch (default=12)')
    parser.add_option("--resume", action='store_true', dest='resume',
                      help='Resume from previous checkpoint')
    parser.add_option("--use-gan", action='store_true', dest='use_gan',
                      help='Resume from previous checkpoint')
    parser.add_option("--pframes", action='store', dest='pframes', default=3,
                      help='Number of frames to predict at once (default=3)')
    parser.add_option("--synth-test", action="store_true", dest="test")
    parser.add_option("--device", action="store", dest="device", default='cuda:0')
    parser.add_option("--lr", action="store", dest="lr", default=2e-4, type=float, help='Learning rate (default=2e-4)')
    parser.add_option("--teacher-forcing", action="store", dest="teacher_forcing", default=0.0, type=float,
                      help='Probability to use generated samples instead of ground '
                           'truth for training: 0.0-never 1.0-always (default=0.0)')
    parser.add_option("--disable-guided-attention", action="store_true", dest="disable_guided_attention",
                      help='Disable guided attention (monotonic)')

    (params, _) = parser.parse_args(sys.argv)

    if params.test:
        _test_synth(params)
    else:
        _start_train(params)

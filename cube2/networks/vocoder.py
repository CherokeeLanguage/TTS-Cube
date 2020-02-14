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

sys.path.append('')
from cube2.networks.modules import UpsampleNet


class CubeNetOLD(nn.Module):
    def __init__(self, mgc_size=80, lstm_size=500, lstm_layers=1, upsample_scales=[4, 4, 4, 4]):
        super(CubeNet, self).__init__()
        self.upsample = UpsampleNet(mgc_size, 256, upsample_scales)
        self.rnn = nn.LSTM(256 + 1, lstm_size, num_layers=lstm_layers)
        self.output = nn.Linear(lstm_size, 2)

    def synthesize(self, mgc, batch_size=16, temperature=0.8):
        if mgc.shape[0] // batch_size != 0:
            empty_slots = np.zeros(
                ((mgc.shape[0] // batch_size) * batch_size + batch_size - mgc.shape[0], mgc.shape[1]))
            mgc = np.concatenate((mgc, empty_slots), axis=0)
            c = torch.tensor(mgc, dtype=torch.float32).view(-1, batch_size, mgc.shape[1]).to(
                self.output.weight.device.type)
        else:
            c = torch.tensor(mgc, dtype=torch.float32).unsqueeze(0).to(
                self.output.weight.device.type)
        _, _, signal = self.forward(c, temperature=temperature, eps_min=-20)

        return np.array(np.clip(signal.detach().cpu().view(-1).numpy(), -1.0, 1.0) * 32500, dtype=np.int16)

    def forward(self, mgc, ext_conditioning=None, temperature=1.0, x=None, eps_min=-7):
        cond = self.upsample(mgc)
        if x is not None:
            x = x.view(x.shape[0], -1)
            x = torch.cat((torch.zeros((x.shape[0], 1), device=x.device.type), x[:, 0:-1]), dim=1)
            rnn_input = torch.cat((cond, x.unsqueeze(2)), dim=2)
            rnn_output, _ = self.rnn(rnn_input.permute(1, 0, 2))
            rnn_output = rnn_output.permute(1, 0, 2)
            output = self.output(rnn_output)
            mean = torch.tanh(output[:, :, 0])
            logvar = output[:, :, 1]
            eps = torch.randn_like(mean)
            zz = self._reparameterize(mean, logvar, eps * temperature)
        else:
            mean_list = []
            logvar_list = []
            zz_list = []
            hidden = None
            x = torch.zeros((mgc.shape[0], 1), device=mgc.device.type)
            for ii in range(cond.shape[1]):
                rnn_input = torch.cat((cond[:, ii, :].unsqueeze(1), x.unsqueeze(2)), dim=2)
                rnn_output, hidden = self.rnn(rnn_input.permute(1, 0, 2), hx=hidden)
                rnn_output = rnn_output.permute(1, 0, 2)
                output = self.output(rnn_output)
                mean = torch.tanh(output[:, :, 0])
                logvar = output[:, :, 1]
                eps = torch.randn_like(mean)
                zz = self._reparameterize(mean, logvar, eps * temperature, eps_min=eps_min)
                mean_list.append(mean)
                logvar_list.append(logvar)
                zz_list.append(zz)
                x = zz

            mean = torch.cat(mean_list, dim=1)
            logvar = torch.cat(logvar_list, dim=1)
            zz = torch.cat(zz_list, dim=1)

        return mean, logvar, zz

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def _reparameterize(self, mu, logvar, eps, eps_min=-7):
        logvar = torch.clamp(logvar, min=eps_min)
        std = torch.exp(logvar)
        return mu + eps * std


class CubeNet2(nn.Module):
    def __init__(self, mgc_size=80, lstm_size=500, upsample_scales_input=[4, 4],
                 output_samples=16):
        super(CubeNet2, self).__init__()
        COND_SIZE = lstm_size
        self.output_samples = output_samples
        self.upsample_input = UpsampleNet(mgc_size, COND_SIZE, upsample_scales_input)
        cond_size = COND_SIZE + output_samples
        self.rnn = nn.LSTM(cond_size, lstm_size, num_layers=1)
        self.output = nn.Linear(lstm_size, 2)
        self.output_samples = output_samples
        if output_samples > 1:
            self.m_outputs = nn.ModuleList()
            for x in range(1, output_samples):
                # self.m_outputs.append(nn.Linear(lstm_size + 1, 2))
                self.m_outputs.append(nn.Sequential(nn.Linear(lstm_size + x, 300), nn.Tanh(), nn.Linear(300, 2)))
            self.output_nc = nn.Linear(lstm_size, output_samples * 2)
        else:
            self.m_outputs = None

    def synthesize(self, mgc, batch_size=16, temperature=0.8):
        total_audio_size = mgc.shape[0] * 256
        if mgc.shape[0] // batch_size != 0:
            empty_slots = np.zeros(
                ((mgc.shape[0] // batch_size) * batch_size + batch_size - mgc.shape[0], mgc.shape[1]))
            mgc = np.concatenate((mgc, empty_slots), axis=0)
            c = torch.tensor(mgc, dtype=torch.float32).view(-1, batch_size, mgc.shape[1]).to(
                self.output.weight.device.type)
        else:
            c = torch.tensor(mgc, dtype=torch.float32).unsqueeze(0).to(
                self.output.weight.device.type)
        _, _, signal, _, _ = self.forward(c, temperature=temperature, eps_min=-7)

        signal = signal.detach().cpu().view(-1).numpy()[:total_audio_size]
        # s_min = np.min(signal)
        # s_max = np.max(signal)
        # norm = (s_max - s_min) / 2.0
        # stdev = np.std(signal)
        # signal = signal / (stdev * 2)

        return np.array(np.clip(signal, -1.0, 1.0) * 32767, dtype=np.int16)

    def forward(self, mgc, ext_conditioning=None, temperature=1.0, x=None, eps_min=-7):
        cond = self.upsample_input(mgc)
        if x is not None:
            x = x.view(x.shape[0], -1)
            x = torch.cat(
                (torch.zeros((x.shape[0], self.output_samples), device=x.device.type), x),
                dim=1)
            x_rnn = x.view(x.shape[0], x.shape[1] // self.output_samples, -1)

            eps = torch.randn_like(x_rnn[:, :-1, :])
            rnn_input = torch.cat((cond, x_rnn[:, :-1, :]), dim=2)

            rnn_output, _ = self.rnn(rnn_input.permute(1, 0, 2))
            rnn_output = rnn_output.permute(1, 0, 2) + cond
            output = self.output(rnn_output)
            m_list = []
            l_list = []
            nc_m_list = []
            nc_l_list = []
            mean = output[:, :, 0]
            logvar = output[:, :, 1]
            m_list.append(mean.unsqueeze(2))
            l_list.append(logvar.unsqueeze(2))
            nc_m_list.append(mean.unsqueeze(2))
            nc_l_list.append(logvar.unsqueeze(2))

            if self.output_samples > 1 is not None:
                for ii, o_layer in zip(range(1, self.output_samples), self.m_outputs):
                    inp = torch.cat((rnn_output, x_rnn[:, 1:, :ii]), dim=-1)
                    output = o_layer(inp)
                    mean = output[:, :, 0]
                    logvar = output[:, :, 1]
                    m_list.append(mean.unsqueeze(2))
                    l_list.append(logvar.unsqueeze(2))
                output_nc = self.output_nc(rnn_output)
                mean_nc = output_nc[:, :, :self.output_samples]
                logvar_nc = output_nc[:, :, self.output_samples:]
            else:
                mean_nc = None
                logvar_nc = None

            mean = torch.cat(m_list, dim=2)  # .reshape(mean.shape[0], -1, self.output_samples)
            logvar = torch.cat(l_list, dim=2)  # .reshape(logvar.shape[0], -1, self.output_samples)

            zz = self._reparameterize(mean, logvar, eps, eps_min=eps_min)
        else:
            mean_list = []
            logvar_list = []
            zz_list = []
            hidden = None
            x = torch.zeros((mgc.shape[0], 1, self.output_samples), device=mgc.device.type)
            import tqdm
            for ii in tqdm.tqdm(range(cond.shape[1])):
                rnn_input = torch.cat((cond[:, ii, :].unsqueeze(1), x), dim=2)
                eps = torch.randn_like(x)
                rnn_output, hidden = self.rnn(rnn_input.permute(1, 0, 2), hx=hidden)
                rnn_output = rnn_output.permute(1, 0, 2) + cond[:, ii, :].unsqueeze(1)
                output = self.output(rnn_output)

                mean = output[:, :, 0]
                logvar = output[:, :, 1]
                zz = self._reparameterize(mean, logvar, eps[:, :, 0] * temperature, eps_min=eps_min)
                if self.output_samples > 1:
                    zz_cache = [zz.unsqueeze(2)]
                    for ii, o_layer in zip(range(1, self.output_samples), self.m_outputs):
                        zz_tmp = torch.cat(zz_cache, dim=-1)
                        inp = torch.cat((rnn_output, zz_tmp), dim=-1)
                        output = o_layer(inp)
                        mean = output[:, :, 0]
                        logvar = output[:, :, 1]
                        zz = self._reparameterize(mean, logvar, eps[:, :, ii] * temperature, eps_min=eps_min)
                        # if any(np.isnan(zz.numpy())):
                        #     from ipdb import set_trace
                        #     set_trace()

                        zz_cache.append(zz.unsqueeze(2))
                    zz_list.append(torch.cat(zz_cache, dim=-1))
                else:
                    zz_list.append(zz.unsqueeze(2))
                # output_nc = self.output_nc(rnn_output)
                # mean_nc = output_nc[:, :, :self.output_samples]
                # logvar_nc = output_nc[:, :, self.output_samples:]
                # from ipdb import set_trace
                # set_trace()
                x = zz_list[-1]
            zz = torch.cat(zz_list, dim=1)
            mean = None
            logvar = None

        if not self.training:
            return mean, logvar, zz, None, None
        else:
            if mean_nc is not None:
                return mean, logvar, zz, mean_nc, logvar_nc
            else:
                return mean, logvar, zz, mean, logvar

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def _reparameterize(self, mu, logvar, eps, eps_min=-7):
        logvar = torch.clamp(logvar, min=eps_min)
        std = torch.exp(logvar)
        return mu + eps * std


class CubeNet(nn.Module):
    def __init__(self, mgc_size=80, lstm_size=500, lstm_layers=1, upsample_scales_input=[4, 4, 4],
                 output_samples=4):
        super(CubeNet, self).__init__()
        self.output_samples = output_samples
        self.upsample_input = UpsampleNet(mgc_size, 256, upsample_scales_input)
        self.rnn = nn.LSTM(256 + output_samples, lstm_size, num_layers=lstm_layers)
        self.output_samples = output_samples
        self.output = nn.Linear(lstm_size, 2 * output_samples)

    def synthesize(self, mgc, batch_size=16, temperature=0.8):
        empty_slots = np.zeros(((mgc.shape[0] // batch_size) * batch_size + batch_size - mgc.shape[0], mgc.shape[1]))
        mgc = np.concatenate((mgc, empty_slots), axis=0)
        c = torch.tensor(mgc, dtype=torch.float32).view(-1, batch_size, mgc.shape[1]).to(self.output.weight.device.type)
        _, _, signal = self.forward(c, temperature=temperature, eps_min=-20)

        signal = signal.detach().cpu().view(-1).numpy()
        s_min = np.min(signal)
        s_max = np.max(signal)
        norm = (s_max - s_min) / 2.0
        signal = signal / norm

        return np.array(np.clip(signal, -1.0, 1.0) * 32767, dtype=np.int16)

    def forward(self, mgc, ext_conditioning=None, temperature=1.0, x=None, eps_min=-7):
        cond = self.upsample_input(mgc)
        if x is not None:
            x = x.view(x.shape[0], -1)
            x = torch.cat(
                (torch.zeros((x.shape[0], self.output_samples), device=x.device.type), x[:, 0:-self.output_samples]),
                dim=1)
            x_rnn = x.view(x.shape[0], x.shape[1] // self.output_samples, -1)
            rnn_input = torch.cat((cond, x_rnn), dim=2)
            rnn_output, _ = self.rnn(rnn_input.permute(1, 0, 2))
            rnn_output = rnn_output.permute(1, 0, 2)
            # upsampled_output = self.upsample_output(rnn_output)
            output = self.output(rnn_output)
            mean = output[:, :, 0:self.output_samples].unsqueeze(1)
            logvar = output[:, :, self.output_samples:].unsqueeze(1)
            eps = torch.randn_like(mean)
            zz = self._reparameterize(mean, logvar, eps * temperature)
        else:
            mean_list = []
            logvar_list = []
            zz_list = []
            hidden = None
            x = torch.zeros((mgc.shape[0], 1, self.output_samples), device=mgc.device.type)
            import tqdm
            for ii in tqdm.tqdm(range(cond.shape[1])):
                rnn_input = torch.cat((cond[:, ii, :].unsqueeze(1), x), dim=2)
                rnn_output, hidden = self.rnn(rnn_input.permute(1, 0, 2), hx=hidden)
                rnn_output = rnn_output.permute(1, 0, 2)
                output = self.output(rnn_output)
                # from ipdb import set_trace
                # set_trace()
                mean = output[:, :, :self.output_samples]
                logvar = output[:, :, self.output_samples:]
                eps = torch.randn_like(mean)
                zz = self._reparameterize(mean, logvar, eps * temperature, eps_min=eps_min)
                mean_list.append(mean)
                logvar_list.append(logvar)
                zz_list.append(zz)
                x = zz

            mean = torch.cat(mean_list, dim=1)
            logvar = torch.cat(logvar_list, dim=1)
            zz = torch.cat(zz_list, dim=1)
        return mean, logvar, zz

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def _reparameterize(self, mu, logvar, eps, eps_min=-7):
        logvar = torch.clamp(logvar, min=eps_min)
        std = torch.exp(logvar)
        return mu + eps * std

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

# import dynet_config
import optparse
import sys

sys.path.append('')
import numpy as np


def write_signal_to_file(signal, output_file, params):
    from cube2.io_modules.dataset import DatasetIO
    dio = DatasetIO()

    dio.write_wave(output_file, signal, params.target_sample_rate, dtype=signal.dtype)


def _trim(mgc, att, pframes):
    mx = att.shape[0]
    count = 0
    for ii in range(mgc.shape[0]):
        if all(mgc[ii] < 0.001):
            count += 1
        if count == 9:
            mx -= 9
            break
    mx = min(mx, mgc.shape[0])
    return mgc[:mx], att[:mx // pframes]


def synthesize(params):
    import torch
    import time
    from cube2.io_modules.dataset import DatasetIO, Encodings
    from cube2.networks.text2mel import Text2Mel
    from cube2.networks.vocoder import CubeNet2
    from cube2.io_modules.vocoder import MelVocoder
    from os.path import exists
    if params.g2p is not None:
        from cube2.networks.g2p import G2P
        g2p = G2P()
        g2p.load(params.g2p)
        g2p.to(params.device)
        g2p.eval()
        if exists(params.g2p + '.lexicon'):
            g2p.load_lexicon(params.g2p + '.lexicon')

    dio = DatasetIO()
    encodings = Encodings()
    encodings.load('{0}.encodings'.format(params.text2mel))
    text2mel = Text2Mel(encodings, pframes=params.pframes)
    text2mel.load('{0}.best'.format(params.text2mel))
    text2mel.to(params.device)
    text2mel.eval()
    # from ipdb import set_trace
    # set_trace()
    if params.use_gl:
        cubenet = None
        vocoder = MelVocoder()
    elif params.model == 'single':
        cubenet = CubeNet2(upsample_scales_input=[4, 4, 4, 4], output_samples=1)
        cubenet.load('{0}-single.best'.format(params.cubenet))
        cubenet.to(params.device)
        cubenet.eval()
    else:
        cubenet = CubeNet2(upsample_scales_input=[4, 4, 4], output_samples=4, lstm_size=500)
        cubenet.load('{0}-multi.best'.format(params.cubenet))
        cubenet.to(params.device)
        cubenet.eval()

    with torch.no_grad():
        if params.g2p:
            text = open(params.txt_file).read().strip()
            tokens = g2p.transcribe_utterance(text)
            trans = []
            for token in tokens:
                for ph in token.transcription:
                    trans.append(ph)
            text = ['<START>'] + trans + ['<STOP>']
        else:
            text = [c for c in open(params.txt_file).read().strip()]
        start_text2mel = time.time()
        mgc, stop, att = text2mel([text], token=params.token)
        stop_text2mel = time.time()

    mgc, att = _trim(mgc[0].detach().cpu().numpy(), att[0].detach().cpu().numpy(),
                     params.pframes)
    if not params.use_gl:
        with torch.no_grad():
            start_cubenet = time.time()
            wav = cubenet.synthesize(mgc, batch_size=64,
                                     temperature=params.temperature)
            stop_cubenet = time.time()

    else:
        start_cubenet = time.time()
        import librosa
        nmgc = vocoder._denormalize(mgc)
        nmgc = vocoder._db_to_amp(nmgc)
        S = librosa.feature.inverse.mel_to_stft(nmgc.transpose(), power=1, sr=16000, n_fft=1024)
        wav = librosa.griffinlim(S, n_iter=50, hop_length=256, win_length=1024) * 32767
        stop_cubenet = time.time()

    synth = wav
    dio.write_wave(params.output, synth, 16000, dtype=np.int16)

    bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
    for x in range(mgc.shape[0]):
        for y in range(mgc.shape[1]):
            val = mgc[x, y]
            color = np.clip(val * 255, 0, 255)
            bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]  # bitmap[y, x] = [color, color, color]
    from PIL import Image
    img = Image.fromarray(bitmap)
    img.save('{0}.mgc.png'.format(params.output))

    new_att = np.zeros((att.shape[1], att.shape[0], 3), dtype=np.uint8)
    for ii in range(att.shape[1]):
        for jj in range(att.shape[0]):
            val = np.clip(int(att[jj, ii] * 255), 0, 255)
            new_att[ii, jj, 0] = val
            new_att[ii, jj, 1] = val
            new_att[ii, jj, 2] = val

    img = Image.fromarray(new_att)
    img.save('{0}.att.png'.format(params.output))
    sys.stdout.write(
        'Text2mel time: {0}\nCubenet time: {1}\nTotal audio:{2}\n'.format(stop_text2mel - start_text2mel,
                                                                          stop_cubenet - start_cubenet,
                                                                          synth.shape[0] / 16000))


def quick_test(params):
    import torch
    import time
    from cube2.io_modules.dataset import DatasetIO, Encodings
    from cube2.networks.text2mel import Text2Mel
    from cube2.networks.vocoder import CubeNet

    dio = DatasetIO()
    encodings = Encodings()
    encodings.load('{0}.encodings'.format(params.text2mel))
    text2mel = Text2Mel(encodings)
    text2mel.load('{0}.best'.format(params.text2mel))
    text2mel.to(params.device)
    text2mel.eval()
    from cube2.io_modules.vocoder import MelVocoder
    vocoder = MelVocoder()
    if not params.use_gl:
        cubenet = CubeNet()
        cubenet.load('{0}'.format(params.cubenet))
        cubenet.to(params.device)
        cubenet.eval()
    else:
        cubenet = None
    with torch.no_grad():
        start_text2mel = time.time()
        mgc, _, stop, att = text2mel([open(params.txt_file).read().strip()])
        stop_text2mel = time.time()

    import PIL.Image
    image = PIL.Image.open(r'../test.en.wav.png')
    image = np.array(image)
    mgc = np.zeros((image.shape[1], image.shape[0]))

    for ii in range(image.shape[0]):
        for jj in range(image.shape[1]):
            mgc[jj, mgc.shape[1] - ii - 1] = image[ii, jj, 0] / 255

    # mgc, att = _trim(mgc[0].detach().cpu().numpy(), att[0].detach().cpu().numpy())
    with torch.no_grad():
        start_cubenet = time.time()
        if cubenet is not None:
            wav = cubenet.synthesize(mgc, batch_size=64,
                                     temperature=params.temperature)
        else:
            # wav = vocoder.griffinlim(fft.detach.cpu()) * 32767
            from ipdb import set_trace
            set_trace()
            import librosa
            S = librosa.feature.inverse.mel_to_stft(mgc)
            wav = librosa.griffinlim(S)
        stop_cubenet = time.time()

    synth = wav
    dio.write_wave(params.output, synth, 16000, dtype=np.int16)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='txt_file',
                      help='Path to the text file that will be synthesized')
    parser.add_option('--device', dest='device', action='store', default='cuda:0',
                      help='Use this device')
    parser.add_option('--text2mel', dest='text2mel', action='store', default='data/text2mel',
                      help='default: data/text2mel')
    parser.add_option('--cubenet', dest='cubenet', action='store', default='data/cube',
                      help='default: data/cube')
    parser.add_option('--cubenet-model-type', dest='model', choices=['single', 'multi'], default='single')
    parser.add_option('--output', dest='output', action='store', default='test.wav',
                      help='test.wav')
    parser.add_option('--g2p', dest='g2p', action='store')
    parser.add_option("--temperature", action="store", dest="temperature", type='float', default=0.35)
    parser.add_option("--token", action="store", dest="token", type=int, help='Token to use (0-7)')
    parser.add_option("--pframes", action='store', dest='pframes', default=3,
                      help='Number of frames to predict at once (default=3)')
    parser.add_option("--use-gl", action="store_true", dest="use_gl", help='Use GL reconstruction')

    (params, _) = parser.parse_args(sys.argv)

    synthesize(params)

    # quick_test(params)

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

import optparse
import sys
import numpy as np

sys.path.append('')

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--cleanup', action='store_true', dest='cleanup',
                      help='Cleanup temporary training files and start from fresh')
    parser.add_option('--train-folder', action='store', dest='train_folder',
                      help='Location of the training files')
    parser.add_option('--dev-folder', action='store', dest='dev_folder',
                      help='Location of the development files')
    parser.add_option('--target-sample-rate', action='store', dest='target_sample_rate',
                      help='Resample input files at this rate (default=16000)', type='int', default=16000)
    parser.add_option('--mgc-order', action='store', dest='mgc_order', type='int',
                      help='Order of MGC parameters (default=80)', default=80)
    parser.add_option('--speaker', action='store', dest='speaker', help='Import data under given speaker')
    parser.add_option('--device', action='store', dest='device', help='Device to use for g2p', default='cpu')
    parser.add_option('--prefix', action='store', dest='prefix', help='Use this prefix when importing files')
    parser.add_option('--g2p-model', dest='g2p', action='store',
                      help='Use this G2P model for processing')

    (params, _) = parser.parse_args(sys.argv)


    def _normalize(data):
        m = np.max(np.abs(data))
        data = (data / m) * 0.999
        return data


    def array2file(a, filename):
        np.save(filename, a)


    def file2array(filename):
        a = np.load(filename)
        return a


    def render_spectrogram(mgc, output_file):
        bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
        mgc_min = mgc.min()
        mgc_max = mgc.max()

        for x in range(mgc.shape[0]):
            for y in range(mgc.shape[1]):
                val = (mgc[x, y] - mgc_min) / (mgc_max - mgc_min)

                color = val * 255
                bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]
        from PIL import Image

        img = Image.fromarray(bitmap)  # smp.toimage(bitmap)
        img.save(output_file)


    def create_lab_file(txt_file, lab_file, speaker_name=None, g2p=None):
        fin = open(txt_file, 'r', encoding='utf-8')
        line = fin.readline().strip().replace('\t', ' ')
        json_obj = {}
        while True:
            nl = line.replace('  ', ' ')
            if nl == line:
                break
            line = nl
        line = line.strip()

        if speaker_name is not None:
            json_obj['speaker'] = speaker_name  # speaker = 'SPEAKER:' + speaker_name
        elif len(txt_file.replace('\\', '/').split('/')[-1].split('_')) != 1:
            json_obj['speaker'] = txt_file.replace('\\', '/').split('_')[0].split('/')[-1]
        else:
            json_obj['speaker'] = 'none'

        json_obj['text'] = line
        if g2p is not None:
            trans = ['<START>']
            utt = g2p.transcribe_utterance(line)
            for word in utt:
                for ph in word.transcription:
                    trans.append(ph)
            trans.append('<STOP>')
            json_obj['transcription'] = trans
        else:
            json_obj['transcription'] = ['<START>'] + [c.lower() for c in line] + ['<STOP>']

        fin.close()
        fout = open(lab_file, 'w', encoding='utf-8')
        import json
        json.dump(json_obj, fout)
        fout.close()
        return ""


    def phase_1_prepare_corpus(params):
        from os import listdir
        from os.path import isfile, join
        from os.path import exists
        train_files_tmp = [f for f in listdir(params.train_folder) if isfile(join(params.train_folder, f))]
        if params.dev_folder is not None:
            dev_files_tmp = [f for f in listdir(params.dev_folder) if isfile(join(params.dev_folder, f))]
        else:
            dev_files_tmp = []

        if params.g2p is not None:
            from cube2.networks.g2p import G2P
            g2p = G2P()
            g2p.load(params.g2p)
            g2p.to(params.device)
            g2p.eval()
            if exists(params.g2p + '.lexicon'):
                g2p.load_lexicon(params.g2p + '.lexicon')
        else:
            g2p = None

        sys.stdout.write("Scanning training files...")
        sys.stdout.flush()
        final_list = []
        for file in train_files_tmp:
            base_name = file[:-4]
            lab_name = base_name + '.txt'
            wav_name = base_name + '.wav'
            if exists(join(params.train_folder, lab_name)) and exists(join(params.train_folder, wav_name)):
                if base_name not in final_list:
                    final_list.append(base_name)

        train_files = final_list
        sys.stdout.write(" found " + str(len(train_files)) + " valid training files\n")
        sys.stdout.write("Scanning development files...")
        sys.stdout.flush()
        final_list = []
        for file in dev_files_tmp:
            base_name = file[:-4]
            lab_name = base_name + '.txt'
            wav_name = base_name + '.wav'
            if exists(join(params.dev_folder, lab_name)) and exists(join(params.dev_folder, wav_name)):
                if base_name not in final_list:
                    final_list.append(base_name)

        dev_files = final_list
        sys.stdout.write(" found " + str(len(dev_files)) + " valid development files\n")
        from cube2.io_modules.dataset import DatasetIO
        from cube2.io_modules.vocoder import MelVocoder
        from shutil import copyfile
        dio = DatasetIO()

        vocoder = MelVocoder()
        base_folder = params.train_folder
        total_files = 0
        for index in range(len(train_files)):
            total_files += 1
            sys.stdout.write("\r\tprocessing file " + str(index + 1) + "/" + str(len(train_files)))
            sys.stdout.flush()
            base_name = train_files[index]
            txt_name = base_name + '.txt'
            wav_name = base_name + '.wav'
            spc_name = base_name + '.png'
            lab_name = base_name + '.lab'

            tgt_txt_name = txt_name
            tgt_spc_name = spc_name
            tgt_lab_name = lab_name
            if params.prefix is not None:
                tgt_txt_name = params.prefix + "_{:05d}".format(total_files) + '.txt'
                tgt_spc_name = params.prefix + "_{:05d}".format(total_files) + '.png'
                tgt_lab_name = params.prefix + "_{:05d}".format(total_files) + '.lab'

            # LAB - copy or create
            if exists(join(base_folder, lab_name)):
                copyfile(join(base_folder, lab_name), join('data/processed/train', tgt_lab_name))
            else:
                create_lab_file(join(base_folder, txt_name),
                                join('data/processed/train', tgt_lab_name), speaker_name=params.speaker, g2p=g2p)
            # TXT
            copyfile(join(base_folder, txt_name), join('data/processed/train', tgt_txt_name))
            # WAVE
            data, sample_rate = dio.read_wave(join(base_folder, wav_name), sample_rate=params.target_sample_rate)
            data = _normalize(data)
            mgc = vocoder.melspectrogram(data, sample_rate=params.target_sample_rate, num_mels=params.mgc_order)
            fft = vocoder.fft(data, sample_rate=params.target_sample_rate)
            # SPECT
            render_spectrogram(mgc, join('data/processed/train', tgt_spc_name))
            if params.prefix is None:
                dio.write_wave(join('data/processed/train', base_name + '.orig.wav'), data, sample_rate)
                array2file(mgc, join('data/processed/train', base_name + '.mgc'))
                array2file(fft, join('data/processed/train', base_name + '.fft'))
            else:
                tgt_wav_name = params.prefix + "_{:05d}".format(total_files) + '.orig.wav'
                tgt_mgc_name = params.prefix + "_{:05d}".format(total_files) + '.mgc'
                tgt_fft_name = params.prefix + "_{:05d}".format(total_files) + '.fft'
                dio.write_wave(join('data/processed/train', tgt_wav_name), data, sample_rate)
                array2file(mgc, join('data/processed/train', tgt_mgc_name))
                array2file(fft, join('data/processed/train', tgt_fft_name))

        sys.stdout.write('\n')
        base_folder = params.dev_folder
        for index in range(len(dev_files)):
            total_files += 1
            sys.stdout.write("\r\tprocessing file " + str(index + 1) + "/" + str(len(dev_files)))
            sys.stdout.flush()
            base_name = dev_files[index]
            txt_name = base_name + '.txt'
            wav_name = base_name + '.wav'
            spc_name = base_name + '.png'
            lab_name = base_name + '.lab'

            tgt_txt_name = txt_name
            tgt_spc_name = spc_name
            tgt_lab_name = lab_name
            if params.prefix is not None:
                tgt_txt_name = params.prefix + "_{:05d}".format(total_files) + '.txt'
                tgt_spc_name = params.prefix + "_{:05d}".format(total_files) + '.png'
                tgt_lab_name = params.prefix + "_{:05d}".format(total_files) + '.lab'

            # LAB - copy or create
            if exists(join(base_folder, lab_name)):
                copyfile(join(base_folder, lab_name), join('data/processed/dev', tgt_lab_name))
            else:
                create_lab_file(join(base_folder, txt_name),
                                join('data/processed/dev', tgt_lab_name), speaker_name=params.speaker, g2p=g2p)
            # TXT
            copyfile(join(base_folder, txt_name), join('data/processed/dev', tgt_txt_name))
            # WAVE
            data, sample_rate = dio.read_wave(join(base_folder, wav_name), sample_rate=params.target_sample_rate)
            data = _normalize(data)
            mgc = vocoder.melspectrogram(data, sample_rate=params.target_sample_rate, num_mels=params.mgc_order)
            fft = vocoder.fft(data, sample_rate=params.target_sample_rate)
            # SPECT
            render_spectrogram(mgc, join('data/processed/dev', tgt_spc_name))
            if params.prefix is None:
                dio.write_wave(join('data/processed/dev', base_name + '.orig.wav'), data, sample_rate)
                array2file(mgc, join('data/processed/dev', base_name + '.mgc'))
                array2file(fft, join('data/processed/dev', base_name + '.fft'))
            else:
                tgt_wav_name = params.prefix + "_{:05d}".format(total_files) + '.orig.wav'
                tgt_mgc_name = params.prefix + "_{:05d}".format(total_files) + '.mgc'
                tgt_fft_name = params.prefix + "_{:05d}".format(total_files) + '.fft'
                dio.write_wave(join('data/processed/dev', tgt_wav_name), data, sample_rate)
                array2file(mgc, join('data/processed/dev', tgt_mgc_name))
                array2file(fft, join('data/processed/dev', tgt_fft_name))

        sys.stdout.write('\n')


    phase_1_prepare_corpus(params)

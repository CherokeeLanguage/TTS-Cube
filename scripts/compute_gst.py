import torch
import torch.nn as nn
import sys
import tqdm
import numpy as np

sys.path.append('')


def _normalize(data):
    m = np.max(np.abs(data))
    data = (data / m) * 0.999
    return data


if __name__ == '__main__':
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('--input', action='store', dest='input_folder')
    parser.add_option('--output', action='store', dest='output_folder')
    parser.add_option('--device', action='store', dest='device', default='cpu')
    parser.add_option('--speaker', action='store', dest='prefix')

    (params, _) = parser.parse_args(sys.argv)

    if not params.output_folder or not params.input_folder or not params.prefix:
        print("Must specify --output, --input and --speaker")
        sys.exit(0)

    from cube2.io_modules.dataset import DatasetIO, Encodings, Dataset
    from cube2.networks.text2mel import Text2Mel
    from cube2.io_modules.vocoder import MelVocoder

    encodings = Encodings()
    encodings.load('data/text2mel.encodings')
    text2mel = Text2Mel(encodings, pframes=3)
    text2mel.load('data/text2mel.best')
    text2mel.to(params.device)
    text2mel.eval()

    from os import listdir
    from os.path import isfile, join
    from os.path import exists
    from shutil import copyfile

    train_files_tmp = [f for f in listdir(params.input_folder) if isfile(join(params.input_folder, f))]
    final_list = []
    for file in train_files_tmp:
        base_name = file[:-4]
        lab_name = base_name + '.txt'
        wav_name = base_name + '.wav'
        if exists(join(params.input_folder, lab_name)) and exists(join(params.input_folder, wav_name)):
            if base_name not in final_list:
                final_list.append(base_name)

    files = sorted(final_list)
    dio = DatasetIO()
    vocoder = MelVocoder()
    index = 0
    for file in tqdm.tqdm(files):
        index += 1
        wav_name = file + '.wav'
        txt_name = file + '.txt'

        tgt_txt_name = params.prefix + "_{:05d}".format(index) + '.txt'
        tgt_gst_name = params.prefix + "_{:05d}".format(index) + '.gst'
        data, sample_rate = dio.read_wave(join(params.input_folder, wav_name), sample_rate=16000)
        data = _normalize(data)
        mgc = vocoder.melspectrogram(data, sample_rate=16000, num_mels=80)

        mgc_arr = torch.tensor(mgc, device=params.device).unsqueeze(0)
        with torch.no_grad():
            gsts, style = text2mel.mel2style(mgc_arr)
            copyfile(join(params.input_folder, txt_name), join(params.output_folder, tgt_txt_name))
            gst_arr = gsts.detach().cpu().numpy()
            np.save(join(params.output_folder, tgt_gst_name), gst_arr)

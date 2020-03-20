import torch
import json
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
    parser.add_option('--output', action='store', dest='output_file')
    parser.add_option('--device', action='store', dest='device', default='cpu')

    (params, _) = parser.parse_args(sys.argv)

    if not params.output_file or not params.input_folder:
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
        lab_name = base_name + '.lab'
        wav_name = base_name + '.orig.wav'
        if exists(join(params.input_folder, lab_name)) and exists(join(params.input_folder, wav_name)):
            if base_name not in final_list:
                final_list.append(base_name)

    files = sorted(final_list)
    dio = DatasetIO()
    vocoder = MelVocoder()
    mse_loss = torch.nn.MSELoss(reduction='mean')
    loss_func = mse_loss
    index = 0
    sent2score = {}
    for file in tqdm.tqdm(files):
        index += 1
        wav_name = file + '.orig.wav'
        txt_name = file + '.txt'
        lab_name = file + '.lab'

        data, sample_rate = dio.read_wave(join(params.input_folder, wav_name), sample_rate=16000)
        data = _normalize(data)
        mgc = vocoder.melspectrogram(data, sample_rate=16000, num_mels=80)

        mgc_arr = torch.tensor(mgc, device=params.device).unsqueeze(0)
        with torch.no_grad():
            json_obj = json.load(open(join(params.input_folder, lab_name)))
            trans = json_obj['transcription']
            speaker = json_obj['speaker']
            x = [[speaker, trans]]
            x_lengths = [len(trans)]
            y_lengths = [mgc.shape[0]]

            post_mgc, pred_stop, pred_att = text2mel(x, gs_mgc=mgc_arr, x_lengths=x_lengths,
                                                     y_lengths=y_lengths)
            post_mgc = post_mgc.view(-1)
            mgc_arr = mgc_arr.view(-1)
            m = min([post_mgc.shape[0], mgc_arr.shape[0]])
            lss = loss_func(post_mgc[:m], mgc_arr[:m])
            loss = lss.item()
            sent2score[file] = loss
    sorted_scores = {k: v for k, v in sorted(sent2score.items(), key=lambda item: item[1], reverse=True)}
    f = open(params.output_file, 'w')
    for sent in sorted_scores:
        f.write(sent + '\t' + str(sorted_scores[sent]) + '\n')
    f.close()

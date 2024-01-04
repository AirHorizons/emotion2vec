import traceback
from dataclasses import dataclass
import os, sys
from glob import glob
import argparse
import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F
import fairseq

@dataclass
class UserDirModule:
    user_dir: str

def get_io_dict_iemocap():
    '''
        create and return a dictionary of keys of iemocap audio file paths and values of corresponding output path
    '''
    input_base_dir = '/mnt/IEMOCAP_full_release'
    output_base_dir = './iemocap_outputs'
    io_dict = {}

    os.makedirs(output_base_dir, exist_ok=True)
    session_dirs = [os.path.join(input_base_dir, f'Session{i}/dialog/wav/') for i in range(1, 6)]

    for session_dir in session_dirs:
        for input_file in glob(session_dir + '[A-Za-z0-9_]*.wav'):
            file_name = input_file.split('/')[-1]
            output_file = os.path.join(output_base_dir, file_name)
            io_dict[input_file] = output_file

    return io_dict

def get_parser():
    parser = argparse.ArgumentParser(
        description="extract emotion2vec features for downstream tasks"
    )
    parser.add_argument('--model_dir', type=str, help='pretrained model', default='/mnt/ACM2024/feature_extractor/emotion2vec/upstream')
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for pre-trained model', default='/mnt/ACM2024/feature_extractor/emotion2vec/emotion2vec_base.pt')
    parser.add_argument('--granularity', type=str, help='which granularity to use, frame or utterance', default='utterance')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    model_dir = args.model_dir
    checkpoint_dir = args.checkpoint_dir
    granularity = args.granularity
    
    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()
    model.cuda()

    io_dict = get_io_dict_iemocap()
    for source_file, target_file in io_dict.items():

        wav, sr = sf.read(source_file)
        channel = sf.info(source_file).channels
        assert sr == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, source_file)
        
        # assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)
        if channel == 2:
            wav = wav.mean(axis=1)

        with torch.no_grad():
            source = torch.from_numpy(wav).float().cuda()
            if task.cfg.normalize:
                source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)
            try:
                feats = model.extract_features(source, padding_mask=None)
                feats = feats['x'].squeeze(0).cpu().numpy()
                if granularity == 'frame':
                    feats = feats
                elif granularity == 'utterance':
                    feats = np.mean(feats, axis=0)
                else:
                    raise ValueError("Unknown granularity: {}".format(args.granularity))
                np.save(target_file, feats)
                print(f'{target_file} created successfully')
            except:
                print("Error in extracting features from {}".format(source_file))
                traceback.print_exc()
                quit()
                


if __name__ == '__main__':
    # print(get_io_dict_iemocap())
    main()



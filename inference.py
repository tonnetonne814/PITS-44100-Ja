import warnings
warnings.filterwarnings(action='ignore')

import os
import utils
import argparse
import torch
from models import SynthesizerTrn
from text.symbols import symbol_len
from data_utils import infer_text_process

import soundcard as sc
import soundfile as sf

import time

def inference(args):

    hps     = utils.get_hparams(args)
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(hps.train.seed)

    net_g   = SynthesizerTrn(symbol_len(hps.data.languages),
                             hps.data.filter_length // 2 + 1,
                             hps.train.segment_size // hps.data.hop_length,
                             n_speakers=len(hps.data.speakers),
                             midi_start=hps.data.midi_start,
                             midi_end=hps.data.midi_end,
                             octave_range=hps.data.octave_range,
                             **hps.model,
                             ### add ###
                             sr=hps.data.sampling_rate,
                             W=hps.data.ying_window,
                             w_step=hps.data.ying_hop,
                             tau_max=hps.data.tau_max
                             ###########
                             ).to(device)
    
    # load checkpoint
    utils.load_checkpoint(args.model_path, model_g=net_g)

    # play audio by system default
    speaker = sc.get_speaker(sc.default_speaker().name)
    s_name_list = hps.data.speakers

    # parameter settings
    noise_scale     = torch.tensor(0.66)    # adjust z_p noise
    noise_scale_w   = torch.tensor(0.66)    # adjust SDP noise
    length_scale    = torch.tensor(1.0)     # adjust sound length scale (talk speed)
    max_len         = None                  # frame max length

    if args.is_save is True:
        n_save = 0
        save_dir = os.path.join("./infer_logs/", args.model)
        os.makedirs(save_dir, exist_ok=True)

    while True:
        
        print("\n")

        # get speaker id
        speaker_name = input("Enter speaker name. ==> ")
        if speaker_name=="":
            print("Empty input is detected... Exit...")
            break
        try:
            sid = int(s_name_list.index(speaker_name))
        except:
            while True:
                print(f"{speaker_name} does not existed. The list of speakers is displayed below.")
                print(s_name_list)
                speaker_name = input("Enter speaker name. ==> ")
                try:
                    sid = int(s_name_list.index(speaker_name))
                    break
                except:
                    print("~RETRY~")
                    pass
        sid = torch.unsqueeze(torch.tensor(sid, dtype=torch.int64), dim=0)

        # get text
        text = input("Enter text. ==> ")
        if text=="":
            print("Empty input is detected... Exit...")
            break
        
        # get shift pitch
        scope = input("Enter pitch shift value(int). -15〜15 ==> ")
        if scope=="":
            print("Empty input is detected... Exit...")
            break
        try:
            scope = int(scope)
        except:
            while True:
                print(f"Enter an integer value. {scope} is not integer value.")
                scope = input("Enter pitch shift value(int). -15〜15 ==> ")
                try:
                    scope = int(scope)
                    break
                except:
                    print("~RETRY~")
                    pass
        if scope > 15:
            scope = 15
            print("The upper limit of pitch shift value is 15.  15 is entered.")
        elif scope < -15:
            scope = -15
            print("The lower limit of pitch shift value is -15.  -15 is entered.")
        scope_shift = torch.tensor(scope, dtype=torch.int64)      # pitch adjust
        
        # measure the execution time 
        torch.cuda.synchronize()
        start = time.time()

        # required_grad is False
        with torch.inference_mode():
            x, t, x_lengths = infer_text_process(text=text, hps=hps)

            # generate audio
            y_hat, _, _, _ = net_g.infer(x=x.to(device), 
                                         t=t.to(device), 
                                         x_lengths=x_lengths.to(device), 
                                         sid=sid.to(device), 
                                         noise_scale=noise_scale.to(device), 
                                         noise_scale_w=noise_scale_w.to(device), 
                                         length_scale=length_scale.to(device), 
                                         max_len=max_len, 
                                         scope_shift=scope_shift.to(device))

        # measure the execution time 
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        print(f"Gen Time : {elapsed_time}")
        
        # play audio
        speaker.play(y_hat.view(-1).to('cpu').detach().numpy().copy(), hps.data.sampling_rate)
        
        # save audio
        if args.is_save is True:
            n_save += 1
            save_path = os.path.join(save_dir, str(n_save).zfill(3)+f"_{speaker_name}_scope={scope}_{text}.wav")
            data = y_hat.view(-1).to('cpu').detach().numpy().copy()
            sf.write(
                     file=save_path,
                     data=data,
                     samplerate=hps.data.sampling_rate,
                     format="WAV")
            print(f"Audio is saved at : {save_path}")


    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        required=True,     
                        help='Path to configuration file')
    parser.add_argument('--model',
                        type=str,
                        required=True,           
                        help='Model name')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to checkpoint')
    parser.add_argument('--is_save',
                        type=str,
                        default=True,
                        help='Whether to save output or not')
    args = parser.parse_args()
    
    inference(args)
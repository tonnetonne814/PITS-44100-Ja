import librosa
import os
import soundfile
from tqdm import tqdm
import random
import argparse
 
def main(dataset_dir:str = "./jvs_ver1/", target_sr:int = 22050):
    
    use_wav_folder = ["parallel100", "nonpara30"] #, "whisper10","falset10"]
    text_list = list()

    file_count = 0
    sentence_count = 0
    for spk_idx in range(100):
        spk_idx += 1
        spk_name = "jvs" + str(spk_idx).zfill(3)
        for folder in use_wav_folder:
            target_folder = os.path.join(dataset_dir, spk_name, folder, "wav24kHz16bit")
            results_folder = os.path.join(dataset_dir, spk_name, folder, f"wav{target_sr}Hz16bit")
            os.makedirs(results_folder, exist_ok=True)
            
            for filename in tqdm(os.listdir(target_folder), desc=spk_name):
                wav_path = os.path.join(target_folder, filename)
                y, sr = librosa.load(wav_path, sr=24000)
                y_converted = librosa.resample(y, orig_sr=24000, target_sr=target_sr)
                save_path = os.path.join(results_folder, filename)
                soundfile.write(save_path, y_converted, target_sr)
                
            txt_path = os.path.join(dataset_dir, spk_name, folder, "transcripts_utf8.txt")
            for txt in read_txt(txt_path):
                if txt == "\n":
                    continue
                name, sentence = txt.split(":")
                sentence = sentence.replace("\n", "")
                wav_filepath = os.path.join(spk_name, folder, f"wav{target_sr}Hz16bit", name+".wav")

                out_txt = wav_filepath + "|" + sentence + "|" + spk_name + "\n"
                text_list.append(out_txt)

    max_n = len(text_list)
    test_list = list()
    for _ in range(int(max_n * 0.005)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        test_list.append(txt)

        
    max_n = len(text_list)
    val_list = list()
    for _ in range(int(max_n * 0.005)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        val_list.append(txt)

    write_txt(f"./filelists/jvs_train_{target_sr}.txt", text_list)
    write_txt(f"./filelists/jvs_val_{target_sr}.txt", val_list)
    write_txt(f"./filelists/jvs_test_{target_sr}.txt", test_list)

    
    return 0

def read_txt(path):
    with open(path, mode="r", encoding="utf-8")as f:
        lines = f.readlines()
    return lines

def write_txt(path, lines):
    with open(path, mode="w", encoding="utf-8")as f:
        f.writelines(lines)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_path',
                        type=str,
                        required=True, 
                        help='Path to jvs corpus folder')
    parser.add_argument('--sampling_rate',
                        type=str,
                        required=True, 
                        help='Target sampling rate')

    args = parser.parse_args()

    main(dataset_dir=args.folder_path, target_sr=int(args.sampling_rate))
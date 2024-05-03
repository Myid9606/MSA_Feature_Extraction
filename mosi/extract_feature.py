from MSA_FET import FeatureExtractionTool
import shlex
import subprocess
import librosa
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor)
import torch
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import time
import csv

def audio_pad(raw_item,audio_len=400):
    if raw_item['audio'].shape[0] > audio_len:
        raw_item['audio_lengths'] = audio_len
        raw_item['audio'] = raw_item['audio'][:audio_len]
    elif raw_item['audio'].shape[0] < audio_len:
        raw_item['audio_lengths'] = raw_item['audio'].shape[0] 
        raw_item['audio'] = np.pad(raw_item['audio'], ((0, audio_len - raw_item['audio'].shape[0]), (0, 0)), 'constant')
    return raw_item

def text_pad(raw_item, text_len=50):
    if raw_item['text_bert'].shape[1] > text_len:
        raw_item['text_lengths'] = text_len
        raw_item['text_bert'] = raw_item['text_bert'][:, :text_len]
        raw_item['text'] = raw_item['text'][:text_len]
    elif raw_item['text_bert'].shape[1] < text_len:
        raw_item['text_lengths'] = raw_item['text_bert'].shape[1]
        raw_item['text_bert'] = np.pad(raw_item['text_bert'], ((0, 0), (0, text_len - raw_item['text_bert'].shape[1])), 'constant')
        raw_item['text'] = np.pad(raw_item['text'], ((0, text_len - raw_item['text'].shape[0]), (0, 0)), 'constant')      
    return raw_item

def vision_pad(raw_item,vision_len = 200):
    if raw_item['vision'].shape[0] > vision_len:
        raw_item['vision_lengths'] = vision_len
        raw_item['vision'] = raw_item['vision'][:vision_len]
    elif raw_item['vision'].shape[0] < vision_len:
        raw_item['vision_lengths'] = raw_item['vision'].shape[0]
        raw_item['vision'] = np.pad(raw_item['vision'], ((0, vision_len - raw_item['vision'].shape[0]), (0, 0)), 'constant')
    return raw_item

DEVICE = "cuda"
WAV2VEC_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

def execute_cmd(cmd: str) -> bytes:
    args = shlex.split(cmd)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ffmpeg", out, err)
    return out

def do_asr(audio_file) -> str:
    try:
        sample_rate = 16000
        speech, _ = librosa.load(audio_file, sr=sample_rate)
        processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL_NAME).to(DEVICE)
        features = processor(
            speech,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="longest"
        )
        with torch.no_grad():
            logits = model(features.input_values.to(DEVICE)).logits.cpu()[0]
        predicted_ids = torch.argmax(logits, dim=-1)
        asr_text = processor.decode(predicted_ids)
        return asr_text
    except Exception as e:
        raise e

def get_asr_text(mode,audio_type):
    video_list = glob(f'{VIDEO_PATH}/audio_{mode}_{audio_type}/miss*/*mp4')
    for item in tqdm(video_list):
        parent =  Path(item).parent
        name = Path(item).stem + '.txt'
        if os.path.exists(f"{parent}/{name}"):
            continue
        audio_save_path = "assets/temp/audio_{audio_type}.wav"
        cmd = f"ffmpeg -i {item} -vn -acodec pcm_s16le -ac 1 -y {audio_save_path}"
        execute_cmd(cmd)
        transcript = do_asr(audio_save_path)
        with open(f"{parent}/{name}", 'w') as f:
            f.write(transcript)
        os.remove(audio_save_path)

def get_video_feature(mode,video_type):
    openface_fet = FeatureExtractionTool(config='/home/liuweilong/MMSA-FET/config.json')
    # video_path = Path(dataset_dir) / 'Raw' / video_id / (str(clip_id) + '.mp4')
    video_list = glob(f'{VIDEO_PATH}/video_{mode}_{video_type}/miss*/*mp4')
    for item in tqdm(video_list):
        parent =  Path(item).parent
        name = Path(item).stem + '.pkl'
        if os.path.exists(f"{parent}/{name}"):
            continue
        try:
            openface_item = openface_fet.run_single(item)
        except Exception as e:
            openface_fet = FeatureExtractionTool(config='configs/openface.json')
            openface_item = openface_fet.run_single(item)
        pickle.dump(openface_item, open(f"{parent}/{name}",'wb'))

def get_audio_feature(mode,audio_type):
    opensmile_fet = FeatureExtractionTool(config='configs/opensmile.json')
    bert_fet = FeatureExtractionTool(config="configs/bert.json")
    video_list = glob(f'{VIDEO_PATH}/audio_{mode}_{audio_type}/miss*/*mp4')
    for item in tqdm(video_list):
        parent =  Path(item).parent
        name = Path(item).stem
        if os.path.exists(f"{parent}/{name}.pkl"):
            continue
        try:
            opensmile_item = opensmile_fet.run_single(item)
            bert_item = bert_fet.run_single(f"{parent}/{name}.txt")
            opensmile_item.update(bert_item)
        except Exception as e:
            opensmile_fet = FeatureExtractionTool(config='configs/opensmile.json')
            bert_fet = FeatureExtractionTool(config="configs/bert.json")
            opensmile_item = opensmile_fet.run_single(item)
            bert_item = bert_fet.run_single(f"{parent}/{name}.txt")
            opensmile_item.update(bert_item)
        pickle.dump(opensmile_item, open(f"{parent}/{name}.pkl",'wb'))

def get_simsv2_feature(mode):
    # feat_original_path = f"{DATA_PATH}/Processed/simsv2_unaligned.pkl"
    # with open(feat_original_path, 'rb') as f:
    #     feature = pickle.load(f)
    # feature = feature[mode]
    openface_fet = FeatureExtractionTool(config='configs/openface.json')
    opensmile_fet = FeatureExtractionTool(config='configs/opensmile.json')
    bert_fet = FeatureExtractionTool(config="configs/bert.json")
    video_list = glob(f'/home/sharing/disk2/zhangbaozheng/dataset/simsv2/{mode}_raw/*mp4')
    for item in tqdm(video_list):
        parent =  Path(item).parent
        name = Path(item).stem
        if os.path.exists(f"{parent}/{name}.pkl"):
            continue
        try:
            opensmile_item = opensmile_fet.run_single(item)
            opensmile_item = audio_pad(opensmile_item)
            openface_item = openface_fet.run_single(item)
            openface_item = vision_pad(openface_item)
            bert_item = bert_fet.run_single(f"{parent}/{name}.txt")
            
        except Exception as e:
            openface_fet = FeatureExtractionTool(config='configs/openface.json')
            opensmile_fet = FeatureExtractionTool(config='configs/opensmile.json')
            bert_fet = FeatureExtractionTool(config="configs/bert.json")
            opensmile_item = opensmile_fet.run_single(item)
            opensmile_item = audio_pad(opensmile_item)
            openface_item = openface_fet.run_single(item)
            openface_item = vision_pad(openface_item)
            bert_item = bert_fet.run_single(f"{parent}/{name}.txt")
            opensmile_item.update(openface_item)
        pickle.dump(opensmile_item, open(f"{parent}/{name}.pkl",'wb'))

def get_mintrec_feature():
    # feat_original_path = f"{DATA_PATH}/Processed/simsv2_unaligned.pkl"
    # with open(feat_original_path, 'rb') as f:
    #     feature = pickle.load(f)
    # feature = feature[mode]
    openface_fet = FeatureExtractionTool(config='configs/openface.json')
    opensmile_fet = FeatureExtractionTool(config='configs/opensmile.json')
    # bert_fet = FeatureExtractionTool(config="configs/bert.json")
    video_list = glob(f'/home/sharing/disk1/Datasets/SIMS3/SIMS3/video_clip_s/video*/*mp4')
    
    for item in tqdm(video_list):
        parent =  Path(item).parent
        name = Path(item).stem
        if os.path.exists(f"{parent}/{name}.pkl"):
            continue
        try:
            opensmile_item = opensmile_fet.run_single(item)
            opensmile_item = audio_pad(opensmile_item)
            openface_item = openface_fet.run_single(item)
            openface_item = vision_pad(openface_item)
            opensmile_item.update(openface_item)
            # bert_item = bert_fet.run_single(f"{parent}/{name}.txt")
            # bert_item = text_pad(bert_item)
            # opensmile_item.update(bert_item)
            pickle.dump(opensmile_item, open(f"{parent}/{name}.pkl",'wb'))
            time.sleep(0.5)
            
        except Exception as e:
            openface_fet = FeatureExtractionTool(config='configs/openface.json')
            opensmile_fet = FeatureExtractionTool(config='configs/opensmile.json')
            # bert_item = bert_fet.run_single(f"{parent}/{name}.txt")
            # bert_item = text_pad(bert_item)
            # opensmile_item.update(bert_item)
            time.sleep(1)

def simsv3_bert():
    with open('/home/sharing/disk1/Datasets/SIMS3/SIMS3/features/text_bert.pkl', 'rb') as f:
        bert_data = pickle.load(f)
    feat_original_path = f"/home/sharing/disk1/zhangbaozheng/code/interpretability_work/datasets/process/zbz_labels.pkl"
    with open(feat_original_path, 'rb') as f:
        feature = pickle.load(f)
    bert_fet = FeatureExtractionTool(config="configs/ch_bert.json")
    new_dict = bert_data
    for item in zip(feature['clip_id'],feature['text']):
        try:
            if item[0] in bert_data['id']:
                continue
            bert_item = bert_fet.run_single(in_file='', text=item[1])
            bert_item = text_pad(bert_item)
            new_dict['id'].append(item[0])
            new_dict['text_raw'].append(item[1])
            new_dict['text_bert'].append(bert_item['text_bert'])
            new_dict['text_lengths'].append(bert_item['text_lengths'])
        except:
            continue
    pickle.dump(new_dict, open(f"/home/sharing/disk1/Datasets/SIMS3/SIMS3/features/text_bert_1.pkl",'wb'))

def get_mosi_feature():
    # feat_original_path = f"{DATA_PATH}/Processed/simsv2_unaligned.pkl"
    # with open(feat_original_path, 'rb') as f:
    #     feature = pickle.load(f)
    # feature = feature[mode]
    openface_fet = FeatureExtractionTool(config='configs/openface.json')
    opensmile_fet = FeatureExtractionTool(config='configs/opensmile.json')
    bert_fet = FeatureExtractionTool(config="configs/bert.json")
    # video_path = Path(dataset_dir) / 'Raw' / video_id / (str(clip_id) + '.mp4')
    video_list = glob((f'/home/liuweilong/MMSA-FET/MOSI/Raw/**/*.mp4'), recursive=True)
    # video_list = glob(f'/home/sharing/disk2/zhangbaozheng/dataset/simsv2/{mode}_raw/*mp4')
    for item in tqdm(video_list):
            parent =  Path(item).parent
            name = Path(item).stem
            if os.path.exists(f"{parent}/{name}.pkl"):
                continue

            opensmile_item = opensmile_fet.run_single(item)
            opensmile_item = audio_pad(opensmile_item)
            openface_item = openface_fet.run_single(item)
            openface_item = vision_pad(openface_item)
            
            videoid = parent.name
            clipid = name
                # 遍历每一行数据
            with open('/home/liuweilong/MMSA-FET/MOSI/regression_label.csv', 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # 检查video_id和clip_id是否匹配
                    if row['video_id'] == str(videoid) and row['clip_id'] == str(clipid):
                        rawtext = row['text']
                        mode =    row['mode']
            bert_item = bert_fet.run_single(in_file=str('/home/liuweilong/MMSA-FET/MOSI/Raw/{videoid}/{clipid}'),text = rawtext)
            bert_item = text_pad(bert_item)
            opensmile_item.update(openface_item)
            opensmile_item.update(bert_item)
            # csv mode  单个的pk
            # l
                
            if not os.path.exists(f"/home/liuweilong/MMSA-FET/MOSI/{mode}_Raw/{videoid}"):
                os.makedirs(f"/home/liuweilong/MMSA-FET/MOSI/{mode}_Raw/{videoid}")
            pickle.dump(opensmile_item, open(f"/home/liuweilong/MMSA-FET/MOSI/{mode}_Raw/{videoid}/{name}.pkl",'wb'))  



if __name__ == "__main__":
    DATA_PATH = '/home/liuweilong/MMSA-FET/MOSI'
    VIDEO_PATH = '/home/liuweilong/MMSA-FET/MOSI'
    get_mosi_feature()
    # audio_type = 'color_w'
    # get_asr_text(mode,audio_type)
    # audio_type = 'bg_park'
    # get_asr_text(mode,audio_type)
    
    # video_type = 'gblur'
    # get_video_feature(mode,video_type)

    # video_type = 'impulse_value'
    # get_video_feature(mode,video_type)

    # audio_type = 'color_w'
    # get_audio_feature(mode,audio_type)

    # audio_type = 'bg_park'
    # get_audio_feature(mode,audio_type)


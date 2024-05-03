import csv

data = {}

with open('/home/liuweilong/MMSA-FET/MOSI/regression_label.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        mode = row['mode']
        
        if mode not in data:
            data[mode] = {
                'id': [],
                'audio': [],
                'vision': [],
                'raw_text': [],
                'text_bert': [],
                'text': [],
                'audio_lengths': [],
                'vision_lengths': [],
                'annotations': [],
                'regression_labels': [],
            }
        
        video_id = row['video_id']
        clip_id = row['clip_id']
        id = f"{video_id}$_${clip_id}"
        
        data[mode]['id'].append(id)
        # data[mode]['audio'].append(row['audio'])
        # data[mode]['vision'].append(row['vision'])
        data[mode]['raw_text'].append(row['text'])
        # data[mode]['text_bert'].append(row['text_bert'])
        # data[mode]['text'].append(row['text'])
        # data[mode]['audio_lengths'].append(row['audio_lengths'])
        # data[mode]['vision_lengths'].append(row['vision_lengths'])
        data[mode]['annotations'].append(row['annotation'])
        data[mode]['regression_labels'].append(row['label'])
import pickle

# 假设保存路径为'/path/to/save/data.pkl'
save_path = '/home/liuweilong/MMSA-FET/MOSI/Processed/nofeat_mosi.pkl'

# 保存data字典为.pkl文件
with open(save_path, 'wb') as f:
    pickle.dump(data, f)

import csv
'''
Afterwards, use nofeat_mosi. py to save the existing information
as a file to be supplemented, 
which contains information such as ID and source text,
but the feature bar has no content
'''
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

save_path = 'MOSI/Processed/nofeat_mosi.pkl'

with open(save_path, 'wb') as f:
    pickle.dump(data, f)

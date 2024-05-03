import pickle
from glob import glob
import os
import numpy as np
from pathlib import Path



def init_list():
    new_list = {
        'id': [],
        'audio': [],
        'vision': [],
        'raw_text': [],
        'text_bert': [],
        'audio_lengths': [],
        'vision_lengths': [],
        'annotations': [],
        'regression_labels': []
    }
    return new_list



def mosipkl(data):
    # {mode}_Raw/{videoid}/{name}.pkl"
    content = {}
    for mode in ['train','valid','test']:
        #
        
        new_list = {
            'id': [],
            'audio': [],
            'vision': [],
            'raw_text': [],
            'text_bert': [],
            'text':[],
            'audio_lengths': [],
            'vision_lengths': [],
            'annotations': [],
            'regression_labels':[],
        }
        noNosie = data[mode]
        for item in zip(noNosie['id'],noNosie['raw_text'],noNosie['annotations'],noNosie['regression_labels']):
            
            # name = item[0].replace('$_$','_')
            parent, subid = item[0].split('$_$')
            # 将每个字段的值添加到new_list字典中对应的列表中
            new_list['id'].append(item[0])
            new_list['raw_text'].append(item[1])
            # new_list['text_bert'].append(item[2])
            # new_list['text'].append(item[3])
            new_list['annotations'].append(item[2])
            new_list['regression_labels'].append(item[3])

            with open(f'{SAVE_PATH}/{mode}_Raw/{parent}/{subid}.pkl', 'rb') as f:
                raw_item = pickle.load(f)#pickle.load()赋值给raw_item变量
                
            # 将raw_item字典中的一些字段值添加到new_list字典中对应的列表中d
            new_list['text'].append(raw_item['text'])
            new_list['text_bert'].append(raw_item['text_bert'])
            new_list['audio'].append(raw_item['audio'])
            new_list['vision'].append(raw_item['vision'])
            if raw_item['audio_lengths'] > 1432:
                raw_item['audio_lengths'] = 1432
            if raw_item['vision_lengths'] > 143:
                raw_item['vision_lengths'] = 143
            new_list['audio_lengths'].append(raw_item['audio_lengths'])
            new_list['vision_lengths'].append(raw_item['vision_lengths'])

        for key, value in new_list.items():
            new_list[key] = np.array(value)
        content[mode] = new_list

    pickle.dump(content, open(f"{DATA_PATH}/Processed/unaligned_mosi_v200.pkl",'wb'))





if __name__ == "__main__":
    DATA_PATH = '/home/liuweilong/MMSA-FET/MOSI'
    SAVE_PATH = '/home/liuweilong/MMSA-FET/MOSI'
    with open(f"{DATA_PATH}/Processed/nofeat_mosi.pkl", 'rb') as f:
        data = pickle.load(f)
    mosipkl(data)

    # data_type = 'impulse_value'
    # # gblur impulse_value
    # video_feature(mode, data_type)

    # data_type = 'bg_park'/home/liuweilong/MMSA-FET/MOSI/Processed/nofeat_mosi.pkl
    # color_w bg_park
    # audio_feature(mode, data_type)



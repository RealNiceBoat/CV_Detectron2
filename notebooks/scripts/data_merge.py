'''
Attempted to merge all the datasets, out of RAM while creating the dict. No way to handle unless someone feels like investigating JSON streamed output.
'''
import json
import pandas as pd
import os
os.chdir('./data')

dataset = {}
dataset['images'] = []
dataset['annotations'] = []
dataset['categories'] = [
    {"id": 1, "name": "tops"},
    {"id": 2, "name": "trousers"},
    {"id": 3, "name": "outerwear"},
    {"id": 4, "name": "dresses"},
    {"id": 5, "name": "skirts"}
]

remap_ims = {
    'df2':{},
    'modanet':{},
    'til2020':{}
}

total_len = 0
total_ann_len = 0
for ds in remap_ims.keys():
    with open('./'+ds+'/train.json','r') as f: data = json.load(f)
    print(len(data['images']))
    #remap images
    df_ims = pd.DataFrame(data['images'])
    df_ims['file_name'] = (ds+'/train/') + df_ims.file_name.astype(str)
    new_ids = list(range(total_len,total_len+len(df_ims)))
    remap_ims[ds] = dict(zip(list(df_ims.id),new_ids))
    df_ims['id'] = new_ids
    total_len+=len(df_ims)
    dataset['images'] += df_ims.iloc[].to_dict('records')
    del df_ims
    #remap anns
    df_ann = pd.DataFrame(data['annotations'])
    df_ann['image_id'] = df_ann.image_id.replace(remap_ims[ds])
    df_ann['id'] = range(total_ann_len,total_ann_len+len(df_ann))
    total_ann_len += len(df_ann)
    dataset['annotations'] += df_ann.to_dict('records')
    del df_ann
print(total_len)

with open('train.json','w') as f: json.dump(dataset,f)

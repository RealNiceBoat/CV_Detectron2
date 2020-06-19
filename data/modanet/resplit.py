import json
import pandas as pd
import os
os.chdir('./data/modanet')
with open('train.json','r') as f: data = json.load(f)

remap = {
    5:3,
    6:4,
    8:2,
    9:1,
    10:2,
    11:5
}

df = pd.DataFrame(data['annotations'])
df = df[df.category_id.isin(remap.keys())]
df['category_id'] = df.category_id.replace(remap)
data['annotations'] = df.to_dict('records')

keep_images = list(set(df['image_id']))
df = pd.DataFrame(data['images'])
df = df[df.id.isin(keep_images)]
data['images'] = df.to_dict('records')
data['categories'] = [
    {"id": 1, "name": "tops"},
    {"id": 2, "name": "trousers"},
    {"id": 3, "name": "outerwear"},
    {"id": 4, "name": "dresses"},
    {"id": 5, "name": "skirts"}
]
print(f'images remaining: {len(df)}')
with open('filtered_train.json','w') as f: json.dump(data,f)
'''
MODANET:
{'supercategory': 'fashion', 'id': 1, 'name': 'bag'},
{'supercategory': 'fashion', 'id': 2, 'name': 'belt'},
{'supercategory': 'fashion', 'id': 3, 'name': 'boots'},
{'supercategory': 'fashion', 'id': 4, 'name': 'footwear'},
{'supercategory': 'fashion', 'id': 5, 'name': 'outer'},
{'supercategory': 'fashion', 'id': 6, 'name': 'dress'}, 
{'supercategory': 'fashion', 'id': 7, 'name': 'sunglasses'},
{'supercategory': 'fashion', 'id': 8, 'name': 'pants'},
{'supercategory': 'fashion', 'id': 9, 'name': 'top'},
{'supercategory': 'fashion', 'id': 10, 'name': 'shorts'},
{'supercategory': 'fashion', 'id': 11, 'name': 'skirt'},
{'supercategory': 'fashion', 'id': 12, 'name': 'headwear'},
{'supercategory': 'fashion', 'id': 13, 'name': 'scarf/tie'}

TIL:
{"id": 1, "name": "tops"},
{"id": 2, "name": "trousers"},
{"id": 3, "name": "outerwear"},
{"id": 4, "name": "dresses"},
{"id": 5, "name": "skirts"}
'''
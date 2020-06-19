import json
from PIL import Image
import numpy as np

dataset = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

dataset['categories'].append({
    'id': 1,
    'name': "short_sleeved_shirt",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 2,
    'name': "long_sleeved_shirt",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 3,
    'name': "short_sleeved_outwear",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 4,
    'name': "long_sleeved_outwear",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 5,
    'name': "vest",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 6,
    'name': "sling",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 7,
    'name': "shorts",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 8,
    'name': "trousers",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 9,
    'name': "skirt",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 10,
    'name': "short_sleeved_dress",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 11,
    'name': "long_sleeved_dress",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 12,
    'name': "vest_dress",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 13,
    'name': "sling_dress",
    'supercategory': "clothes"
})
remap = {
    1:1,
    2:1,
    3:3,
    4:3,
    5:3,
    7:2,
    8:2,
    9:5,
    10:4,
    11:4,
    12:4,
    13:4
}

dataset['categories'] = [
    {"id": 1, "name": "tops"},
    {"id": 2, "name": "trousers"},
    {"id": 3, "name": "outerwear"},
    {"id": 4, "name": "dresses"},
    {"id": 5, "name": "skirts"}
]

sub_index = 0 # the index of ground truth instance
num_images = 191961
for num in range(1,num_images+1):
    json_name = './annos/' + str(num).zfill(6)+'.json'
    image_name = './image/' + str(num).zfill(6)+'.jpg'

    if (num>=0):
        imag = Image.open(image_name)
        width, height = imag.size
        with open(json_name, 'r') as f:
            temp = json.loads(f.read())
            pair_id = temp['pair_id']

            dataset['images'].append({
                'file_name': str(num).zfill(6) + '.jpg',
                'id': num,
                'license': 0,
                'width': width,
                'height': height
            })
            for i in temp:
                if i == 'source' or i=='pair_id':
                    continue
                else:
                    points = np.zeros(294 * 3)
                    sub_index = sub_index + 1
                    box = temp[i]['bounding_box']
                    w = box[2]-box[0]
                    h = box[3]-box[1]
                    x_1 = box[0]
                    y_1 = box[1]
                    bbox=[x_1,y_1,w,h]
                    cat = temp[i]['category_id']
                    if cat not in remap.keys(): continue
                    style = temp[i]['style']
                    seg = temp[i]['segmentation']
                    landmarks = temp[i]['landmarks']

                    dataset['annotations'].append({
                        'area': w*h,
                        'bbox': bbox,
                        'category_id': remap[cat],
                        'id': sub_index,
                        'image_id': num,
                        'iscrowd': 0,
                        'segmentation': seg
                    })


json_name = './deepfashion2.json'
with open(json_name, 'w') as f:
  json.dump(dataset, f)






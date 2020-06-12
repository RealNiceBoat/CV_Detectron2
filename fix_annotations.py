#Paths
from pathlib import Path
base_folder = Path('.')
data_folder = base_folder/'til2020'
train_imgs_folder = data_folder/'train'/'train'
train_annotations = data_folder/'train.json'
val_imgs_folder = data_folder/'val'/'val'
val_annotations = data_folder/'val.json'

from PIL import Image
import json
import pandas as pd
def applyFix(anns,idir):
    with open(anns,'r+') as f:
        data = json.load(f)
        for image in data['images']: image['width'],image['height'] = Image.open(idir/image["file_name"]).size
        df = pd.DataFrame(data['annotations'])
        df.id = range(len(df))
        data['annotations'] = df.to_dict('records')
        f.seek(0)
        json.dump(data,f)
        f.truncate()

applyFix(train_annotations,train_imgs_folder)
applyFix(val_annotations,val_imgs_folder)   
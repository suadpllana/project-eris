import base64
import io
import shutil
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

SOURCE_WIDTH,SOURCE_HEIGHT,CHANNELS=192,160,4
WIDTH,HEIGHT=SOURCE_WIDTH//2,SOURCE_HEIGHT//2
EXPECTED_ROWS,EXPECTED_TRAIN,EXPECTED_TEST=2700,2400,300

def decode_source(value):
    raw=zlib.decompress(base64.b64decode(value,validate=True))
    if len(raw)!=CHANNELS*SOURCE_HEIGHT*SOURCE_WIDTH:raise ValueError('raw attribution tensor has the wrong byte count')
    tensor=np.frombuffer(raw,dtype=np.uint8).reshape(CHANNELS,SOURCE_HEIGHT,SOURCE_WIDTH)
    if not np.all(tensor.astype(np.uint16).sum(axis=0)==255):raise ValueError('raw attribution channels must sum to 255')
    return tensor

def pool_tensor(tensor):
    # Mean of each 2x2 pixel block, then largest-remainder rounding so the four channels still sum to 255.
    exact=tensor.astype(np.float64).reshape(CHANNELS,HEIGHT,2,WIDTH,2).mean(axis=(2,4))
    floor=np.floor(exact).astype(np.int16);shortfall=255-floor.sum(axis=0)
    order=np.argsort(-(exact-floor),axis=0,kind='stable');rank=np.argsort(order,axis=0,kind='stable')
    pooled=floor+(rank<shortfall[None]).astype(np.int16)
    if not np.all(pooled.sum(axis=0)==255):raise ValueError('pooled attribution does not sum to 255')
    return pooled.astype(np.uint8)

def encode_png(light):
    buffer=io.BytesIO();Image.fromarray(np.ascontiguousarray(light.transpose(1,2,0))).save(buffer,format='PNG',compress_level=9)
    return base64.b64encode(buffer.getvalue()).decode('ascii')

def prepare(raw:Path,public:Path,private:Path)->None:
    raw,public,private=Path(raw),Path(public),Path(private);labels=pd.read_csv(raw/'labels.csv',dtype=str,keep_default_na=False)
    required={'id','filename','scene_id','eval_domain','attribution_b64'}
    if set(labels.columns)!=required or len(labels)!=EXPECTED_ROWS:raise ValueError('invalid raw labels')
    if labels.id.duplicated().any() or labels.filename.duplicated().any():raise ValueError('IDs and filenames must be unique')
    counts=labels.groupby('scene_id').size();test_scenes=set(counts[counts==1].index)
    if int((counts==1).sum())!=EXPECTED_TEST or not counts.isin([1,2]).all():raise ValueError('scene grouping is invalid')
    labels['split']=labels.scene_id.map(lambda x:'test' if x in test_scenes else 'train')
    if (labels.split=='train').sum()!=EXPECTED_TRAIN or (labels.split=='test').sum()!=EXPECTED_TEST:raise ValueError('split size is invalid')
    train_images,test_images=public/'train_images',public/'test_images';train_images.mkdir(parents=True,exist_ok=True);test_images.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    labels=labels.sort_values('id',kind='mergesort').reset_index(drop=True);paths=[];payloads=[]
    for row in labels.itertuples(index=False):
        source=raw/'images'/row.filename;dest=train_images if row.split=='train' else test_images;name=row.id+'.png';shutil.copyfile(source,dest/name);paths.append(row.split+'_images/'+name)
        payloads.append(encode_png(pool_tensor(decode_source(row.attribution_b64))[:3]))
    labels['image']=paths;labels['attribution_png']=payloads
    train=labels.loc[labels.split=='train',['id','image','scene_id']];targets=labels.loc[labels.split=='train',['id','attribution_png']]
    test=labels.loc[labels.split=='test',['id','image','scene_id']];answers=labels.loc[labels.split=='test',['id','eval_domain','attribution_png']]
    sample=test[['id']].copy();sample['attribution_png']=encode_png(np.zeros((3,HEIGHT,WIDTH),dtype=np.uint8))
    train.to_csv(public/'train.csv',index=False,lineterminator='\n');targets.to_csv(public/'train_targets.csv',index=False,lineterminator='\n');test.to_csv(public/'test.csv',index=False,lineterminator='\n');sample.to_csv(public/'sample_submission.csv',index=False,lineterminator='\n');answers.to_csv(private/'answers.csv',index=False,lineterminator='\n')

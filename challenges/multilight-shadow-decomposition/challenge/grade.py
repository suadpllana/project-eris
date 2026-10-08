import base64
import binascii
import io
import math
import numpy as np
import pandas as pd
from PIL import Image

WIDTH,HEIGHT=96,80
PNG_SIGNATURE=b'\x89PNG\r\n\x1a\n'
MAX_PAYLOAD_CHARS=1_000_000

def decode_payload(value):
    if value is None or (isinstance(value,float) and not math.isfinite(value)) or not str(value).strip():raise ValueError('attribution_png must be a nonempty Base64 PNG')
    text=str(value).strip()
    if len(text)>MAX_PAYLOAD_CHARS:raise ValueError('attribution_png payload is too large')
    try:raw=base64.b64decode(text,validate=True)
    except (binascii.Error,ValueError) as error:raise ValueError('attribution_png is not valid Base64') from error
    if len(raw)<26 or raw[:8]!=PNG_SIGNATURE or raw[12:16]!=b'IHDR':raise ValueError('attribution_png must decode to a PNG file')
    if raw[24]!=8 or raw[25]!=2:raise ValueError('attribution PNG must be 8-bit RGB (PNG colour type 2)')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format!='PNG' or image.mode!='RGB' or image.size!=(WIDTH,HEIGHT):raise ValueError('attribution PNG must be RGB with width 96 and height 80')
            light=np.asarray(image,dtype=np.uint8).transpose(2,0,1)
    except ValueError:raise
    except Exception as error:raise ValueError('attribution PNG could not be decoded') from error
    if light.shape!=(3,HEIGHT,WIDTH):raise ValueError('attribution tensor has the wrong shape')
    if np.any(light.astype(np.uint16).sum(axis=0)>255):raise ValueError('light channels must sum to at most 255 per pixel')
    return light.astype(np.float32)/255.0

def image_score(prediction,truth):
    dice=[];l1_skill=[]
    for channel in range(3):
        p,t=prediction[channel],truth[channel]
        dice_denominator=np.sum(p*p)+np.sum(t*t)+1e-7
        mass_denominator=np.sum(p)+np.sum(t)+1e-7
        dice.append((2*np.sum(p*t)+1e-7)/dice_denominator)
        l1_skill.append(max(0.0,1.0-np.sum(np.abs(p-t))/mass_denominator))
    return .7*float(np.mean(dice))+.3*float(np.mean(l1_skill))

def grade(submission:pd.DataFrame,answers:pd.DataFrame)->float:
    if not {'id','attribution_png'}.issubset(submission.columns):raise ValueError('Submission must contain columns id,attribution_png')
    if not {'id','eval_domain','attribution_png'}.issubset(answers.columns):raise ValueError('Answers are missing required columns')
    submitted=submission[['id','attribution_png']].copy();truth=answers[['id','eval_domain','attribution_png']].copy()
    if submitted.id.isna().any() or truth.id.isna().any():raise ValueError('IDs must be present')
    submitted.id=submitted.id.astype(str);truth.id=truth.id.astype(str)
    if submitted.id.duplicated().any() or truth.id.duplicated().any():raise ValueError('IDs must be unique')
    missing=set(truth.id)-set(submitted.id)
    if missing:raise ValueError(f'Submission is missing {len(missing)} scored IDs')
    aligned=truth.merge(submitted,on='id',how='left',validate='one_to_one',suffixes=('_true','_pred'));domains=[]
    for _,frame in aligned.sort_values('id',kind='mergesort').groupby('eval_domain',sort=True):domains.append(float(np.mean([image_score(decode_payload(row.attribution_png_pred),decode_payload(row.attribution_png_true)) for row in frame.itertuples(index=False)])))
    if not domains or not all(math.isfinite(x) for x in domains):raise ValueError('No scorable answer domains')
    return float(np.clip(.5*np.mean(domains)+.5*np.min(domains),0,1))

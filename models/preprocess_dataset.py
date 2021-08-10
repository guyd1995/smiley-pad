from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
import requests
import os
from glob import glob
import numpy as np
from collections import Counter
from PIL import Image
from concurrent.futures import as_completed, ThreadPoolExecutor
from requests_futures.sessions import FuturesSession
from argparse import ArgumentParser

original_csv_header = [
          'url0', 'left0', 'right0', 'top0', 'bottom0', 
          'url1', 'left1', 'right1', 'top1', 'bottom1', 
          'url2', 'left2', 'right2', 'top2', 'bottom2', 
           'triplet_type',
           'annot_id0', 'annot0',
           'annot_id1', 'annot1',
           'annot_id2', 'annot2',
           'annot_id3', 'annot3',
           'annot_id4', 'annot4',
           'annot_id5', 'annot5'
          ]


def _get_fec_urls(folder='FEC_dataset', force_https=False):
    csv_folder = folder + "/FEC_dataset"
    train_path = csv_folder + "/faceexp-comparison-data-train-public.csv"
    test_path = csv_folder + "/faceexp-comparison-data-test-public.csv"
    train_csv = pd.read_csv(train_path, header=None, names=original_csv_header)
    test_csv = pd.read_csv(test_path, header=None, names=original_csv_header)

    urls_df = pd.concat([train_csv
               .rename(columns={'url0': 'url', 'top0': 'top', 'bottom0': 'bottom', 'left0': 'left',
                        'right0': 'right'}),
              train_csv
               .rename(columns={'url1': 'url', 'top1': 'top', 'bottom1': 'bottom', 'left1': 'left',
                        'right1': 'right'}),
              train_csv
               .rename(columns={'url2': 'url', 'top2': 'top', 'bottom2': 'bottom', 'left2': 'left', 
                        'right2': 'right'}),
              test_csv
               .rename(columns={'url0': 'url', 'top0': 'top', 'bottom0': 'bottom', 'left0': 'left',
                        'right0': 'right'}),
              test_csv
               .rename(columns={'url1': 'url', 'top1': 'top', 'bottom1': 'bottom', 'left1': 'left',
                        'right1': 'right'}),
              test_csv
               .rename(columns={'url2': 'url', 'top2': 'top', 'bottom2': 'bottom', 'left2': 'left', 
                        'right2': 'right'}),
              ])[['url', 'top', 'bottom', 'left', 'right']]
    
    urls_df = urls_df.assign(rect=urls_df.apply(lambda r: (r['left'], r['right'], r['top'], r['bottom']), axis=1))
    urls_df = urls_df.groupby('url').agg({'rect': lambda x: list(set(x))}).reset_index()
    
    all_urls = urls_df.url
    all_urls = list(all_urls)
    urls_df['orig_url'] = all_urls
    if force_https:
        arr = []
        for x in tqdm(all_urls):
            assert isinstance(x, str)
            assert (x.startswith('http://') or x.startswith('https://'))
            if x.startswith('http://'):
                x = "https://" + x[len("http://"):]
            arr += [x]
        all_urls = arr
    urls_df.url = all_urls
    return train_csv, test_csv, urls_df


def _fetch_fec_imgs(urls_df, max_imgs=None, timeout=None, folder="FEC_dataset", start_pos=0):
    all_urls = list(urls_df.url)
    max_imgs = min(len(all_urls)-start_pos, max_imgs) if max_imgs is not None else len(all_urls)-start_pos
    assert max_imgs >= 0
    randomized_urls = np.random.permutation(all_urls)[start_pos:start_pos + max_imgs]
    
    session = FuturesSession()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {session.get(url): url for url in randomized_urls}
        pbar = tqdm(as_completed(future_to_url), initial=start_pos, total=start_pos + len(future_to_url.keys()))
        for future in pbar:
            try:
                resp = future.result(timeout=timeout)
            except:
                continue
            url = future_to_url[future]
            real_idx = all_urls.index(url)
            img_folder = f'{folder}/images'
            img_path = f'{img_folder}/{real_idx}.jpg'
            pbar.set_postfix_str(url)

            with open(img_path, 'wb') as f:
                f.write(resp.content)
            try:
                with open(img_path, 'rb') as f_img:
                    whole_img = Image.open(f_img)
                    if whole_img is not None:
                        w, h = whole_img.width, whole_img.height
                        for rect_idx, rect in enumerate(urls_df.iloc[real_idx].rect):
                            left, right, top, bottom = np.floor(np.array(rect) * np.array([w, w, h, h])).astype(int)
                            try:
                                whole_img.crop((left, top, right, bottom)).save(f'{img_folder}/{real_idx}_{rect_idx}.jpg')
                            except:
                                print("cropping error")
            except:
                print("img opening/cropping error")

            os.remove(img_path)


def _normalize_csvs(train_csv, test_csv, urls_df, folder='FEC_dataset'):
    all_imgs = list(map(os.path.normpath, glob(f'{folder}/images/*.jpg')))
    all_urls = list(urls_df.orig_url)
    all_urls = {all_urls[i]: i for i in range(len(all_urls))}
    for name, df in [('train', train_csv), ('test', test_csv)]:
        new_df = {'img0': [], 'img1': [], 'img2': [], 'target': [], 'triplet_type': []}
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            for i in range(3):
                idx1 = all_urls[row[f'url{i}']]
                idx2 = urls_df.iloc[idx1].rect.index((row[f'left{i}'], row[f'right{i}'], row[f'top{i}'], row[f'bottom{i}']))
                img_path = os.path.normpath(f"{folder}/images/{idx1}_{idx2}.jpg")
                new_df[f'img{i}'].append(img_path if img_path in all_imgs else None)
            
            cntr = Counter([row['annot0'], row['annot1'], row['annot2'], row['annot3'], row['annot4'], row['annot5']])
            (target, freq), = cntr.most_common(1)
            new_df['target'].append(target-1 if freq >= 4 else None)
            new_df['triplet_type'].append(['ONE_CLASS_TRIPLET', 'TWO_CLASS_TRIPLET', 
                                           'THREE_CLASS_TRIPLET'].index(row['triplet_type']))
        new_df = pd.DataFrame(new_df)
        new_df = new_df.dropna()
        new_df.to_csv(f"{folder}/processed_{name}.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max-imgs", default=None, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--timeout", default=None, type=int)
    args = parser.parse_args()
    max_imgs = args.max_imgs
    seed = args.seed
    start_pos = args.start
    timeout = args.timeout
    
    print(f"setting seed to {seed}")
    np.random.seed(seed)    
    print("organize urls..")
    train_csv, test_csv, urls_df = _get_fec_urls()
    print("fetch images..")
    _fetch_fec_imgs(urls_df, max_imgs=max_imgs, timeout=timeout, start_pos=start_pos)
    print("process csvs..")
    _normalize_csvs(train_csv, test_csv, urls_df)

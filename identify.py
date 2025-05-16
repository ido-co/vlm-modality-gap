import os
import pandas as pd
import torch
from tqdm import tqdm
tqdm.pandas()
from setup import setup
from utils import *
import argparse
from PIL import Image

def main(args):
    print(f"Started identify") 
    seedEverything()
    
    model_name = args.model_name
    batch_size = args.batch_size
    
    mp = setup(model_name)
    df = pd.read_csv(args.dataset)

    results = []
    batch = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        full_path = get_full_path(row['resized_path'])
        raw_image = Image.open(full_path)
        query, _ = get_identify_query_and_prefix_by_type(row['type'])
        prompt = mp.get_prompt(query=query, image=True)
        batch.append((raw_image, prompt, row))
        
        if len(batch) >= batch_size:
            process_batch(batch, results, mp)
            batch = []
        
        if i % 50 == 0:
            save_results(results, model_name, args.csv, i, dir=f'tmp_{model_name}')
    
    if batch:
        process_batch(batch, results, mp)
    
    save_results(results, model_name, args.csv, suffix='id')


def process_batch(batch, results, mp):
    images, prompts, rows = zip(*batch)
    with torch.no_grad():
        inputs = mp.processor(images=list(images), text=list(prompts), return_tensors='pt', padding=True, truncation=True)
        inputs = inputs.to(mp.model.device, torch.float16)
        output = mp.model.generate(**inputs, max_new_tokens=20, do_sample=False, 
                                output_hidden_states=False, return_dict_in_generate=True, use_cache=True)
    
        generated_texts = mp.processor.batch_decode(output['sequences'], skip_special_tokens=True)
    
    for row, generated_text in zip(rows, generated_texts):
        correct = identified(generated_text, row['subject'], row['aliases'])
        results.append({'type': row['type'], 's_uri': row['s_uri'], 'subject': row['subject'], 
                        'path': row['resized_path'], 'identified': correct, 'generation': generated_text})


def save_results(results, model_name, csv_path, suffix=None, dir='results'):
    res_df = pd.DataFrame.from_records(results)
    if suffix == 'id':
        res_df = res_df[res_df['identified']]
    _, filename = os.path.split(csv_path)
    filename, _ = os.path.splitext(filename)
    res_file = f"{model_name}_{filename}.csv" if not suffix else f"{model_name}_{filename}_{suffix}.csv"
    os.makedirs(os.path.join(dir, model_name), exist_ok=True)
    res_df.to_csv(os.path.join(dir, model_name, res_file), index=False)


def get_exp_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model-name', type=str, default="llava_7B")
    parser.add_argument('--dataset', type=str, default="data/entities.csv")
    parser.add_argument('--batch-size', type=int, default=4, help='Number of images to process in a batch')
    return parser


if __name__ == "__main__":
    parser = get_exp_parser()
    args = parser.parse_args()
    main(args)

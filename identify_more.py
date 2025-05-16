from setup_more import *
import argparse
import os
import pandas as pd
import torch
from utils import *
from tqdm import tqdm
tqdm.pandas()

def main(args):
    print(f"Started identify") 
    seedEverything()
    
    model_path = MODEL_NAME_PATH[args.model_name]
    model_name = args.model_name
    tokenizer, model, prepare_inputs = initialize_model(model_path)
    
    df = pd.read_csv(args.dataset)

    results = []

    for i, row in tqdm(df.iterrows(), total=len(df)):       
        query, _ = get_identify_query_and_prefix_by_type(row['type'])
        prompt = PROMPT_IM_TEMP.format(query=query)
        input_ids, images_tensor, image_sizes = prepare_inputs(get_full_path(row['resized_path']), prompt)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=100,
                use_cache=True,
            )
        
        generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        correct = identified(generated_text, row['subject'], row['aliases'])
        results.append({'type':row['type'], 's_uri':row['s_uri'] ,'subject':row['subject'], 'path':row['resized_path'], 'identified':correct, 'generation':generated_text})
        if (i+1) % 200 == 0:
            save_results(results, model_name, args.csv, i, dir=f'tmp_{model_name}')
    
    save_results(results, model_name, args.csv, suffix='id')

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
    parser.add_argument("--model-name", type=str, default="clip")
    parser.add_argument('--dataset', type=str, default="data/entities.csv")
    parser.add_argument('--batch-size', type=int, default=4, help='Number of images to process in a batch')

    return parser


if __name__ == "__main__":
    parser = get_exp_parser()
    args = parser.parse_args()
    main(args)

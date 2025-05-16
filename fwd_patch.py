import os
import pandas as pd
from PIL import Image
import torch
from setup import *
from utils import *
from tqdm import tqdm
tqdm.pandas()
import json
import model_utils
import argparse
    
def main(args):
    print(f"Started fwd patch") 
    seedEverything()
 
    model_name = args.model_name
    target_layer = args.target
    
    mp = setup(model_name)
    wrapped = model_utils.WrappedMP(mp)
        
    df = pd.read_csv(args.dataset)    
    subset = pd.read_csv(args.subjects).drop_duplicates('s_uri')
    df = df[df['s_uri'].isin(subset['s_uri'])]

    save_dir = os.path.join('results', model_name, f'fwd_patch_{target_layer}')
    os.makedirs(save_dir, exist_ok=True)

    res_file = os.path.join(save_dir, f'fwd_patching_{target_layer}.jsonl')
    if args.block:
        res_file = add_suffix_to_filepath(res_file, f"block")
        
    wrapped.buffer = 1
    for i, row in tqdm(df.iterrows(), total=len(df)):
        safe_subj = row['subject'].replace(' ', '_')
        image_file = get_full_path(row["resized_path"])
        query, prefix = get_identify_query_and_prefix_by_type(row['type'])
        prompt = mp.get_prompt(query=query, image=True, prefix=prefix)
        raw_image = Image.open(image_file)
        inputs = mp.processor(images=raw_image, text=prompt, return_tensors='pt').to(wrapped.device, torch.float16)
        res = []
        null_examples = []

        for l in range(0, target_layer):
            if args.block:
                for injection in range(l+1, target_layer+1):
                    wrapped.inject_block(injection)
            else:
                wrapped.inject_block(target_layer)
                raise NameError
            wrapped.wrap_block(l)
            
            output = wrapped.mp.model.generate(**inputs, max_new_tokens=30, do_sample=False, output_hidden_states=False, return_dict_in_generate=True, use_cache=False)
            generated_text = wrapped.mp.processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
            correct = identified(generated_text, row['subject'], row['aliases'])
            if not correct:
                null_examples.append(generated_text)
            else:
                null_examples.append('correct')
            res.append(correct)
            wrapped.reset_layers()
        res = {'subject':safe_subj, 'res':res, 'examples':null_examples}
        with open(res_file, 'a') as f:
            json_record = json.dumps(res)
            f.write(json_record + '\n')

        
        
def get_exp_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model-name', type=str, default="llava_7B")
    parser.add_argument('--subjects', type=str, help="results/llava_7B/llava_7B_entities_id.csv")
    parser.add_argument('--dataset', type=str, help="data/entities.csv")
    parser.add_argument('--target', type=int, default=20)
    parser.add_argument('--block', action='store_false', default=True)
    return parser


if __name__ == "__main__":
    parser = get_exp_parser()
    args = parser.parse_args()
    main(args)
    
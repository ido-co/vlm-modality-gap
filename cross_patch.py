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
    print(f"Started cross patch")
    seedEverything()
   
    model_name = args.model_name
    
    mp = setup(model_name)
    wrapped = model_utils.WrappedMP(mp)
    
    s_uris = {'celebs': ['Q22686', 'Q16397'],
            'landmarks': ['Q243', 'Q9141'],
            'paintings': ['Q12418', 'Q157541'],
            'brands': ['Q689141', 'Q246']
        }
     
    all_s_uris = []
    for value in s_uris.values():
        all_s_uris.extend(value)
    
    df = pd.read_csv(args.dataset)
    injected_subjects = df[df['s_uri'].isin(all_s_uris)]
    assert len(injected_subjects) == 8
    injected_subjects['safe'] = injected_subjects['subject'].apply(lambda x: x.replace(' ', '_'))

    injected_subjects_per_type = {}

    for type in injected_subjects['type'].unique():
        to_include = []
        for other_type, entities in s_uris.items():
            if other_type == type:
                to_include.extend(entities)
            else:
                to_include.append(entities[0])
        injected_subjects_per_type[type] = injected_subjects[injected_subjects['s_uri'].isin(to_include)]
    
    subset = pd.read_csv(args.subjects).drop_duplicates('s_uri')
    for s_uri in all_s_uris:
        assert s_uri in subset['s_uri'].values, f"Missing {s_uri} in {args.subjects}"
    df = df[df['s_uri'].isin(subset['s_uri'])]
    
    res_prefix =  f"{model_name}_{{subject}}"
    save_dir = os.path.join('results', model_name, 'cross_patch')
    os.makedirs(save_dir, exist_ok=True)
    res_prefix = os.path.join(save_dir, res_prefix)
    
    if args.extract:
        print('Starting extraction')
        
        for l in range(len(wrapped.lm.layers)):
            wrapped.wrap_block(l)

        for i, row in tqdm(injected_subjects.iterrows(), total=len(injected_subjects)):
            safe_subj = row['subject'].replace(' ', '_')
            save_dst = res_prefix.format(subject=safe_subj)
            if os.path.exists(save_dst + "_layer_0.pt"):
                continue
            wrapped.save_name = save_dst
            query, prefix = get_identify_query_and_prefix_by_type(row['type'])
            prompt = mp.get_prompt(query=query, image=True, prefix=prefix) 
            image_file = get_full_path(row["resized_path"])
            raw_image = Image.open(image_file)
            inputs = mp.processor(images=raw_image, text=prompt, return_tensors='pt').to(wrapped.device, torch.float16)
            output = wrapped.mp.model.generate(**inputs, max_new_tokens=5, do_sample=False, output_hidden_states=False, return_dict_in_generate=True, use_cache=False)

        wrapped.reset_layers()
        wrapped.save_name = None
    
    res_file = os.path.join(save_dir, "inject_tokens_results.jsonl")
    
    print('Starting patching')        
    for i, row in tqdm(df.iterrows(), total=len(df)):
        for j, injected_row in injected_subjects_per_type[row['type']].iterrows():
            orig_results = []
            new_results = []

            safe_subj = row['subject'].replace(' ', '_')
            image_file = get_full_path(row["resized_path"])
            raw_image = Image.open(image_file)
            if row['type'] == injected_row['type']:
                query, prefix = get_identify_query_and_prefix_by_type(row['type'])
            else:
                query = "Identify by name the subject in the image."
                prefix = "The subject in the image is"

            prompt = mp.get_prompt(query=query, image=True, prefix=prefix)
            
            inputs = mp.processor(images=raw_image, text=prompt, return_tensors='pt').to(wrapped.device, torch.float16)

            for l in range(5, 30):
                source_layer = l - 1
                wrapped.buffer = torch.load(res_prefix.format(subject=injected_row['safe']) + f"_layer_{source_layer}.pt", map_location=torch.device("cuda"), weights_only=True)
                wrapped.inject_block(l)
                output = mp.model.generate(**inputs, max_new_tokens=30, do_sample=False, output_hidden_states=False, return_dict_in_generate=True, use_cache=False)
                generated_text = mp.processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
                orig = identified(generated_text, row['subject'], row['aliases'])
                new = identified(generated_text, injected_row['subject'], injected_row['aliases'])
                orig_results.append(orig)
                new_results.append(new)
                wrapped.reset_layers()
                
            res = {'orig_subject':row['subject'],
                   'injected_subject':injected_row['subject'],
                   'orig_type':row['type'],
                   'injected_type':injected_row['type'],
                   'orig':orig_results,
                   'new':new_results}
            
            with open(res_file, 'a') as f:
                json_record = json.dumps(res)
                f.write(json_record + '\n')
    


def get_exp_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model-name', type=str, default="llava_7B")
    parser.add_argument('--subjects', type=str, default="results/llava_7B/llava_7B_entities_id.csv")
    parser.add_argument('--entities', type=str, default="data/entities.csv")
    parser.add_argument('--extract', action='store_false', default=True)
    return parser


if __name__ == "__main__":
    parser = get_exp_parser()
    args = parser.parse_args()
    main(args)
    
    
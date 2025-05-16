from setup_more import *
import os
import pandas as pd
import torch
from utils import *
from tqdm import tqdm
tqdm.pandas()
import json
import model_utils_more
import argparse

def main(args):
    print(f"Started fwd patch") 
    seedEverything()
    
    model_path = MODEL_NAME_PATH[args.model_name]
    model_name = args.model_name
    target_layer = args.target
    
    tokenizer, model, prepare_inputs = initialize_model(model_path)
    wrapped = model_utils_more.WrappedMP(model, model_name)
    
    df = args.dataset
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
        query, _ = get_identify_query_and_prefix_by_type(row['type'])
        prompt = PROMPT_IM_TEMP.format(query=query)
        input_ids, images_tensor, image_sizes = prepare_inputs(get_full_path(row['resized_path']), prompt)
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
            print(generated_text)
            correct = identified(generated_text, row['subject'], row['aliases'])
            if not correct:
                null_examples.append(generated_text)
            else:
                null_examples.append('correct')
            res.append(correct)
            wrapped.reset_layers()
            print("\n\nAMAZING!!!!\n\n")
            exit()
        res = {'subject':safe_subj, 'res':res, 'examples':null_examples}
        with open(res_file, 'a') as f:
            json_record = json.dumps(res)
            f.write(json_record + '\n')

        
        
def get_exp_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model-name', type=str, default="clip")
    parser.add_argument('--subjects', type=str, default="results/clip/clip_entities_id.csv")
    parser.add_argument('--dataset', type=str, default="data/entities.csv")
    parser.add_argument('--target', type=int, default=20)
    parser.add_argument('--block', action='store_false', default=True)

    return parser


if __name__ == "__main__":
    parser = get_exp_parser()
    args = parser.parse_args()
    main(args)
    
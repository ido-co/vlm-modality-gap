from setup_more import *
import argparse
import os
import pandas as pd
import torch
from utils import *
from tqdm import tqdm
tqdm.pandas()
import sys

def main(args):    
    print(f"Started") 
    seedEverything()
    
    model_path = MODEL_NAME_PATH[args.model_name]
    model_name = args.model_name
    from_image = args.from_image
    from_image_str = "image" if args.from_image else "text"

    tokenizer, model, prepare_inputs = initialize_model(model_path)
    
    df = pd.read_csv(args.subjects)

    df = df[df['identified']][['type', 's_uri', 'subject', 'path']]
    questions = pd.read_csv(args.questions)
    df = df.merge(questions, on=["s_uri", "subject"])

    results = []

    for i, row in tqdm(df.iterrows(), total=len(df)):       
        if from_image:
            q = 'question_for_image'
            prompt = PROMPT_IM_TEMP.format(query=row[q])
            input_ids, images_tensor, image_sizes = prepare_inputs(get_full_path(row['path']), prompt)
        else:
            q = 'question'
            prompt = PROMPT_NO_IM_TEMP.format(query=row[q])
            input_ids, _, _ = prepare_inputs(get_full_path(row['path']), prompt)

        with torch.inference_mode():
            if from_image:
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    max_new_tokens=100,
                    use_cache=True,
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=100,
                    use_cache=True,
                )
            
        generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        correct = identified(generated_text, None, row['possible_answers'])
        results.append({'s_uri': row['s_uri'], 'subject': row['subject'], 
                        'path': row['path'], 'question': row[q], 'generation': generated_text, 'correct': correct})
        
        if (i+1) % 200 == 0:
            save_results(results, model_name, args.subjects, f"{from_image_str}_{i}", dir=f'tmp_{model_name}')
    
    save_results(results, model_name, args.subjects, suffix=from_image_str)

def save_results(results, model_name, csv_path, suffix=None, dir='results'):
    res_df = pd.DataFrame.from_records(results)
    _, filename = os.path.split(csv_path)
    filename, _ = os.path.splitext(filename)
    res_file = f"{filename}.csv" if not suffix else f"{filename}_{suffix}.csv"
    os.makedirs(os.path.join(dir, model_name), exist_ok=True)
    res_df.to_csv(os.path.join(dir, model_name, res_file), index=False)
    print(f"Saved to {os.path.join(dir, model_name, res_file)}")

def get_exp_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-name", type=str, default="clip")
    parser.add_argument('--subjects', type=str, default="results/clip/clip_enitites_id.csv")
    parser.add_argument('--questions', type=str, default="data/questions.csv")
    parser.add_argument('--from-image', action='store_true', default=False)

    return parser


if __name__ == "__main__":
    parser = get_exp_parser()
    args = parser.parse_args()
    main(args)

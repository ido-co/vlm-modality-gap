import argparse
import os
import pandas as pd
from PIL import Image
import torch
from setup import *
from utils import *
from tqdm import tqdm
tqdm.pandas()

def process_batch(batch, results, mp, from_image):
    images, prompts, rows = zip(*batch)
    with torch.no_grad():
        if from_image:
            inputs = mp.processor(images=list(images), text=list(prompts), return_tensors='pt', padding=True)
            inputs = inputs.to(mp.model.device, torch.float16)
        else:
            inputs = mp.processor.tokenizer(text=list(prompts), return_tensors='pt', padding=True)
            inputs["input_ids"] = inputs["input_ids"].cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
        output = mp.model.generate(**inputs, max_new_tokens=100, do_sample=False, 
                                output_hidden_states=False, return_dict_in_generate=True, use_cache=True)
    
        generated_texts = mp.processor.batch_decode(output.sequences, skip_special_tokens=True)
    
    q = 'question_for_image' if from_image else 'question'
    for row, generated_text in zip(rows, generated_texts):
        generated_text = generated_text.split("ASSISTANT:")[-1]
        correct = identified(generated_text, None, row['possible_answers'])
        results.append({'s_uri': row['s_uri'], 'subject': row['subject'], 
                        'path': row['path'], 'question': row[q], 'generation': generated_text, 'correct': correct})


def save_results(results, model_name, csv_path, suffix=None, dir='results'):
    res_df = pd.DataFrame.from_records(results)
    _, filename = os.path.split(csv_path)
    filename, _ = os.path.splitext(filename)
    res_file = f"{filename}.csv" if not suffix else f"{filename}_{suffix}.csv"
    os.makedirs(os.path.join(dir, model_name), exist_ok=True)
    res_df.to_csv(os.path.join(dir, model_name, res_file), index=False)
    print(f"Saved to {os.path.join(dir, model_name, res_file)}")


def main(args):
    batch_size = args.batch_size
    model_name = args.model_name
    from_image = args.from_image
    from_image_str = "image" if args.from_image else "text"

    mp = setup(model_name)

    df = pd.read_csv(args.subjects).drop_duplicates('s_uri')
    df = df[df['identified']][['type', 's_uri', 'subject', 'path']]
    questions = pd.read_csv(args.questions)
    df = df.merge(questions, on=["s_uri", "subject"])
    
    results = []
    batch = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):       
        full_path = get_full_path(row['path'])
        if from_image:
            raw_image = Image.open(full_path)
            prompt = mp.get_prompt(query=row['question_for_image'], image=True)
        else:
            raw_image = None
            prompt = mp.get_prompt(query=row['question'], image=False)
        batch.append((raw_image, prompt, row))
        
        if len(batch) >= batch_size:
            process_batch(batch, results, mp, from_image)
            batch = []
        
        if i % 200 == 0:
            save_results(results, model_name, args.subjects, f"{from_image_str}_{i}", dir='tmp_qa')
    
    if batch:
        process_batch(batch, results, mp, from_image)
    
    save_results(results, model_name, args.subjects, suffix=from_image_str)
    


def get_exp_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model-name', type=str, default="llava_7B")
    parser.add_argument('--subjects', type=str, default="results/llava_7B/llava_7B_entities_id.csv")
    parser.add_argument('--questions', type=str, default="data/questions.csv")
    parser.add_argument('--from-image', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=4, help='Number of images to process in a batch')

    return parser


if __name__ == "__main__":
    parser = get_exp_parser()
    args = parser.parse_args()
    main(args)
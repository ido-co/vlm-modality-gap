import argparse
import subprocess
import sys
from dotenv import load_dotenv
load_dotenv()

def get_unified_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default="clip")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=4)
    parser.add_argument('--env', type=str, default='python')
    return parser

if __name__ == "__main__":
    parser = get_unified_parser()
    args = parser.parse_args()

    model_name = args.model_name
    env = args.env
    suffix = '_more' if model_name == 'clip' or model_name == 'siglip' else ''

    print(f"Running scripts for model: {model_name} with environment: {env}")

    if args.start <= 0 and args.end >= 0:
        print("Running identification script...")
        result1 = subprocess.run([
            env, f"identify{suffix}.py",
            "--model-name", model_name
        ])

        if result1.returncode != 0:
            print("Identification script failed. Aborting.")
            sys.exit(result1.returncode)

    results_file = f"results/{model_name}/{model_name}_id.csv"

    if args.start <= 1 and args.end >= 1:
        print("Running generation with image script...")
        result2 = subprocess.run([
            env, "generation_qa{suffix}.py",
            "--model-name", model_name,
            "--subjects", results_file,
            "--from-image"
        ])

        if result2.returncode != 0:
            print("Generation QA with image failed. Aborting.")
            sys.exit(result2.returncode)

    if args.start <= 2 and args.end >= 2:
        print("Running generation with text script...")
        result3 = subprocess.run([
            env, "generation_qa{suffix}.py",
            "--model-name", model_name,
            "--subjects", results_file
        ])
        
        if result3.returncode != 0:
            print("Generation QA with text failed. Aborting.")
            sys.exit(result3.returncode)
    
    if args.start <= 3 and args.end >= 3:
        print("Running cross patch script...")
        result4 = subprocess.run([
            env, "cross_patch{suffix}.py",
            "--model-name", model_name,
            "--subjects", results_file,
        ])
        
        if result4.returncode != 0:
            print("Cross patch failed. Aborting.")
            sys.exit(result4.returncode)
            
    if args.start <= 4 and args.end >= 4:
        print("Running forward patch script...")
        result5 = subprocess.run([
            env, "fwd_patch{suffix}.py",
            "--model-name", model_name,
            "--subjects", results_file,
            "--target", "20"
            ])
        
        if result5.returncode != 0:
            print("Forward patch failed. Aborting.")
            sys.exit(result5.returncode)
        
    print("All scripts ran successfully.")

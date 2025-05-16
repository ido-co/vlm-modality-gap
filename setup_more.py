import os
import torch
import sys
from dotenv import load_dotenv
load_dotenv()

llava_more_path = os.path.abspath("LLaVA-MORE")
if llava_more_path not in sys.path:
    sys.path.insert(0, llava_more_path)

from llava.constants import IMAGE_TOKEN_INDEX

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from PIL import Image

MODEL_NAME_PATH = {
  "clip":"aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning",
  "siglip":"aimagelab/LLaVA_MORE-llama_3_1-8B-siglip-finetuning"
}

PROMPT_IM_TEMP = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>

<image>
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

PROMPT_NO_IM_TEMP = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

def load_image_more(image_file):
    return Image.open(image_file).convert("RGB")

def initialize_model(model_path):
    disable_torch_init()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, 
        model_base=None, 
        model_name='llava',
        load_4bit=True
    )

    def prepare_inputs(image_path, prompt):
        if image_path is not None:
            images = [load_image_more(image_path)]
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
        else:
            images_tensor = None
            image_sizes = None
        
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        return input_ids, images_tensor, image_sizes
    
    return tokenizer, model, prepare_inputs

if __name__ == "__main__":
    model_path = MODEL_NAME_PATH["clip"]
    tokenizer, model, prepare_inputs = initialize_model(model_path)
    
    # Example usage
    image_path = "path_to_image.jpg"
    prompt = "What is in the image?"
    input_ids, images_tensor, image_sizes = prepare_inputs(image_path, prompt)
    
    print("Input IDs:", input_ids)
    print("Image Tensor Shape:", images_tensor.shape)
    print("Image Sizes:", image_sizes)
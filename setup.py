from dotenv import load_dotenv
load_dotenv()

import torch
import transformers
from transformers import AutoProcessor, BitsAndBytesConfig

MODEL_NAME_PATH = {
  "vicuna_7B": "lmsys/vicuna-7b-v1.5",
  "vicuna_13B": "lmsys/vicuna-13b-v1.5",
  "llava_7B": "llava-hf/llava-1.5-7b-hf",
  "llava_13B": "llava-hf/llava-1.5-13b-hf",
  "llava_34B": "llava-hf/llava-v1.6-34b-hf",
  "clip":"aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning",
  "siglip":"aimagelab/LLaVA_MORE-llama_3_1-8B-siglip-finetuning",
  "qwen":"Qwen/Qwen2-VL-7B-Instruct"
}

def is_flash_attention_supported():
    if not torch.cuda.is_available():
        return False
    
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 0)

def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)

class VicunaModelAndTokenizer:
    def __init__(self, model_name):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.tokenizer = tokenizer
        self.model = model
        self.model.cuda()
        self.num_layers = model.config.num_hidden_layers
        self.vocabulary_projection_function = lambda x, layer: self.model.lm_head(self.model.model.norm(x)) if layer < self.num_layers else self.model.lm_head(x) 

    def __repr__(self):
        """String representation of this class.
        """
        return (
            f"VicunaModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
            )

class LLaVAModelAndProcessor:
    def __init__(self, model_name, accelerator=None, device_id=None):
        
        if '34' in model_name:
            if accelerator is not None:
                device_map = {"": accelerator.process_index}
            else:
                device_map = 'auto'

            from transformers import AutoModelForImageTextToText
            quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            # if is_flash_attention_supported():
            #     from flash_attn import flash_attn_func
            #     self.model = AutoModelForImageTextToText.from_pretrained(
            #         model_name, 
            #         quantization_config=quantization_config, 
            #         device_map="auto",
            #         use_flash_attention_2=True)
            #     print('using flash attention')
            # else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, 
                quantization_config=quantization_config, 
                device_map="auto")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.prompt_template_image = '<|im_start|>system\nYou are a helpful assistant. <|im_end|><|im_start|>user\n<image>\n{query}<|im_end|><|im_start|>assistant\n{prefix}'
            self.prompt_template_text = '<|im_start|>system\nYou are a helpful assistant. <|im_end|><|im_start|>user\n{query}<|im_end|><|im_start|>assistant\n{prefix}'

        else:
            from transformers import LlavaForConditionalGeneration
            if device_id:
                device = f"cuda:{device_id}"
                device_map = {"": device}
            else:
                device_map = 'auto'
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
                )               
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, 
                quantization_config=quantization_config,
                device_map=device_map
                # attn_implementation="flash_attention_2",
            )
            self.processor = AutoProcessor.from_pretrained(model_name, revision='a272c74')
            self.prompt_template_image = 'USER: <image>\n{query}\nASSISTANT:'
            self.prompt_template_text = 'USER: \n{query}\nASSISTANT:'
            
        self.num_layers = len(self.model.language_model.model.layers)
        self.vocabulary_projection_function = lambda x, layer: self.model.language_model.lm_head(self.model.language_model.model.norm(x)) if layer < self.num_layers else self.model.language_model.lm_head(x) 

    def get_prompt(self, query, image=True, prefix=""):
        if image:
            prompt = self.prompt_template_image.format(query=query, prefix=prefix)
        else:
            prompt = self.prompt_template_text.format(query=query, prefix=prefix)
        return prompt

    def __repr__(self):
        """String representation of this class.
        """
        return (
            f"LLaVAModelAndProcessor(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"processor: {type(self.processor).__name__})"
            )

class Qwen:
    def __init__(self, model_name):
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.prompt_template_image = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{query}<|im_end|>\n<|im_start|>assistant\n{prefix}'
        self.prompt_template_text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query} <|im_end|>\n<|im_start|>assistant\n{prefix}'
    
    def get_prompt(self, query, image=True, prefix=""):
        if image:
            prompt = self.prompt_template_image.format(query=query, prefix=prefix)
        else:
            prompt = self.prompt_template_text.format(query=query, prefix=prefix)
        return prompt



def setup(model_name, requires_grad=False, accelerator=None, device_id=None):
    model_name = MODEL_NAME_PATH[model_name]
    if "vicuna" in model_name:
        mt = VicunaModelAndTokenizer(model_name)
    elif "Qwen" in model_name:
        mt = Qwen(model_name)
    else:
        mt = LLaVAModelAndProcessor(model_name, accelerator=accelerator, device_id=device_id)
    mt.model.eval()
    set_requires_grad(requires_grad, mt.model)
    return mt
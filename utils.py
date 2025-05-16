import os
from PIL import Image
import torch
import ast
import random
import numpy as np
import re
from typing import Union, List
import unicodedata
import math

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.environ['DATA_DIR']
DEFAULT_RANDOM_SEED = 2024

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
       
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

def get_identify_query_and_prefix_by_type(type):
    if type == 'paintings':
        return "Identify by name the painting in the image.", "The name of the painting is "
    if type == 'brands':
        return "Identify by name the brand who's logo appears in the image.", "The name of the brand is "
    if type == 'landmarks':
        return "Identify by name the place in the image.", "The name of the place is "
    if type == 'celebs':
        return "Identify by name the subject in the image.", "The subject in this image is "
    raise ValueError(type)
   
def get_full_path(path):
    return os.path.join(DATA_DIR, path)
      
def apply_chat_template(query, mp):
    conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": query},
          {"type": "image"},
        ],
    },
    ]
    return mp.processor.apply_chat_template(conversation, add_generation_prompt=True)

def add_suffix_to_filepath(path: str, suffix: str) -> str:
    dirname, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{suffix}{ext}"
    return os.path.join(dirname, new_filename)

################
# Output utils #
################

def normalize_text_to_list(text: str, min_word_length: int = 3) -> List[str]:
    """
    Normalizes text by handling unicode, converting to lowercase, and splitting into a list of words.
    Filters out words shorter than the specified minimum length.
    
    Args:
        text (str): Input text to normalize
        min_word_length (int): Minimum length of words to keep (default: 3)
    
    Returns:
        List[str]: List of normalized words
    """
    if not text:
        return []
        
    # Unicode normalization (NFKC handles compatibility equivalents)
    text = unicodedata.normalize('NFKC', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and split into words
    words = re.findall(r'\b\w+\b', text)
    
    # Filter out short words and return
    return [word for word in words if len(word) > min_word_length]

def check_lists_overlap(gen_list: Union[str, List[str]],
                       ans_list: Union[str, List[str]],
                       min_overlap: int = 2) -> bool:
    
    set1 = set(ans_list)
    set2 = set(gen_list)
    
    min_overlap = min_overlap if len(set1) > 1 else 1
    
    # Get intersection and check its length
    overlap = set1.intersection(set2)
    return len(overlap) >= min_overlap

def check_ans(generated_text, possible_answers):
    gen_norm = normalize_text_to_list(generated_text)
    for ans in possible_answers:
        ans_norm = normalize_text_to_list(ans)
        if check_lists_overlap(gen_norm, ans_norm):
            return True
    return False

def identified(generated_text, subject, aliases=None):
    if subject and aliases:
        possible_answers = [subject] + ast.literal_eval(aliases)
    elif subject:
        possible_answers = [subject]
    elif aliases:
        possible_answers = ast.literal_eval(aliases)
    return check_ans(generated_text, possible_answers)

#########
# Other #
#########

def add_suffix_at(path, suffix, subdir):
    lst = path.split('/')
    lst[-3] = lst[subdir] + '_' + suffix
    return '/'.join(lst)

def resize_crop(image, size):
    """
    Crop the image with a centered rectangle of the specified size
    image:      a Pillow image instance
    size:       a list of two integers [width, height]
    """
    img_format = image.format
    image = image.copy()
    old_size = image.size
    left = (old_size[0] - size[0]) / 2
    top = (old_size[1] - size[1]) / 2
    right = old_size[0] - left
    bottom = old_size[1] - top
    rect = [int(math.ceil(x)) for x in (left, top, right, bottom)]
    left, top, right, bottom = rect
    crop = image.crop((left, top, right, bottom))
    crop.format = img_format
    return crop

def resize_cover(image, size=[336, 336], resample=Image.LANCZOS):
    """
    Resize image according to size.
    image:      a Pillow image instance
    size:       a list of two integers [width, height]
    """
    img_format = image.format
    img = image.copy()
    img_size = img.size
    ratio = max(size[0] / img_size[0], size[1] / img_size[1])
    new_size = [
        int(math.ceil(img_size[0] * ratio)),
        int(math.ceil(img_size[1] * ratio))
    ]
    img = img.resize((new_size[0], new_size[1]), resample)
    img = resize_crop(img, size)
    img.format = img_format
    return img

def resize_square(image, size=336, resample=Image.LANCZOS):
    """
    Resize an image to a square of the given size, first adding a black background if needed.
    image:  a Pillow image instance
    size:   an integer, the desired output size (width and height will be the same)
    """
    img_format = image.format
    image = image.copy()
    
    size = [size, size]
    img_size = image.size
    ratio = min(size[0] / img_size[0], size[1] / img_size[1])
    new_size = [
        int(math.ceil(img_size[0] * ratio)),
        int(math.ceil(img_size[1] * ratio))
    ]
    
    image = image.resize((new_size[0], new_size[1]), resample)
    
    # Make the image square by adding black padding
    max_dim = max(image.size)
    new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    new_img.paste(image, ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2))
    
    # Resize to target size
    # new_img = new_img.resize((size, size), resample)
    new_img.format = img_format
    return new_img

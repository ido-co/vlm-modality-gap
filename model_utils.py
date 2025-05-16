import torch
import functools
from tqdm import tqdm
tqdm.pandas()


LAST_IMG_TOK = 580
FIRST_IMG_TOK = 5
LAST_IMG_TOK_34 = 1178
FIRST_IMG_TOK_34 = 3


class WrappedMP():
    def __init__(self, mp):
        self.mp = mp
        self.device = mp.model.device
        self.lm = mp.model.language_model.model
        self.attention_modules = self.get_attn_modules(0, len(self.lm.layers))
        self.layer_modules = self.get_layers()
        self.attn_forward_funcs = [layer_module[0].forward for layer_module in self.layer_modules]
        self.mlp_forward_funcs = [layer_module[1].forward for layer_module in self.layer_modules]
        self.blocks = self.get_blocks()
        self.block_forwards = [block.forward for block in self.blocks]
        self.original_forward = self.lm.forward
        self.buffer = None
        self.save_name = None
        self.first_img_tok = FIRST_IMG_TOK if len(self.layer_modules) == 32 else FIRST_IMG_TOK_34
        self.last_img_tok = LAST_IMG_TOK if len(self.layer_modules) == 32 else LAST_IMG_TOK_34

    def get_attn_modules(self, start, end):
        assert 0 <= start < len(self.lm.layers)
        assert 0 <= end <= len(self.lm.layers)
        return [self.lm.layers[l].self_attn for l in range(start, end)]

    def get_layers(self):
        return [(self.lm.layers[l].self_attn, self.lm.layers[l].mlp) for l in range(len(self.lm.layers))]

    def get_blocks(self):
        return [self.lm.layers[l] for l in range(len(self.lm.layers))]
    
    def save_model_output(self, original_forward_func, l):
        @functools.wraps(original_forward_func)
        def save_layer_output(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for key, v in kwargs.items():
                new_kwargs[key] = v
            
            new_kwargs['use_cache'] = False
            output = original_forward_func(*args, **kwargs)
            if self.buffer is not None:
                self.buffer = output[0]
            else:
                torch.save(output[0], f"{self.save_name}_layer_{l}.pt")
            return output
        return save_layer_output


    def inject_img_tokens(self, original_forward_func, all=False, last=False):
        @functools.wraps(original_forward_func)
        def injected_forward(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for key, v in kwargs.items():
                new_kwargs[key] = v
            
            if self.buffer is None: 
                # tokens = torch.load(f"{subject}_layer_{l}.pt", map_location=torch.device("cuda"), weights_only=True)
                raise NotImplementedError("buffer must not be None")
            else:
                tokens = self.buffer
            if all:
                new_args[0][:, :, :] = tokens[:, :, :]
            if last:
                new_args[0][:, -1, :] = tokens[:, -1, :]
            else:
                new_args[0][:, self.first_img_tok:self.last_img_tok, :] = tokens[:, self.first_img_tok:self.last_img_tok, :]
            return original_forward_func(*args, **kwargs)
        return injected_forward

    

    def wrap_mlp_forward(self, original_forward_func):
        @functools.wraps(original_forward_func)
        def knockout_mlp_forward(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for key, v in kwargs.items():
                new_kwargs[key] = v

            x = args[0]

            output = original_forward_func(*new_args, **new_kwargs)
            
            output[:, :LAST_IMG_TOK, :] = x[:, :LAST_IMG_TOK, :]
            
            return output
        return knockout_mlp_forward    
    
    def wrap_attention_forward(self, original_forward_func):
        @functools.wraps(original_forward_func)
        def knockout_attention_forward(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for key, v in kwargs.items():
                new_kwargs[key] = v

            new_kwargs['use_cache'] = False

            if "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            else:
                hidden_states = args[0]
            batch_size = hidden_states.shape[0]
            num_tokens = hidden_states.shape[1]
            attention_weight_size = (batch_size, 1, num_tokens, num_tokens)
            prev_attention_mask = kwargs["attention_mask"]
            if prev_attention_mask is not None:
                new_attention_mask = torch.zeros(attention_weight_size, dtype=hidden_states.dtype).to(
                    hidden_states.device)
                new_attention_mask += prev_attention_mask
            else:
                new_attention_mask = torch.tril(torch.ones(attention_weight_size, dtype=hidden_states.dtype)).to(
                    hidden_states.device)
                new_attention_mask[:, :, self.last_img_tok:, self.first_img_tok:self.last_img_tok] = 0 
                new_attention_mask = (1.0 - new_attention_mask) * torch.finfo(self.lm.dtype).min
            new_kwargs["attention_mask"] = new_attention_mask
            
            output = original_forward_func(*new_args, **new_kwargs)
            return output
            
        return knockout_attention_forward
        
    def wrap_attn(self, l):
        self.attention_modules[l].forward = self.wrap_attention_forward(self.attention_modules[l].forward)
    
    def wrap_block(self, l):
        self.blocks[l].forward = self.save_model_output(self.block_forwards[l], l)

    def inject_block(self, l, all=False, last=False):
        self.blocks[l].forward = self.inject_img_tokens(self.block_forwards[l], all=all, last=last)

    def wrap_layer(self, l):
        self.layer_modules[l][0].forward = self.wrap_attention_forward(self.layer_modules[l][0].forward)
        self.layer_modules[l][1].forward = self.wrap_mlp_forward(self.layer_modules[l][1].forward)

    def reset_layers(self):
        for l in range(len(self.lm.layers)):
            self.layer_modules[l][0].forward = self.attn_forward_funcs[l]
            self.layer_modules[l][1].forward = self.mlp_forward_funcs[l]
            self.blocks[l].forward = self.block_forwards[l]
            self.lm.forward = self.original_forward

    def double_pass(self, l):
        @functools.wraps(self.original_forward)
        def double_pass_forward(*args, **kwargs):
            self.wrap_block(l, subject=None)
            self.original_forward(*args, **kwargs)
            self.reset_layers()
            self.inject_block(1, subject=None)
            return self.original_forward(*args, **kwargs)
        self.lm.forward = double_pass_forward
        
        
    
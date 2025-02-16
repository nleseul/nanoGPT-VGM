"""
Sample from a trained VGM model
"""
import io
import gzip
import os
from contextlib import nullcontext
import pyvgm
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-vgm' # ignored if init_from is not 'resume'
output_name = 'gen'
start_file = ""
start_ticks = 10000
num_samples = 10 # number of samples to draw
tokens_per_pass = 500 # number of tokens generated at a time
max_new_tokens = 50000 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Load vocabulary
vocab_path = os.path.join('data', checkpoint['config']['dataset'], 'vocabulary_list.txt')
vocab_list = open(vocab_path, 'r').read().splitlines()

vocab_map = {}
for index, token in enumerate(vocab_list):
    vocab_map[token] = index

end_id = vocab_map["*END*"]
terminator = (torch.tensor([end_id], dtype=torch.long, device=device)[None, ...])

# encode the beginning of the prompt
start_ids = [end_id]
if start_file is not None and len(start_file) > 0:
    start_ids = []
    if start_file.endswith(".vgz"):
        in_file = gzip.open(start_file, 'rb')
    else:
        in_file = open(start_file, 'rb')
    
    start_vgm = pyvgm.VGMFile.load(in_file)
    
    current_ticks = 0
    for command in start_vgm.commands:
        if isinstance(command, pyvgm.WaitCommand):
            current_ticks += command.ticks
        command_data = command.data.hex()
        start_ids.append(vocab_map[command_data])
        if current_ticks >= start_ticks:
            break
    in_file.close()
                
    print(f"Imported {len(start_ids)} tokens from {start_file}")

x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, terminator_idx=terminator)
            
            vgm = pyvgm.VGMFile()
            token_count = 0
            tick_count = 0
            for token_index in y[0].tolist():
                if token_index == end_id:
                    if token_count == 0:
                        continue
                    else:
                        done = True
                        break
                else:
                    token = vocab_list[token_index]
                    command = pyvgm.VGMFile.command_from_bytes(io.BytesIO(bytes.fromhex(token)))
                    if command is None:
                        break;
                    else:
                        if isinstance(command, pyvgm.WaitCommand):
                            tick_count += command.ticks
                        vgm.add_command(command)
                        token_count += 1
                    
            filename = f"{output_name}{k}.vgm"
            with open(filename, 'w+b') as out_file:
                vgm.save(out_file)
            print(f"{filename} - {token_count} tokens, {tick_count} ticks")

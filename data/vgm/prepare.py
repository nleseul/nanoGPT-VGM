import fnmatch
import gzip
import math
import os
import pickle
import pyvgm
import random

vgm_paths = [
    #"../VGM_big",
    "../zophar_scrape/Zophar.net VGM cleaned"
]

def swap_channels_processor(commands):
    for command in vgm.commands:
        if command is pyvgm.WriteRegisterCommand and command.chip == pyvgm.Chip.NESAPU:
            if command.register >= 0x0 and command.register <= 0x3:
                yield pyvgm.WriteRegisterCommand(command.chip, command.register + 0x4, command.value)
            elif command.register >= 0x4 and command.register <= 0x7:
                yield pyvgm.WriteRegisterCommand(command.chip, command.register - 0x4, command.value)
            else:
                yield command
        else:
            yield command

'''
if not os.path.exists("data/vgm/vgmrips.zip"):
    print("Downloading file...")
    urllib.request.urlretrieve("https://chiparchive.com/files/VGMRips_all_of_them_2023-04-14.zip", "data/vgm/vgmrips.zip")
    
os.makedirs("NES", exist_ok=True)
    
zip = ZipFile("data/vgm/vgmrips.zip", 'r')
#nes_dir = Path(zip) / "NES"
#for f in nes_dir.iterdir():
#    print(f.name)
zip.extract("NES")
'''

if os.path.exists("data/vgm/vocabulary_list.txt"):
    print("Loading existing vocabulary...")
    vocabulary = open("data/vgm/vocabulary_list.txt", 'r').read().splitlines()
else:
    print("Importing vocabulary...")

    vocabulary_set = set()
    token_count = 0

    vocabulary_set.add("*END*")

    for vgm_path in vgm_paths:
        for path, dirs, files in os.walk(vgm_path):
            for filename in fnmatch.filter(files, "*.vg?"):
                full_path = os.path.join(path, filename)
                
                if filename.endswith("z"):
                    in_file = gzip.open(full_path, 'rb')
                else:
                    in_file = open(full_path, 'rb')
                
                vgm = pyvgm.VGMFile.load(in_file)

                if vgm.total_duration < 22050: # Half a second
                    print(f"SFX: {full_path}")
                    continue
                else:
                    print(full_path)

                for command in vgm.commands:
                    if not isinstance(command, pyvgm.PCMDataCommand) and not isinstance(command, pyvgm.PCMRamWriteCommand):
                        vocabulary_set.add(command.data.hex())
                        token_count += 1
                for command in swap_channels_processor(vgm.commands):
                    if not isinstance(command, pyvgm.PCMDataCommand) and not isinstance(command, pyvgm.PCMRamWriteCommand):
                        vocabulary_set.add(command.data.hex())
                        token_count += 1
                        
                in_file.close()
                    
    print(len(vocabulary_set), token_count)

    vocabulary = sorted(list(vocabulary_set))
    with open("data/vgm/vocabulary_list.txt", "w+") as out_file:
        for token in vocabulary:
            print(token, file=out_file)
    
print("Building data set...")        
token_to_index = {}
for index, token in enumerate(vocabulary):
    token_to_index[token] = index
    
work_list = []
total_duration = 0

for vgm_path in vgm_paths:
    for path, dirs, files in os.walk(vgm_path):
        for filename in fnmatch.filter(files, "*.vg?"):
            full_path = os.path.join(path, filename)
            
            if filename.endswith("z"):
                in_file = gzip.open(full_path, 'rb')
            else:
                in_file = open(full_path, 'rb')

            vgm = pyvgm.VGMFile.load(in_file)

            if vgm.total_duration < 22050: # Half a second
                print(f"SFX: {full_path}")
                continue
            else:
                print(full_path)

            work = bytearray()
            for command in vgm.commands:
                if not isinstance(command, pyvgm.PCMDataCommand) and not isinstance(command, pyvgm.PCMRamWriteCommand):
                    token_index = token_to_index[command.data.hex()]
                    work += token_index.to_bytes(2, byteorder='little')
                if isinstance(command, pyvgm.WaitCommand):
                    total_duration += command.ticks
            work += token_to_index["*END*"].to_bytes(2, byteorder='little')
            work_list.append(work)
            
            work = bytearray()
            for command in swap_channels_processor(vgm.commands):
                if not isinstance(command, pyvgm.PCMDataCommand) and not isinstance(command, pyvgm.PCMRamWriteCommand):
                    token_index = token_to_index[command.data.hex()]
                    work += token_index.to_bytes(2, byteorder='little')
                if isinstance(command, pyvgm.WaitCommand):
                    total_duration += command.ticks
            work += token_to_index["*END*"].to_bytes(2, byteorder='little')
            work_list.append(work)
            
            in_file.close()
        
random.shuffle(work_list)

corpus = b''.join(work_list)

validation_size = 2 * ((len(corpus) // 2) // 10)
training_size = len(corpus) - validation_size
print(f"training size={training_size}, validation size={validation_size}")

with open("data/vgm/train.bin", "w+b") as out_file:
    out_file.write(corpus[:training_size])
with open("data/vgm/val.bin", "w+b") as out_file:
    out_file.write(corpus[training_size:])

optimized_vocab_size = math.ceil(len(vocabulary) / 64) * 64

print(f"vocabulary size={len(vocabulary)}, optimized to {optimized_vocab_size}")
print(f"total duration={total_duration} ({(total_duration / 44100) / 60} minutes)")

with open("data/vgm/meta.pkl", 'w+b') as meta_out:
    pickle.dump( { 'vocab_size': optimized_vocab_size }, meta_out)
import os 
import pickle 
import numpy as np
import random as rnd 
import itertools
import trax 
import trax.fastmath.numpy as fnp 
from  trax import fastmath
from trax import layers as tl 
#
# trax.supervised.training.init_host_and_devices(None, 32)
rnd.seed(32)

dirname = './data/'
lines  = []
for filename in os.listdir(dirname):
    with open(os.path.join(dirname, filename)) as files:
        for line in files:
            pure_line = line.strip()
            if pure_line:
                lines.append(pure_line)

n_lines = len(lines)

# convert to lower case 
for i, line in enumerate(lines):
    lines[i] = line.lower()

eval_lines = lines[-1000:]
lines = lines[0:-1000]
print(f'Number of lines for training: {len(lines)}')
print(f'Number of lines for validattion: {len(eval_lines)}')

# convert each character in a list to a number 
print(f"ord('a'): {ord('a')}")

# write a function that takes in a single line and transforms each character into its unicode integer, we will refer to this as tensor 
def line_to_tensor(line, EOS_int=1):
    tensor = [ord(c) for c in line]
    tensor.append(EOS_int)
    return tensor

# implement the data generator 
def data_generator(batch_size, max_length, data_lines, line_to_tensor=line_to_tensor, shuffle=True):
    index = 0 
    cur_batch = []
    num_lines = len(data_lines)
    lines_index = [*range(num_lines)]
    if shuffle:
        rnd.shuffle(lines_index)
    while True:
        if index >= num_lines:
            index = 0
            if shuffle:
                rnd.shuffle(lines_index)

        line = data_lines[lines_index[index]]
        print(line)
        if len(line) < max_length:
            cur_batch.append(line)
        
        index += 1 

        if len(cur_batch) == batch_size:
            batch = []
            mask = []
            for li in cur_batch:
                tensor = line_to_tensor(li)
                pad = [0] * (max_length - len(tensor))
                tensor_pad = tensor + pad
                batch.append(tensor_pad)

                example_mask = [0 if t == 0 else 1 for t in tensor_pad]
                mask.append(example_mask)

            batch_np_arr = np.array(batch)
            mask_np_arr = np.array(mask)

            yield batch_np_arr, batch_np_arr, mask_np_arr
            
            cur_batch = []

tmp_lines = ['1245678901',
        '234567890',
        '345678901',
        '456789012']

tmp_data_gen = data_generator(batch_size=2, max_length=10, data_lines=tmp_lines, shuffle=False)

# for i in range(4):
#     tmp_batch = next(tmp_data_gen)
#     print(tmp_batch)

infinite_data_generator = itertools.cycle(data_generator(batch_size=2, max_length=10, data_lines=tmp_lines))    

ten_lines = [next(infinite_data_generator) for _ in range(10)]
print(len(ten_lines))

def GRULM(vocab_size=256, d_model=512, n_layers=2, mode='train'):
    model = tl.Serial(
        tl.ShiftRight(n_positions=1),
        tl.Embedding(vocab_size=vocab_size,d_feature=d_model),
        tl.GRU(d_model),
        tl.GRU(d_model),
        tl.Dense(vocab_size),
        tl.LogSoftmax())
    return model

batch_size = 32 
max_length = 64 



    

if __name__ == "__main__":
    pass

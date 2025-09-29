import os, random, json

all_files = [f for f in os.listdir('ravdess_ast_16k') if f.endswith('.wav')]
random.seed(42)
random.shuffle(all_files)
split = int(0.8 * len(all_files))
train_files = all_files[:split]
val_files = all_files[split:]

with open('ravdess_train.scp', 'w') as f:
    for name in train_files:
        f.write(name + '\n')
with open('ravdess_val.scp', 'w') as f:
    for name in val_files:
        f.write(name + '\n')
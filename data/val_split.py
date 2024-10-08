import random
random.seed(42)
ratio=0.2
# Read the contents of the file
with open('data/INCART_10s/val.txt', 'r') as file:
    lines = file.readlines()

with open('data/INCART_10s/all.txt', 'w') as file:
    file.writelines(lines)

# Shuffle the rows
random.shuffle(lines)

# Calculate 20% and 80% split
split_index = int(len(lines) *ratio)

# Write 20% of the shuffled rows to a new file
with open('data/INCART_10s/train.txt', 'w') as file:
    file.writelines(lines[:split_index])

# Write 80% of the shuffled rows back to the original file (or to another file if preferred)
with open('data/INCART_10s/val.txt', 'w') as file:
    file.writelines(lines[split_index:])

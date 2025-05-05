import random

# Set a seed for reproducibility
random.seed(42)

# Read the original train split file
with open("train_split.txt", "r") as f:
    lines = f.readlines()

# Shuffle the lines
random.shuffle(lines)

# Define the desired percentages
percentages = [10, 25, 50, 75]

# Generate and write the subsets
for pct in percentages:
    k = int(len(lines) * pct / 100)
    subset = lines[:k]
    output_file = f"/data_splits/sen1floods11/train_{pct}.txt"
    with open(output_file, "w") as f_out:
        f_out.writelines(subset)
    print(f"Created {output_file} with {k} entries.")

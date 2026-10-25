with open('/home/manas/reservoir/sediment_pipeline.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'raise ValueError(

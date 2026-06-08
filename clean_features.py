import json
import re

with open('models/dead_factors.json', 'r', encoding='utf-8') as f:
    dead = json.load(f)['dead_factors']

with open('ml_pipeline.py', 'r', encoding='utf-8') as f:
    code = f.read()

v3_start = code.find('V3_FEATURES = [')
v3_end = code.find(']', v3_start) + 1
target = code[v3_start:v3_end]

lines = target.split('\n')
new_lines = []
for line in lines:
    if line.strip().startswith('#') or 'V3_FEATURES = [' in line or line.strip() == ']':
        new_lines.append(line)
        continue
    parts = line.split(',')
    new_parts = []
    for p in parts:
        if not p.strip(): continue
        match = re.search(r'\"([^\"]+)\"', p)
        if match:
            if match.group(1) not in dead:
                new_parts.append(p.strip())
        else:
            new_parts.append(p.strip())
    if new_parts:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(indent + ', '.join(new_parts) + ',')

final_target = '\n'.join(new_lines)
final_target = final_target.rstrip(',')  # clean up last trailing comma if necessary

print(final_target)

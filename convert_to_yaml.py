import os

with open('requirements.txt', 'r') as f:
    dependencies = [line.strip() for line in f]

with open('environment.yml', 'a') as f:
    f.write('\n')
    f.write('dependencies:\n')
    for dep in dependencies:
        f.write(f'  - {dep}\n')

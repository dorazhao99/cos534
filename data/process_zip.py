import sys

filename = sys.argv[1]

files = open(filename, 'r')
lines = files.readlines()[3:-2]
for i,line in enumerate(lines):
    line = line.split()[-1]
    line = line.split()[0]
    lines[i] = line
    
f = open(filename, 'w')
for line in lines:
    f.write('{}\n'.format(line))
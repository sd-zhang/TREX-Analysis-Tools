import os, numpy, pandas
import matplotlib


cwd = os.getcwd()
path = os.path.join(cwd, 'sundance_candidates')
candidate_list_path = os.path.join(path, 'candidate_scan.txt')

with open(candidate_list_path) as fp:
    contents = fp.readlines()
    num_entries = int(len(contents)/4)
    candidates = []
    for entry in range(num_entries):
        candidate = contents[entry*4:(entry*4+4)]
        candidate_houses = candidate[0][1:-2].replace("'","").split(',')
        candidate_OV = int(candidate[1][:-1].split(':')[1])
        candidate_UV = int(candidate[2][:-1].split(':')[1])

        candidates.append([candidate_houses, candidate_OV, candidate_UV])


candidates = sorted(candidates, key = lambda x: (x[1], x[2]), reverse=True)
top_5 = candidates[:5]
for candidate in top_5:
    print(candidate[0])
    print('OVs: ', candidate[1])
    print('UVs: ', candidate[2])

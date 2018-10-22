import os, sys

for i in range(140):
    logfn = 'naming_pre%d_output' % (i+1)
    with open(logfn, 'r') as fid:
        lines = fid.readlines()
        f = False
        for j in range(len(lines)-2):
            line = lines[j]
            if line.find('error') >= 0:
                f = True
                break
        if f is True:
            #os.system
            print('sbatch scripts/feat.pbs%d' % (i+1))

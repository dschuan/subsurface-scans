#!/usr/bin/env python
# coding: utf-8
import glob
import json
from matplotlib import pyplot as plt
import numpy as np

db = lambda x: -10 * np.log10(abs(x))
convertDb = np.vectorize(db)
print('hi')
filepaths = [file for file in glob.glob('./results/*.json') if 'simple' not in file]

for file in filepaths:
    with open(file) as f:
        data = json.load(f)
    values = list(data.values())
    for val in values:
        plt.clf()
        try:
            amp = np.asarray(val['amplitude'])
            db_amp = convertDb(amp)
            #time = np.asarray(val['time'])
            db_amp = db_amp.tolist()
            val['amplitude'] = db_amp
            #plt.plot(db_amp, time)
            #plt.show()
            #print(len(db_amp))
        except:
            pass
    filename = file.replace('.json', '_dB.json')
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sklearn.metrics as met
import pandas as pd


def extractDat(file):
    arr = np.loadtxt(file, dtype=str, delimiter=',', comments='%')
    arr = arr[:, 0:3]
    arr = arr.astype(np.float)
    freq = arr[:, 0]
    amp = arr[:, 1]
    ang = arr[:, 2]
    return freq, amp, ang

def getAverage(files):
    amp_list = []
    freq = -1
    ang = -1
    if len(files) > 0:
        for file in files:
            if type(freq) is int and type(ang) is int:
                freq, amp, ang = extractDat(file)
            else:
                _, amp, _ = extractDat(file)
            amp_list.append(amp.tolist())
        amp_list = np.array(amp_list)
        amp_list = np.mean(amp_list, axis=0)
        return freq, amp_list, ang


def getData(filenames, material, is_soil = False):
    if is_soil == False:
        background = [file for file in filenames if "nem" in file]
        back_freq, back_amp, back_ang = getAverage(background)
    else:
        background = [file for file in filenames if "box" in file]
        back_freq, back_amp, back_ang = getAverage(background)

    print('backamp', back_amp.shape)
    files = [file for file in filenames if material in file]
    freq, amp, ang = getAverage(files)
    print(freq.shape)
    amp = amp - back_amp
    # peaks, _ = find_peaks(amp)
    # amp = np.take(amp, peaks)
    # freq = np.take(freq, peaks)
    ang = ang - back_ang
    return freq, amp, ang

def getAllData(filenames, material):
    background = [file for file in filenames if "nem" in file]
    back_freq, back_amp, back_ang = getAverage(background)

    files = [file for file in filenames if material in file]
    res = []
    for file in files:
        freq, amp, ang = extractDat(file)
        freq = freq - back_freq
        amp = amp - back_amp
        # peaks, _ = find_peaks(amp)
        # amp = np.take(amp, peaks)
        # freq = np.take(freq, peaks)
        ang = ang - back_ang
        res.append((freq, amp, ang, file))
    return res

def analyzeSimilarity(results, avg_data, method='Frechet'):
    #takes in return value from getAllData and getAvgData
    avg_freq, avg_amp, _, material = avg_data
    avg = np.zeros(shape=(len(avg_freq), 2))
    avg[:, 0] = avg_freq
    avg[:, 1] = avg_amp
    similarity = []
    for res in results:
        freq, amp, _, file = res
        file = file.replace(r'./vna', '')
        file = file.replace('\\', '')
        raw = np.zeros(shape=(len(avg_freq), 2))
        raw[:, 0] = freq
        raw[:, 1] = amp
        sim_score = met.mean_squared_error(avg[:,1], raw[:,1])
        sim_score = round(sim_score, 3)
        similarity.append((file, sim_score))
    return similarity


def plotAll(filenames, material, is_soil = False):
    if is_soil == False:
        background = [file for file in filenames if "empty" in file]
        back_freq, back_amp, back_ang = getAverage(background)
    else:
        background = [file for file in filenames if "box" in file]
        back_freq, back_amp, back_ang = getAverage(background)

    files = [file for file in filenames if material in file]
    for file in files:
        freq, amp, ang = extractDat(file)
        amp = amp - back_amp
        peaks, _ = find_peaks(amp)
        amp = np.take(amp, peaks)
        freq = np.take(freq, peaks)
        ang = ang - back_ang
        plt.plot(freq, amp, label=file)
    plt.title('All signals of ' + material + ' with background signal removed')
    plt.show()

def getAvgData(materials):
    res = []
    for material in materials:
        freq, amp, ang = getData(filenames, material)
        res.append((freq, amp, ang, material))
    return res

def plotAvg(materials):
    res = getAvgData(materials)
    fig, ax = plt.subplots(len(materials), 1, sharex=True, sharey=True)
    for i in range(len(res)):
        freq, amp, ang, material = res[i]
        ax[i].plot(freq, amp, label=material)
        ax[i].legend()
    plt.title('Average Plot of Targets with Background Removed')
    plt.show()

def create_comparison_table(filenames, ref_mat, compared_materials):
    """
    filenames takes in array of files,
    ref_mat takes in string of reference material,
    compared_materials takes in array of material data to be compared to ref
    """
    df = pd.DataFrame()
    ref_dat = getAvgData([ref_mat])
    ref_dat = ref_dat[0]
    for mat in compared_materials:
        mat_data = getAllData(filenames, mat)
        sim_matrix = analyzeSimilarity(mat_data, ref_dat)
        sim_matrix = np.array(sim_matrix)
        column_header1 = mat + ' DataSets'
        column_header2 = mat + ' Difference'
        kwargs = { column_header1: sim_matrix[:, 0], column_header2: sim_matrix[:,1], ' ': ' '}
        df = df.assign(**kwargs)
    return df

if __name__ == '__main__':

    filenames = glob.glob('./vna/*.dat')
    material = 'metal'
    material2 = 'dply'
    material3 = 'sply'
    material4 = 'alum'

    df_alum = create_comparison_table(filenames, material4, [material4, material2])
    df_wood = create_comparison_table(filenames, material2, [material2, material4])

    with pd.ExcelWriter('comparison_table.xlsx') as writer:  # doctest: +SKIP
        df_alum.to_excel(writer, sheet_name='Aluminum Reference')
        df_wood.to_excel(writer, sheet_name='Wood Reference')
    #plotAvg([material, material2, material4])
    #plotAll(filenames, material2, is_soil=False)

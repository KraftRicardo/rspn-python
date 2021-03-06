import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# C++
# 10x10000: ################# Time = 2835
# 10x20000: ################# Time = 5765
# 10x30000: ################# Time = 8694
# 10x40000: ################# Time = 11595
# 10x50000: ################# Time = 14511
# 10x60000: ################# Time = 17396
# 10x70000: ################# Time = 20295
# 10x80000: ################# Time = 23155
# 10x90000: ################# Time = 26227
# 10x100000: ################# Time = 29078

# Python
# 10x10000 Time= 6.31577205657959
# 10x20000 Time= 10.022038221359253
# 10x30000 Time= 12.41782259941101
# 10x40000 Time= 15.17801308631897
# 10x50000 Time= 17.895564079284668
# 10x60000 Time= 21.351463794708252
# 10x70000 Time= 24.891441345214844
# 10x80000 Time= 27.812203407287598
# 10x90000 Time= 31.205123901367188
# 10x100000 Time= 34.1586639881134

def plotIncreasingRows():
    plt.rcParams["figure.dpi"] = 300
    plt.figure()

    index = ['10000', '20000', '30000', '40000', '50000', '60000', '70000', '80000', '90000', '100000']
    python = [6.31577205657959, 10.022038221359253, 12.41782259941101, 15.17801308631897, 17.895564079284668, 21.351463794708252, 24.891441345214844,  27.812203407287598, 31.205123901367188, 34.1586639881134]
    cpp = [2.835, 5.765, 8.694, 11.595, 14.511, 17.396, 20.295, 23.155, 26.227, 29.078]

    df = pd.DataFrame({'Python': python, 'C++': cpp}, index=index)

    ax = df.plot.bar(rot='0', color=['blue', 'red'])
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Time in seconds")
    # plt.title("C++ vs. Python performance for increasing number of rows")

    plt.show()

# c++
# 5x10000: ################# Time = 1019
# 10x10000: ################# Time = 2091
# 15x10000: ################# Time = 3016
# 20x10000: ################# Time = 5354
# 25x10000: ################# Time = 6206
# 30x10000: ################# Time = 7654
# 35x10000: ################# Time = 10612
# 40x10000: ################# Time = 11596
# 45x10000: ################# Time = 12487
# 50x10000: ################# Time = 13207
# 55x10000: ################# Time = 13917
# 60x10000: ################# Time = 15398
# 65x10000: ################# Time = 20160
# 70x10000: ################# Time = 20937
# 75x10000: ################# Time = 21501
# 80x10000: ################# Time = 22170
# 85x10000: ################# Time = 23882
# 90x10000: ################# Time = 24819

# Python
# 5x10000 Time= 3.1239540576934814
# 10x10000 Time= 5.805063247680664
# 15x10000 Time= 8.08960771560669
# 20x10000 Time= 10.221647500991821
# 25x10000 Time= 11.61333417892456
# 30x10000 Time= 13.07772946357727
# 35x10000 Time= 14.322115659713745
# 40x10000 Time= 15.870172262191772
# 45x10000 Time= 18.75518774986267
# 50x10000 Time= 21.304199934005737
# 55x10000 Time= 20.97739815711975
# 60x10000 Time= 21.943511486053467
# 65x10000 Time= 25.841979503631592
# 70x10000 Time= 28.16636371612549
# 75x10000 Time= 30.238733291625977
# 80x10000 Time= 32.17600655555725
# 85x10000 Time= 35.48106002807617
# 90x10000 Time= 38.932941198349

def plotIncreasingColumns():
    plt.rcParams["figure.dpi"] = 300
    plt.figure()

    index = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90']
    python = [3.1239540576934814, 5.805063247680664, 8.08960771560669, 10.221647500991821, 11.61333417892456,
              13.07772946357727, 14.322115659713745, 15.870172262191772, 18.75518774986267, 21.304199934005737,
              20.97739815711975, 21.943511486053467, 25.841979503631592, 28.16636371612549, 30.238733291625977,
              32.17600655555725, 35.48106002807617, 38.932941198349]
    cpp = [1.019, 2.091, 3.016, 5.354, 6.206, 7.654, 10.612, 11.596, 12.487, 13.207, 13.917, 15.398, 20.160, 20.937,
           21.501, 22.170, 23.882, 24.819]

    df = pd.DataFrame({'Python': python, 'C++': cpp}, index=index)

    ax = df.plot.bar(rot='0', color=['blue', 'red'])
    ax.set_xlabel("Number of columns")
    ax.set_ylabel("Time in seconds")
    # plt.title("C++ vs. Python performance for increasing number of columns")

    plt.show()


if __name__ == '__main__':
    plotIncreasingRows()
    plotIncreasingColumns()

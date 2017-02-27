import pandas as pd
import numpy as np
import random


# dummy (timestamp, heart, steps) >> pipeline >> heatmap >> acf plot
# create dummy dataset


def get_dummy(method='d&w', add_randomness=True, rnd=0.2):
    date_time = pd.date_range(start='2016-08-21 00:00:00', end='2016-10-26 23:45:00', freq='15min')
    heart = abs(np.floor((np.random.normal(50, 120, len(date_time)))))
    steps = abs(np.floor((np.random.normal(1000, 200, len(date_time)))))

    # heart = np.random.randint(43, 157, len(date_time))
    # steps = np.random.randint(0, 2307, len(date_time))
    data = pd.DataFrame({'date_time': date_time, 'heart': heart, 'steps': steps})

    if method == 'd&w':
        for i in data.index:
            # make the user sedentary from 16 to 21 on Monday and Thursday
            if (data.date_time[i].dayofweek in [0, 3, 5]) & (data.date_time[i].hour in range(16, 21)):
                data.loc[i, 'steps'] = 0
    elif method == 'd!w':
        date_set = set([x.date() for x in data.date_time])
        date_sample = random.sample(date_set, len(date_set) // 3)
        for i in data.index:
            if (data.date_time[i].date() in date_sample) & (data.date_time[i].hour in range(16, 21)):
                data.loc[i, 'steps'] = 0
    elif method == 'w!d':
        for i in data.index:
            time_sample = random.sample(range(18), 1)
            if (data.date_time[i].dayofweek in [0, 3, 5]) \
                    & (data.date_time[i].hour in range(time_sample[0], time_sample[0]+4)):
                data.loc[i, 'steps'] = 0
    elif method == 'rnd':
        index_sample = np.random.choice(data.index, size=int(round(.1 * len(data))), replace=False, p=None)
        # index_sample = random.sample(data.index, len(data.index) // 10)
        data.loc[index_sample, 'steps'] = 0

    if add_randomness:
        rndidx = np.random.choice(data.index, size=int(round(.2 * len(data))), replace=False, p=None)
        for i in rndidx:
            data.loc[i, 'steps'] = abs(int(round(np.random.normal(1, 10))))

    return data

"""
Input: csv (date_time, steps, heart)
Output: json (date, time, notification)

"""

# 1. get user profile

# 2. get user data

# 3. data transformation
"""
on original data set:
  new variables: mvpa, vpa, met
new data set: for checking correlation between day of week and daily met >> to decide which plan to use
"""

from __future__ import print_function
import subprocess
import configparser
import datetime as dt
import math
from datetime import timedelta
import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
from scipy.stats.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from lib.MachineLearningModels import *


def get_user_profile():
    config = configparser.ConfigParser()
    config.read('user.ini')
    age = config.get('default', 'user_age')
    choice = config.get('default', 'user_choice')
    goal = config.get('default', 'user_goal')
    user_profile = User(int(age), choice, int(goal), 0, 0)
    return user_profile


class User:
    def __init__(self, age, choice, goal, completed, last_week):
        self.age = age
        self.choice = choice
        # self.max_hr = 220 - self.age
        # self.mvpa_low = 0.64 * self.max_hr
        # self.mvpa_high = 0.77 * self.max_hr
        # self.vpa_low = 0.77 * self.max_hr
        # self.vpa_high = 0.89 * self.max_hr
        self.goal = goal
        self.completed = completed
        self.last_week = last_week
        self.current_goal = self.goal - self.completed - self.last_week
        self.yesterday_vpa = False
        self.today_vpa = False

    def update_goal(self):
        self.current_goal = self.goal - self.completed - self.last_week


# 3.1 get mvpa, vpa, met
# mark out MVPA intervals
def get_pa_params(data, user):
    max_heart = 220 - user.age
    # mvpa
    mvpa_low, mvpa_high = 0.64 * max_heart, 0.77 * max_heart
    mvpa_mask = (data.heart >= mvpa_low) & (data.heart < mvpa_high) & (data.steps > 15)
    data['mvpa'] = mvpa_mask.apply(lambda x: 1 if x else 0)
    # vpa
    vpa_low, vpa_high = 0.77 * max_heart, 0.89 * max_heart
    vpa_mask = (data.heart >= vpa_low) & (data.heart < vpa_high) & (data.steps > 15)
    data['vpa'] = vpa_mask.apply(lambda x: 1 if x else 0)
    data['met'] = data['mvpa'] * 4 * 15 + data['vpa'] * 8 * 15
    return data


# 3.2 get daily met, daily inactive, weekly met,
def get_pa_summary(data):
    vpa_byday = data.groupby([data.index.week, data.index.date])['vpa'].sum().reset_index()
    vpa_byday.rename(columns={'level_0': 'week', 'level_1': 'date'}, inplace=True)
    met_byday = data.groupby([data.index.week, data.index.dayofweek])['met'].sum().reset_index()
    met_byday.rename(columns={'level_0': 'week', 'level_1': 'dayofweek'}, inplace=True)
    met_byday['active'] = met_byday.met.apply(lambda x: 1 if x > 0 else 0)
    met_byweek = met_byday.groupby(data.week)['met'].sum().reset_index()
    met_byweek.rename(columns={'level_0': 'week'}, inplace=True)
    return vpa_byday, met_byday, met_byweek


# 4. data analysis of data set from 3.2; output: 'random' or 'patterned'
# correlation analysis
def get_pa_plan_mode(data):
    # get accumulated met of each day, starting from Monday
    met_from_monday = data['met'].tolist()
    for i in range(1, len(data)):
        if data.index.week[i] == data.index.week[i - 1]:
            met_from_monday[i] += met_from_monday[i - 1]
    corr, p = spearmanr(data.index.dayofweek, met_from_monday)[0:2]
    if (p <= 0.05) & (abs(corr) > 0):
        print("Statistically significant correlation found between days of week and daily MET.")
        print("Execute patterned plan generator.")
        plan_mode = 'pattern'
    else:
        print("No strong correlation found between days of week and daily MET.")
        print("Execute random plan generator.")
        plan_mode = 'random'
    return plan_mode


def update_user_pa_profile(vpa_daily, met_weekly, user):
    # TODO: temporary method
    """
    import time
    today = pd.to_datetime(time.strftime("%d/%m/%Y"))
    """
    config = configparser.ConfigParser()
    config.read('user.ini')
    today = pd.to_datetime(config.get('default', 'today'))
    current_week_num = today.week

    # get MET completed this week
    try:
        user.completed = met_weekly.loc[met_weekly['week'] == current_week_num, 'met']
    except KeyError:
        pass

    # get MET completed last week
    try:
        user.last_week = met_weekly.loc[met_weekly['week'] == current_week_num - 1, 'met']
    except KeyError:
        pass

    # check if the user performed VPA today
    try:
        if vpa_daily.loc[(vpa_daily == today.date()), 'vpa'] > 0:
            user.yesterday_vpa = True
        else:
            user.yesterday_vpa = False
    except KeyError:
        pass

    # check if the user performed VPA yesterday
    try:
        if vpa_daily.loc[(vpa_daily == (today.date() - timedelta(days=1))), 'vpa'] > 0:
            user.yesterday_vpa = True
        else:
            user.yesterday_vpa = False
    except KeyError:
        pass

    # update PA goal
    user.update_goal()


# 5. plan generation
def get_dates_for_notification(today_vpa, yesterday_vpa, today):
    # generate days left in this week

    if dt.datetime.today().time().hour >= 17:
        if today_vpa:
            if today.dayofweek == 6:  # TODO: notify user to sync on Monday morning
                days_left = pd.Series([])
                days_left_start_date = today
            else:
                days_left = pd.Series(list(range(today.dayofweek + 1, 7)))
                days_left_start_date = today + timedelta(days=1)
        else:  # no vpa so far today
            days_left = pd.Series(list(range(today.dayofweek + 1, 7)))
            days_left_start_date = today + timedelta(days=1)

    else:
        if yesterday_vpa:
            if today.dayofweek == 6:  # TODO: notify user to sync on Monday morning
                days_left = pd.Series([])
                days_left_start_date = today
            else:
                if today_vpa:
                    days_left = pd.Series(list(range(today.dayofweek + 2, 7)))
                    days_left_start_date = today + timedelta(days=1)
                else:
                    days_left = pd.Series(list(range(today.dayofweek + 1, 7)))
                    days_left_start_date = today + timedelta(days=1)
        elif today_vpa:
            if today.dayofweek == 6:  # TODO: notify user to sync on Monday morning
                days_left = pd.Series([])
                days_left_start_date = today
            else:
                days_left = pd.Series(list(range(today.dayofweek + 1, 7)))
                days_left_start_date = today + timedelta(days=1)
        else:
            days_left = pd.Series(list(range(today.dayofweek, 7)))
            days_left_start_date = today
        return days_left_start_date, days_left


def random_plan_generator(user):
    """
    Updates every time user syncs data
    :param met_to_complete:
    :param choice:
    :return:
    """

    # TODO: temporary method use
    """
    import time
    today = pd.to_datetime(time.strftime("%d/%m/%Y"))
    """
    config = configparser.ConfigParser()
    config.read('user.ini')
    today = pd.to_datetime(config.get('default', 'today'))

    # get user current status
    met_to_complete = user.current_goal
    choice = user.choice
    yesterday_vpa = user.yesterday_vpa
    today_vpa = user.today_vpa
    # model of randomly assign PA of choice to days left in this week to fulfill MET to complete
    days_left_start_date, days_left = get_dates_for_notification(today_vpa, yesterday_vpa, today)
    days_left_date_range = pd.date_range(days_left_start_date, periods=len(days_left), freq='D').strftime(
        '%Y-%m-%d')

    if choice == 'MVPA':
        bouts = math.ceil(met_to_complete / (4 * 15))
    else:
        bouts = math.ceil(met_to_complete / (8 * 15))

    if len(days_left) > 0:
        base_bouts = math.floor(bouts / len(days_left))
        extra_bouts = bouts % len(days_left)
        plan = pd.DataFrame({'date': days_left_date_range, 'day_of_week': days_left, 'week': today.week,
                             'choice': choice, 'bouts': base_bouts})
        days_extra_bouts = days_left.sample(extra_bouts, replace=False)  # as extra_bouts must be smaller than days_left
        for i in days_extra_bouts:
            plan.loc[plan['day_of_week'] == i, 'bouts'] += 1
        return plan
    else:
        return None


def pattern_plan_generator(user, data):  # input daily MET training data
    # get user status
    met_to_complete = user.current_goal  # TODO: for each day met_to_complete should also add the usual met predicted yet unperformed as usual
    choice = user.choice
    yesterday_vpa = user.yesterday_vpa
    today_vpa = user.today_vpa

    # TODO: temporary method use
    """
    import time
    today = pd.to_datetime(time.strftime("%d/%m/%Y"))
    """
    config = configparser.ConfigParser()
    config.read('user.ini')
    today = pd.to_datetime(config.get('default', 'today'))
    dayofweek_today = pd.to_datetime(today).dayofweek

    # predict this week's PA trend
    pattern_tree = DecisionTreeClassifier()
    pattern_tree = pattern_tree.fit(np.array(data[['day_of_week']]), data.loc[:, 'inactive'])
    predicted = pd.DataFrame({'day_of_week': list(range(7))})
    predicted['predicted_inactive'] = pattern_tree.predict(predicted[['day_of_week']])
    accuracy = pattern_tree.score(np.array(data[['day_of_week']]), data.loc[:, 'inactive'])
    print(accuracy)

    # print(pattern_tree.predict_proba(predicted[['day_of_week']]))
    # print(pattern_tree.classes_)

    days_left_start_date, days_left = get_dates_for_notification(today_vpa, yesterday_vpa, today)

    predicted = predicted.loc[predicted['day_of_week'].isin(days_left),:]
    predicted.reset_index(drop=True, inplace=True)
    predicted.drop('index', axis=1, inplace=True)
    predicted['date'] = pd.date_range(days_left_start_date, periods=len(days_left), freq='D').strftime('%Y-%m-%d')

    # assign PA to days
    if choice == 'MVPA':
        bouts = math.ceil(met_to_complete / (4 * 15))
    else:
        bouts = math.ceil(met_to_complete / (8 * 15))

    predicted['week'] = pd.to_datetime(today).week
    predicted['choice'] = choice
    predicted['bouts'] = 0

    if len(days_left) > 0:
        num_inactive = len(predicted[predicted['predicted_inactive'] == 1])
        base_bouts = math.floor(bouts / num_inactive)
        extra_bouts = bouts % num_inactive
        # assign base bouts to inactive days
        for i in predicted.index:
            if predicted.loc[i, 'predicted_inactive'] == 1:
                predicted.loc[i, 'bouts'] = base_bouts

        # assign extra bouts to all days left, including active days
        # check if the user needs to be more active during weekdays
        # criteria: user did not fulfill the PA goal and tend to have VPA during weekend
        dayofweek_saturday = 5
        dayofweek_sunday = 6
        saturday_vpa = False
        sunday_vpa = False
        if pd.Series(dayofweek_saturday).isin(predicted.loc[predicted['choice'] == 'VPA', 'day_of_week'])[0]:
            saturday_vpa = True
        else:
            print("saturday checked")
        if pd.Series(dayofweek_sunday).isin(predicted.loc[predicted['choice'] == 'VPA', 'day_of_week'])[0]:
            sunday_vpa = True
        else:
            print("sunday checked")
        if (saturday_vpa | sunday_vpa) & (met_to_complete > 0):
            if days_left[0] <= 4:
                weekday_left = list(set(days_left) - {5, 6})
                days_extra_bouts = weekday_left.sample(extra_bouts)
            else:
                days_extra_bouts = []
        else:
            days_extra_bouts = days_left.sample(extra_bouts)

        for i in days_extra_bouts:
            predicted.loc[predicted['day_of_week'] == i, 'bouts'] += 1
        print("print predicted activity")
        print(predicted)
        # predicted.to_csv(config.get('default', 'fitbit_csv') + '_presentation_pa_predicted.csv')
        return predicted
    else:
        return None


# 6. notification generation
def get_plan_notification(plan):
    notification = []
    for i in plan.index:
        minutes = plan.loc[i, 'bouts'] * 15
        if plan.loc[i, 'bouts'] > 0:
            if plan.loc[i, 'choice'] == 'MVPA':
                message = "How about a " + str(minutes) + " minutes brisk walking later?"
                notification.append(message)
            if plan.loc[i, 'choice'] == 'VPA':
                message = "How about a " + str(minutes) + " minutes jogging later?"
                notification.append(message)
        else:
            notification.append('None')
    plan['notification'] = notification
    print(plan)
    return plan

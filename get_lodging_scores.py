import numpy as np
import pandas as pd
from datetime import date
 
def get_labels(bin_size):
    # read by default 1st sheet of an excel file
    dataframe1 =  pd.read_csv('Lodging_scores.csv', parse_dates=['VisualScoreDate','FlightDate'])
    scores = np.array(dataframe1)
    labels = []
    #create labels, [ROI, Visual SCORE, FLIGHTFolder, BIN]
    for i in range(scores.shape[0]):
        labels.append((scores[i,2], scores[i, 4], scores[i,6].replace("m","M"), create_bin(bin_size, scores[i,4])))
    return labels

def create_bin(bin_size, vision_score):
    bin_num = int(vision_score // bin_size)
    if (vision_score == 99 and bin_size == 33):
        return 2
    if (vision_score == 100 and bin_size == 1):
        return bin_num
    if (vision_score == 100):
        return bin_num - 1
    return bin_num

def get_dates_and_labes(bin_size):
    dataframe1 =  pd.read_csv('Lodging_scores.csv', parse_dates=['VisualScoreDate','FlightDate'])
    scores = np.array(dataframe1)
    labels = []
    #create labels, [ROI, Visual SCORE, FLIGHTFolder, BIN]
    for i in range(scores.shape[0]):
        date = timestamp_to_int(scores[i, 3])
        labels.append((scores[i,2], scores[i, 4], scores[i,6].replace("m","M"), create_bin(bin_size, scores[i,4]), date))
    return labels

def timestamp_to_int(timestamp: pd.Timestamp) -> int:
    """Convert a pandas Timestamp to an integer between 0 and 365."""
    date_obj = timestamp.date()
    start_date = date(date_obj.year, 1, 1)
    delta = date_obj - start_date
    return delta.days

def it_bin(scores, bin_size):
    bins = []
    for i in range(len(scores)):
        bins.append(create_bin(bin_size, scores[i]))
    return bins

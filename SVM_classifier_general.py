import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
import numpy as np # linear algebra
import get_lodging_scores
import pickle
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import glob

def match_pic_label_to_names(features, labels, names):
    n = len(labels)
    m = len(names)
    data_frame = []
    error = []
    for i in range(n):
        matched = False
        for j in range(m):
            if (str(labels[i][0]) == str(names[j][0]) and labels[i][2] == names[j][1]):
                data_frame.append(np.hstack((features[j], labels[i][3])))
                matched = True
    return data_frame

def match_pic_label_to_names_new(features, labels, names, date_):
    n = len(labels)
    m = len(names) 
    error = []
    for i in range(n):
        matched = False
        for j in range(m):
            if (date_ == True):
                if (str(labels[i][0]) == str(names[j][0]) and labels[i][2] == names[j][1]):
                    features[j].append(labels[i][4])
                    features[j].append(labels[i][3])
                    matched = True
            else:
                if (str(labels[i][0]) == str(names[j][0]) and labels[i][2] == names[j][1]):
                    features[j].append(labels[i][3])
                    matched = True
    return features

def match_pic_label_to_names_one_year(features, labels, names, date_):
    n = len(labels) 
    m = len(names) 
    validation = []
    training = []
    for i in range(n):
        for j in range(m):
            if (date_ == True):
                if (str(labels[i][0]) == str(names[j][0]) and labels[i][2] == names[j][1]):
                    if(names[j][1][0] == '1' and names[j][1][1] == '9'):
                        features[j].append(labels[i][3])
                        validation.append(features[j])
                    else:
                        features[j].append(labels[i][3])
                        training.append(features[j])
            else:
                if (str(labels[i][0]) == str(names[j][0]) and labels[i][2] == names[j][1]):
                    if(names[j][1][0] == '1' and names[j][1][1] == '9'):
                        features[j].append(labels[i][3])
                        validation.append(features[j])
                    else:
                        features[j].append(labels[i][3])
                        training.append(features[j])
                    
    
    return training, validation 

#error is images that has no labels
def match_pic_names_to_label(features, labels, names):
    m = len(labels)
    n = len(names)
    data_frame = []
    error = []
    for i in range(n):
        matched = False
        for j in range(m):
            if (str(labels[j][0]) == str(names[i][0]) and labels[j][2] == names[i][1]):
                data_frame.append(np.hstack((features[i], labels[j][3])))
                matched = True
        if (matched == False):
            error.append((names[i]))
    return data_frame, error

def save_results(results, path):
    path = path.replace("Features/", "")
    path = "results/" + path.replace(".npy", ".txt")
    with open(path, 'a') as f:
        f.write(results)
        f.write("\n")
    print("Results Saved Succesfull")

def load_labels(bin_size, date_):
    if (date_ == True):
        labels = get_lodging_scores.get_dates_and_labes(bin_size)
    else:
        labels = get_lodging_scores.get_labels(bin_size)
    #COnverts into set (no dublicates in set)
    my_set = set(map(tuple, labels))
    # convert set back to list of arrays
    labels = list(map(np.array, my_set))
    return labels

def partition_data(data_frame, percentage):
    partition = int(len(data_frame)*percentage/100)
    x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
    y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()
    print(x_train)
    print("Data Partioned Succesfull")
    return x_train, x_test, y_train, y_test
    

def partition_data_list(data_frame, percentage):
    partition = int(len(data_frame) * percentage / 100)
    x_train, x_test = [row[:-1] for row in data_frame[:partition]], [row[:-1] for row in data_frame[partition:]]
    y_train, y_test = [row[-1] for row in data_frame[:partition]], [row[-1] for row in data_frame[partition:]]
    print("Data Partitioned Successfully")
    return x_train, x_test, y_train, y_test

#x_train takes everything but the last row
def partition_data_list_year(validation, train):
    if (validation is None or train is None):
        print("Error: validation or train is None")
        return None
    x_train, x_test = [row[:-1] for row in train], [row[:-1] for row in validation]
    y_train, y_test = [row[-1] for row in train], [row[-1] for row in validation]
    print("Data Partitioned Successfully")
    return x_train, x_test, y_train, y_test

def save_model(clf, filename):
    filename = "model/" + filename.replace(".npy", ".sav")
    pickle.dump(clf, open(filename, 'wb'))
    print("Model Saved Succesfull")
    
def create_conf_matrix(y_test, y_pred, n_bins):
    conf_matrix = [[0 for j in range(n_bins)] for i in range(n_bins)]
    for i in range(len(y_test)):
        conf_matrix[int(y_test[i])][int(y_pred[i])] += 1
    
    matrix_string = ""
    for row in conf_matrix:
        row_string = " ".join(str(element) for element in row)
        matrix_string += row_string + "\n"
    return matrix_string
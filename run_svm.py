#We need these in this file:
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
import numpy as np # linear algebra
import get_lodging_scores
import load_read_name_extractor as lrne
import SVM_classifier_general as svm_general
import pickle
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import glob

def main():
    features_paths = [
    'Features\\hog_features_(155, 543, 3)_cells_(8, 8)_block_(1, 1)_norm_L2.npy',
    'Features\\hog_features_(155, 543, 3)_cells_(8, 8)_block_(2, 2)_norm_L2.npy',
    'Features\\hog_features_(155, 543, 3)_cells_(8, 8)_block_(3, 3)_norm_L2.npy',
    'Features\\hog_features_(155, 543, 3)_cells_(8, 8)_block_(4, 4)_norm_L2.npy',
    'Features\\hog_features_(155, 543, 3)_cells_(8, 8)_block_(5, 5)_norm_L2.npy',
    'Features\\hog_features_(155, 543, 3)_cells_(8, 8)_block_(6, 6)_norm_L2.npy']
    names_paths = [
    'Features\\img_names_(155, 543, 3)_cells_(8, 8)_block_(1, 1)_norm_L2.npy',
    'Features\\img_names_(155, 543, 3)_cells_(8, 8)_block_(2, 2)_norm_L2.npy',
    'Features\\img_names_(155, 543, 3)_cells_(8, 8)_block_(3, 3)_norm_L2.npy',
    'Features\\img_names_(155, 543, 3)_cells_(8, 8)_block_(4, 4)_norm_L2.npy',
    'Features\\img_names_(155, 543, 3)_cells_(8, 8)_block_(5, 5)_norm_L2.npy',
    'Features\\img_names_(155, 543, 3)_cells_(8, 8)_block_(6, 6)_norm_L2.npy']
    svm_general.run_mult_models([features_paths[0]], names_paths, bin_size = 10, svm_kernel="rbf")

if __name__ == "__main__":
    main()
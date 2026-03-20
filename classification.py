#Library imports
import shutil
import os
import re
import pandas as pd
import string
from iteration_utilities import intersperse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
# General
import pandas as pd
import random
from tqdm import tqdm
# Feature extraction approach
from sklearn.preprocessing import scale
import nltk
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#from feature_extraction import *
#from pretreatment import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import numpy
all_langs = ['de','en', 'es', 'fr', 'it', 'pt', 'nl']
import os
import re
import numpy
import pandas as pd
import string
from iteration_utilities import intersperse
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statistics
import spacy
# spacy.cli.download('de_core_news_sm')
spacy.cli.download("en_core_web_sm")
from collections import Counter
import multiprocessing


def vary_classifier_and_features_version_combined_unchunked(dir_with_data='Data_for_classification_unchunked', mode='halfhalf', num_authors=8, dir_results='',
                                 list_classifiers=['rf', 'svm', 'xgb'], list_features=[style_features_normalized_new, char_n_grams_normalized_new],
                                 train_size=5000, index_file='index_files/index_file.csv', num_iterations=10,
                                Gender_Group1=['Male'], Gender_Group2=['Female'],Gender_Group3=['Trans'],
                                 no_overlapping_authors=True):
    """
    Varies classifier and feature set used; at the moment works for english text with native language == en
    There is a failsafe that the maximum iterations is not larger than the max_no_of_authors/iterations > 1
    This iteration is specifically for Gender analysis
    :param dir_with_data: folder containing data_for_genAI_bias
    :param mode: halfhalf: half natives, half non-natives
    :param num_authors: how many classes  in setup
    :param dir_results: where to save results
    :param list_classifiers: list of classifiers to use
    :param list_features: list of functions to extract features
    :param train_size: default 6000 words
    :param test_size: default 3000 words
    :param chunk_size: 600 +-25
    :param index_file: file containing information about flair
    :param num_iterations: how many times the experiment is run with different authors
    :param no_overlapping_authors, wheter overlapp of authors is allowed
    :return: None (writes results to files)
    """
    df_index = pd.read_csv(index_file)[['author_name', 'Gender']]#Load index file, but only take author_name and Gender
    df_index.drop_duplicates(inplace=True) #so merge doesn't create too many rows =)
    #load data
    df_train = pd.read_csv(os.path.join(dir_with_data, 'training_data_good.csv'))
    df_test = pd.read_csv(os.path.join(dir_with_data, 'testing_data_good_10.csv'))
    print(df_train.columns, df_test.columns)
    df_train = df_train.merge(df_index, left_on='author', right_on='author_name')  # add in Gender
    df_test = df_test.merge(df_index, left_on='author', right_on='author_name')  # add in Gender
    df_train.dropna(subset=['pretreated_body'], inplace=True)
    df_test.dropna(subset=['pretreated_body'], inplace=True)
    all_train_ids_G1 = set(df_train[df_train['Gender'].isin(set(Gender_Group1))].author_name.unique())  # keep track of viable authors for class N
    all_train_ids_G2 = set(df_train[df_train['Gender'].isin(set(Gender_Group2))].author_name.unique())   # keep track of viable authors for class notN
    all_test_other = set(df_train[df_train['Gender'].isin(set(Gender_Group3))].author_name.unique())
    if mode=='halfhalf': #half native - half non_native
        G1_list = [int(num_authors / 2)]
        G2_list = [int(num_authors / 2)]
    elif mode=='substitute':
        G1_list = [i for i in range(0,num_authors+1)]
        G2_list = [num_authors-i for i in range(0,num_authors+1)]
    else:
        raise ValueError
    for G1, G2 in zip(G1_list, G2_list):
        all_train_ids_G2_temp = all_train_ids_G2
        all_train_ids_G1_temp = all_train_ids_G1
        all_ids = all_train_ids_G1.union(all_train_ids_G2)
        for i in tqdm(range(num_iterations), desc=f'Running Iterations...'):
            if G1>0:
                these_ids_G1 = set(random.sample(list(all_train_ids_G1_temp), G1))  # reduce authors
            elif G1==0:
                these_ids_G1 = set()
            if G2>0:
                these_ids_G2 = set(random.sample(list(all_train_ids_G2_temp), G2))  # reduce authors
            elif G2==0:
                    these_ids_G2 = set()
            these_ids = these_ids_G1.union(these_ids_G2)# combine authors
            other_ids_G1 = all_train_ids_G1.difference(these_ids)
            other_ids_G1_picked = set(random.sample(list(other_ids_G1), 10))
            other_ids_G2 = all_train_ids_G2.difference(these_ids)
            other_ids_G2_picked = set(random.sample(list(other_ids_G2), 10))
            other_ids_G3_picked = set(random.sample(list(all_test_other), 10))
            other_ids = other_ids_G1_picked.union(other_ids_G2_picked)
            other_ids = other_ids.union(other_ids_G3_picked)

            assert these_ids.intersection(other_ids)==set()
            df_train_subset = df_train[df_train['author'].isin(these_ids)]
            df_test_subset = df_test[df_test['author'].isin(these_ids)]
            df_test_subset_lo = df_test[df_test['author'].isin(other_ids)]
            counter = 0
            df_test_subset_lo.dropna(subset=['pretreated_body'],inplace=True)
            df_test_subset.dropna(subset=['pretreated_body'],inplace=True)
            df_train_subset.dropna(subset=['pretreated_body'],inplace=True)
            assert len(df_train_subset.author.unique())== num_authors
            assert len(df_test_subset.author.unique())== num_authors
            df_train_subset.reset_index(inplace=True)
            df_test_subset_lo.reset_index(inplace=True)
            df_test_subset.dropna(subset=['pretreated_body'],inplace=True)
            df_test_subset_lo.dropna(subset=['pretreated_body'],inplace=True)
            while df_train_subset['word_count_pb'].sum() < train_size \
                    or len(df_train_subset.author.unique()) != num_authors \
                    or df_test_subset_lo['pretreated_body'].isnull().values.any() \
                    or df_test_subset['pretreated_body'].isnull().values.any()\
                    or df_train_subset['pretreated_body'].isnull().values.any():  # Make sure everything is ok # while there are any empty entries in the df redo selection
                if G1>0:
                    these_ids_G1 = set(random.sample(list(all_train_ids_G1_temp), G1))# reduce authors
                elif G1==0:
                    these_ids_G1 = set()
                if G2>0:
                    these_ids_G2 = set(random.sample(list(all_train_ids_G2_temp), G2))  # reduce authors
                elif G2==0:
                    these_ids_G2 = set()
                these_ids = these_ids_G1.union(these_ids_G2)# combine authors
                other_ids_G1 = all_train_ids_G1.difference(these_ids)
                other_ids_G1_picked = set(random.sample(list(other_ids_G1), 10))
                other_ids_G2 = all_train_ids_G2.difference(these_ids)
                other_ids_G2_picked = set(random.sample(list(other_ids_G2), 10))
                other_ids_G3_picked = set(random.sample(list(all_test_other), 10))
                other_ids = other_ids_G1_picked.union(other_ids_G2_picked)
                other_ids = other_ids.union(other_ids_G3_picked)
                assert these_ids.intersection(other_ids)==set()
                df_train_subset = df_train[df_train['author'].isin(these_ids)]
                df_test_subset = df_test[df_test['author'].isin(these_ids)]
                df_test_subset_lo = df_test[df_test['author'].isin(other_ids)]
                counter = 0
                df_test_subset_lo.dropna(subset=['pretreated_body'],inplace=True)
                df_test_subset.dropna(subset=['pretreated_body'],inplace=True)
                df_train_subset.dropna(subset=['pretreated_body'],inplace=True)
                assert len(df_train_subset.author.unique())== num_authors
                assert len(df_test_subset.author.unique())== num_authors
                df_train_subset.reset_index(inplace=True)
                df_test_subset_lo.reset_index(inplace=True)
                df_test_subset.reset_index(inplace=True)
                counter += 1
                if counter >= 200:
                    print("Empty authors in selection")
                    raise ValueError
            for feature_function in list_features:
                print(feature_function.__name__)
                df_train_subset_copy = df_train_subset[df_train_subset['pretreated_body'].notna()]
                df_test_subset_copy = df_test_subset[df_test_subset['pretreated_body'].notna()]
                df_test_subset_copy_lo = df_test_subset_lo[df_test_subset_lo['pretreated_body'].notna()]#leave_out
                X_train, X_test = feature_function(df_train_subset_copy, df_test_subset_copy, 'pretreated_body')
                X_train_null, X_test_lo = feature_function(df_train_subset_copy, df_test_subset_copy_lo, 'pretreated_body')
                y_test = df_test_subset_copy['author']
                y_train = df_train_subset_copy['author']
                y_test_lo = df_test_subset_copy_lo['author']
                if X_train.shape[0]==y_train.shape[0] and X_test.shape[0]==y_test.shape[0]:
                    X_train.fillna(0, inplace=True)
                    X_test.fillna(0, inplace=True)
                    X_test_lo.fillna(0, inplace=True)
                    X_train = pd.DataFrame(scale(X_train))
                    X_test = pd.DataFrame(scale(X_test))
                    X_test_lo = pd.DataFrame(scale(X_test_lo))
                    for classifier in list_classifiers:
                        print(classifier)
                        accuracy, f1, y_pred, report, y_pred_proba, labels, y_pred_lo, y_pred_proba_lo = run_classify_combined(X_train, y_train, X_test, y_test, X_test_lo, feature_names=feature_function.__name__, feature_importance=False, ctype=classifier, CLASSES=no_auth)
                        # Write_nodes
                        print(accuracy)
                        G1_n =''.join(Gender_Group1)
                        G2_n =''.join(Gender_Group2)
                        with open(os.path.join(dir_results,"leave_out",str(num_authors),                     f'edges_iteration{i}_mode{mode}_noauthors{num_authors}_{G1_n}authors{G1}_{G2_n}authors{G2}_unchunked_trainsize{train_size}_classifier{classifier}_{feature_function.__name__}.csv'),'w') as file:
                            file.write("true,predicted\n")
                            for author_true, author_pred in zip(y_test_lo, y_pred_lo):
                                file.write(f"{author_true},{author_pred}\n")
                            file.close()
                        with open(os.path.join(dir_results,"normal",str(num_authors),
                                               f'edges_iteration{i}_mode{mode}_noauthors{num_authors}_{G1_n}authors{G1}_{G2_n}authors{G2}_unchunked_trainsize{train_size}_classifier{classifier}_{feature_function.__name__}.csv'),
                                  'w') as file:
                            file.write("true,predicted\n")
                            for author_true, author_pred in zip(y_test, y_pred):
                                file.write(f"{author_true},{author_pred}\n")
                            file.close()
                            print(os.path.join(dir_results,"normal",str(num_authors),
                                               f'edges_iteration{i}_mode{mode}_noauthors{num_authors}_{G1_n}authors{G1}_{G2_n}authors{G2}_unchunked_trainsize{train_size}_classifier{classifier}_{feature_function.__name__}.csv'))
                            # write rankings
                        df_pred = pd.DataFrame(y_pred_proba)
                        df_pred.reset_index(inplace=True)
                        df_y_test = pd.Series(y_test)
                        authors = sorted(list(these_ids))
                        dict_columns = dict([(i, authors[i]) for i in range(len(authors))])
                        df_pred.rename(columns=dict_columns, inplace=True)
                        df_pred = pd.concat([df_y_test, df_pred], axis=1)
                        df_pred.to_csv(os.path.join(dir_results,"normal",str(num_authors),
                                                        f'rankings_iteration{i}_mode{mode}_noauthors{num_authors}_{G1_n}authors{G1}_{G2_n}authors{G2}_unchunked_trainsize{train_size}_classifier{classifier}_{feature_function.__name__}.csv'),
                                           index=None)
                        df_pred = pd.DataFrame(y_pred_proba_lo)
                        df_pred.reset_index(inplace=True)
                        df_y_test = pd.Series(y_test_lo)
                        authors = sorted(list(these_ids))
                        dict_columns = dict([(i, authors[i]) for i in range(len(authors))])
                        df_pred.rename(columns=dict_columns, inplace=True)
                        df_pred = pd.concat([df_y_test, df_pred], axis=1)
                        df_pred.to_csv(os.path.join(dir_results,"leave_out",str(num_authors),
                                                        f'rankings_iteration{i}_mode{mode}_noauthors{num_authors}_{G1_n}authors{G1}_{G2_n}authors{G2}_unchunked_trainsize{train_size}_classifier{classifier}_{feature_function.__name__}.csv'),
                                           index=None)

            if no_overlapping_authors:  # if we don't want overlapping authors we need to take away the used authors
                    all_train_ids_G1_temp = all_train_ids_G1_temp.difference(these_ids_G1)
                    all_train_ids_G2_temp = all_train_ids_G2_temp.difference(these_ids_G2)
    return None


def run_classify_combined(X_train, y_train, X_test, y_test, X_test_leave_out, feature_names,filename_features="",  feature_importance=True, ctype='svm', CLASSES=8):
    counter = 0
    if ctype == 'xgb':
       classifier = xgb.XGBClassifier(objective='multi:softmax', random_state=42, num_class=CLASSES)
    elif ctype == 'rf':
        classifier = RandomForestClassifier(n_estimators=500, n_jobs=multiprocessing.cpu_count())
    elif ctype == 'logR':
        classifier= LogisticRegression(random_state=42)
    else:
        classifier = SVC(probability=True)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred_lo = classifier.predict(X_test_leave_out)
    y_pred_proba_lo = classifier.predict_proba(X_test_leave_out)
    labels = classifier.classes_
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    #print(report)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    if feature_importance:
        if ctype == 'xgb':
            importances = classifier.feature_importances_
            df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False)
            df.to_csv(filename_features, index=False)
        elif ctype == 'rf':
            importances = classifier.feature_importances_
            df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False)
            df.to_csv(filename_features, index=False)
        elif ctype == 'logR':
            feature_names = X_train.columns.tolist()
            r = permutation_importance(classifier, X_train, y_train, n_repeats=10, random_state=42)
            res=[]
            for i in r.importances_mean.argsort()[::-1]:
                res.append({"feature":f"{feature_names[i]}", "mean_importance": f"{r.importances_mean[i]:.3f}",
                            "std_dev":f" +/- {r.importances_std[i]:.3f}"})

            pd.DataFrame.from_dict(res).to_csv(filename_features, index=False)
        else:
            feature_names = X_train.columns.tolist()
            r = permutation_importance(classifier, X_train, y_train, n_repeats=10, random_state=42)
            res=[]
            for i in r.importances_mean.argsort()[::-1]:
                res.append({"feature":f"{feature_names[i]}", "mean_importance": f"{r.importances_mean[i]:.3f}",
                            "std_dev":f" +/- {r.importances_std[i]:.3f}"})


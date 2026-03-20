from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
#Library imports
import shutil
import os
import re

import numpy as np
from iteration_utilities import intersperse
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.common import random_state
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
from sklearn.inspection import permutation_importance
import string
from iteration_utilities import intersperse
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statistics
import datetime
print(datetime.datetime.now())
a = datetime.datetime.now()
print(datetime.datetime.now()-a)
import spacy
#spacy.cli.download('de_core_news_sm')
spacy.cli.download("en_core_web_sm")
from collections import Counter
#import multiprocessing
pos_lang_trad = {'en':'english', 'de':'german', 'nl':'dutch', 'fr':'french', 'pt':'portuguese', 'it':'italian', 'es':'spanish', 'tr':'turkish'}


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x


    return (b_0, b_1)

def evaluate_regression(x,y,b):
    m_y = np.mean(y)
    n = np.size(x)
    error = y - (b[0] + b[1]*x)
    se = np.sum(error ** 2)
    mse = se/n
    SSt = np.sum((y - m_y) ** 2)
    R2 = 1 - (se / SSt)
    return mse, R2

def plot_regression_line(x, y, b, author, savedir, value_of_interest, label_value_of_interest, c,f):
    # Define the y-axis ranges
    #top_ylim = (0.4, 1.0)
    #bottom_ylim = (0.0, 0.1)

    # Compute height ratios to match visual y-scale
    #top_range = top_ylim[1] - top_ylim[0]  # = 0.6
    #bottom_range = bottom_ylim[1] - bottom_ylim[0]  # = 0.1

    # The height ratio should match the actual data range
    #height_ratios = [top_range, bottom_range]

    # Set up the figure with gridspec
    fig = plt.figure(figsize=(6, 4))
    #gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=0.1)

    #ax1 = fig.add_subplot(gs[0])
    #ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Scatter plots
    sns.scatterplot(x=x, y=y, hue=author,legend=False)
    #ax2.scatter(x, y, color=(0.0, 0.0, 1.0, 0.2), marker="o")

    # Regression line
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color="b")
    #ax2.plot(x, y_pred, color="b")

    # Set y-limits
    #ax1.set_ylim(top_ylim)
    #ax2.set_ylim(bottom_ylim)

    # Y-ticks (optional: align tick intervals)
    #ax1.set_yticks(np.arange(0.4, 1.01, 0.1))
    #ax2.set_yticks(np.arange(0.0, 0.11, 0.1))

    # Hide spines between subplots
    #ax1.spines.bottom.set_visible(False)
    #ax2.spines.top.set_visible(False)
    #ax1.xaxis.tick_top()
    #ax1.tick_params(labeltop=False)
    #ax2.xaxis.tick_bottom()

    # Draw break lines
    #d = 0.1  # adjust for vertical scaling
    #kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
    #              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    #ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    #ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # Labels and ticks
    plt.xlabel('Birthyear')
    plt.xticks(rotation=90)
    #plt.ylabel()
    plt.ylabel(label_value_of_interest)
    #plt.xticks(range(x.unique().max() + 1))

    # Title at top of figure
    fig.suptitle(f"Effect of Year Of Birth on {value_of_interest}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f'{value_of_interest}_{c}_{f}_regplot.jpg'))
    plt.clf()
    return None


def one_and_done_regression(value_of_interest, label_value_of_interest, file_origin, save_dir, save_file,
                            classifier_of_interest=['logR'], feature_of_interest=['style_features_normalized']):
    df = pd.read_csv(file_origin)
    df = df[df["yob"]>1950]
    df = df[df["yob"]<2008]
    #df['treshhold']=df['treshhold'].fillna('None')
    for feature in feature_of_interest:
        df_temp = df[df['Features'] == feature]
        for classifier in classifier_of_interest:
            df_temp_1 = df_temp[df_temp['classifier']==classifier]
            #for th in df_temp_1.treshhold.unique():
            #    df_temp_2 = df_temp_1[df_temp_1['treshhold']==th]
            x, y, author = df_temp_1["yob"], df_temp_1["P(correct)"], df_temp_1["author"]
            b = estimate_coef(x, y)
            model = LinearRegression()
            x_special = x.to_numpy(dtype=float).reshape(-1, 1)
            linear_compare = model.fit(x_special,y.to_numpy(dtype=float))
            slope = linear_compare.coef_
            intercept = linear_compare.intercept_
            y_predicted = model.predict(x_special)
            print(slope,intercept)
            mse, R2 = evaluate_regression(x, y, b)
            mse2, R22 = mean_squared_error(y,y_predicted), r2_score(y, y_predicted)
            assert (mse-mse2)**2<0.00001
            print(R2, R22)
            assert (R2- R22)**2<0.00001

            plot_regression_line(x, y, b, author, save_dir, value_of_interest, label_value_of_interest, classifier, feature)
            with open(save_file, 'a') as file:
                #file_origin,measured_var,classifier,feature,b,m,mse,R2
                file.writelines(f'"{file_origin}",{label_value_of_interest},{classifier},{feature},{intercept},{slope},{mse},{R2}\n')


    return None

if __name__ = "__main__":
  index = r"user_ids_to_collect_with_flair.csv"
  for i in [4,8,16]:
      results = rf"Experiment_Regression\{i}"
      map = rf"analysis_information.csv"
      save_res = rf"Experiment_Regression\{i}\result"
  
      accuracy_and_F1_out_of_edgefile(results, os.path.join(save_res, "Accuracy_F1.csv"))
      plt.clf()
      df = pd.read_csv(os.path.join(save_res, "Accuracy_F1.csv"))
      sns.boxplot(df, x="classifier", y="Accuracy", hue="Features")
      plt.savefig(os.path.join(save_res, "Accuracy.png"))
      plt.clf()
      sns.boxplot(df, x="classifier", y="F1", hue="Features")
      plt.savefig(os.path.join(save_res, "F1.png"))
      plt.clf
          #df_res = probability_correct(map,results, index, save_res)
      one_and_done_regression("P(correct)","P(correct)", os.path.join(save_res, "P_correct.csv"),
                                  save_res, os.path.join(save_res, "regression_info.csv"))
      one_and_done_regression("P(correct)", "P(correct)", os.path.join(save_res, "P_correct.csv"),
                                  save_res, os.path.join(save_res, "regression_info.csv"), feature_of_interest=["char_n_grams_normalized"])


import matplotlib.pyplot as plt
import numpy as np

#from new_feature_function import *
from feature_extraction import *
import os
import random
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
#from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

def apply_feature_fuction(id, author_list, dir_res, result_file, train_data, test_data, column_text, save_loc):
    train_data = train_data[train_data["author"].isin(set(author_list))]
    test_data = test_data[test_data["author"].isin(set(author_list))]
    train_data.dropna(subset=[column_text], inplace=True)
    test_data.dropna(subset=[column_text], inplace=True)
    for feature_function in [style_features_normalized]:
        df_train_subset_copy = train_data[train_data[column_text].notna()]
        df_test_subset_copy = test_data[test_data[column_text].notna()]
        df_res = pd.read_csv(os.path.join(dir_res,result_file))
        df_res["type"] = df_res["true"]==df_res["predicted"]
        X_train, X_test = feature_function(df_train_subset_copy, df_test_subset_copy,
                                           column_text)
        y_test = df_test_subset_copy['author']
        y_train = df_train_subset_copy['author']
        X_test.dropna(inplace=True)
        X_train.dropna(inplace=True)
        if X_train.shape[0] == y_train.shape[0] and X_test.shape[0] == y_test.shape[0]:
            X_train.fillna(0, inplace=True)
            X_test.fillna(0, inplace=True)
            if feature_function == style_features_normalized:
                cols_train = set(X_train.columns)
                X_train = X_train.select_dtypes(exclude=['object'])
                print(cols_train.difference(set(X_train.columns)))
                cols_test = set(X_test.columns)
                X_test = X_test.select_dtypes(exclude=['object'])
                print(cols_test.difference(set(X_test.columns)))
            X_train = pd.DataFrame(scale(X_train))
            X_test = pd.DataFrame(scale(X_test))
            X_train.to_csv(save_loc + fr'\id_{id}_training_{feature_function.__name__}.csv', index=False)
            X_test.to_csv(save_loc + fr'\id_{id}_training_{feature_function.__name__}.csv', index=False)
            feature_cols = X_test.columns
            feature_cols_train = X_train.columns
            print(len(feature_cols), len(feature_cols_train))
            y_test=y_test.reset_index()
            df_res.reset_index(inplace=True)
            X_test.reset_index(inplace=True)
            print(y_test.shape, df_res.shape, X_test.shape)
            X_test = pd.concat([y_test, df_res, X_test], axis=1)
            print(X_test.head())
            X_train.reset_index(inplace=True)
            y_train=y_train.reset_index()
            print(y_train.shape,X_train.shape)
            X_train = pd.concat([y_train,X_train], axis=1)
            print(X_train.head())
            #X_train.dropna(inplace=True)
            similarity_list = []
            for author in author_list:
                X_temp_t = X_train[X_train["author"] == author]
                X_temp_t = X_temp_t[feature_cols_train]

                print(X_temp_t.shape)
                X_temp_t.dropna(inplace=True)
                print(X_temp_t.shape)
                row2 = X_temp_t.mean().values.reshape(1, -1)
                if np.nan in row2:
                    pass
                else:
                #average_train[author] =
                    X_temp = X_test[X_test.true == author]
                    y_test_temp = X_temp["predicted"]# Only take the ones where true is authors
                    X_temp = X_temp[feature_cols]
                    for i in range(len(X_temp)):
                        row1 = X_temp.iloc[i, :].values.reshape(1, -1)#each line
                        #aver
                        predicted = y_test_temp.iloc[i]
                        print(row1)
                        print(row2)
                        try:
                            similarity = cosine_similarity(row1, row2)[0][0]
                            similarity_list.append({"true": author, "predcted":predicted,"cosine": similarity, "average":X_temp_t.mean().values.reshape(1, -1)})
                        except:
                            print(row2)
            df = pd.DataFrame.from_dict(similarity_list)
            df.to_csv(os.path.join(save_loc,f"similarity_scores_{feature_function.__name__}_{id}.csv"))
    return None

def visualize_cosine(file):
    df = pd.read_csv(file)
    df["correct"]=df["true"]==df["predcted"]
    le = LabelEncoder()
    df["author"] = le.fit_transform(df["true"])
    sns.boxplot(df, x="author", y="cosine", hue="correct")

    plt.ylim(-1,1)
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity; NL")
    plt.savefig(file[:-4] + ".png")
    plt.clf()




if __name__ == "__main__":
    train_data = fr"path\training_data_good.csv"
    df_train = pd.read_csv(train_data)
    test_data = fr"path\testing_data_good_10.csv"
    df_test = pd.read_csv(test_data)

    #files = random.sample(files, 10)
    res = []
    for gender_combo in [ "GenXGenY"]:
        for author in [8,16]:
            save=fr"path\closed_world\result"
            classifications=fr"path\closed_world"
            files = os.listdir(classifications)
            files = [f for f in files if "classifierlogR" in f]
            files = [f for f in files if "style_features_normalized_new" in f]
            files = [f for f in files if "edges" in f]
            files = random.sample(files, 30)
            for f in files:
                authors = list(pd.read_csv(os.path.join(classifications, f))["true"].unique())
                res.append({"authors": authors, "file_res":f, "id":hash(str(sorted(authors)))})
            pd.DataFrame.from_dict(res).to_csv(os.path.join(save,"analysisiinfo.csv"), index=False)
            for experiment in res:
                apply_feature_fuction(experiment["id"], experiment["authors"],classifications, experiment["file_res"],
                                     df_train, df_test, "pretreated_body", save)
            files = os.listdir(save)
            files = [f for f in files if "similarity_scores" in f]
            for f in files:
                try:
                    visualize_cosine(os.path.join(save, f))
                except:
                    pass







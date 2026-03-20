
def run_closed_world(folder_train_test, suspect_set_size, no_repetitions_per_author, index, no_authors_per_year, save_folder, min_year=1950, max_year=2008,bad_authors=bad, classifiers=[], features=[]):
    df_train = pd.read_csv(os.path.join(folder_train_test, "training_data_good.csv"))
    df_test = pd.read_csv(os.path.join(folder_train_test, "testing_data_good_10.csv"))
    df_train = df_train[df_train['pretreated_body'].notna()]
    df_test = df_test[df_test['pretreated_body'].notna()]
    set_authors = set(df_train.author.unique()).intersection(set(df_test.author.unique()))
    df_index = pd.read_csv(index)
    df_index = parse_yob(df_index)
    df_index = df_index.dropna(subset=['yob'])
    range_of_interest = [str(year) for year in range(min_year, max_year)]
    df_index = df_index[df_index['yob'].isin(set(range_of_interest))]
    index_a = set(df_index.author_name.unique())
    set_authors = set_authors.intersection(index_a)
    set_authors = set_authors.difference(set(bad_authors))
    df_index = df_index[df_index['author_name'].isin(set_authors)]
    candidate_authors = set()
    for yob in df_index.yob.unique():
        df_temp = df_index[df_index.yob == yob]
        a_temp = list(set(df_temp.author_name.unique()))
        if len(a_temp)>no_authors_per_year:
            a_temp = set(random.sample(a_temp, no_authors_per_year))
        candidate_authors=candidate_authors.union(set(a_temp))
    df = construct_suspect_sets_regression(candidate_authors=candidate_authors,all_authors=set_authors,
                                           min_no_of_sets= no_repetitions_per_author,
                                           size_set=suspect_set_size)
    #(candidate_authors: list, all_authors:list,min_no_of_sets: int, size_set: int
    df.to_csv(os.path.join(save_folder, 'misc', 'analysis_information.csv'))
    #print(df.columns)
    for suspect_setty in tqdm(list(df.str_ss.unique())):
        suspect_set = eval(suspect_setty)
        assert hash(suspect_setty) in set(df["hash"].unique())
        if len(set(suspect_set))==suspect_set_size:
            #print(len(set(suspect_set)))
            df_train_temp = df_train[df_train['author'].isin(set(suspect_set))]
            df_test_temp = df_test[df_test['author'].isin(set(suspect_set))]
            authors = df_train_temp['author'].unique()
            authors_t = df_test_temp['author'].unique()
            if len(authors)==len(authors_t)==suspect_set_size:
                for classifier in classifiers:
                    for feature_function in features:
                        #print(feature_function.__name__)
                        df_train_subset_copy = df_train_temp[df_train_temp['pretreated_body'].notna()]
                        df_test_subset_copy = df_test_temp[df_test_temp['pretreated_body'].notna()]
                        X_train, X_test = feature_function(df_train_subset_copy, df_test_subset_copy, 'pretreated_body')
                        y_test = df_test_subset_copy['author']
                        y_train = df_train_subset_copy['author']
                        X_train.dropna(inplace=True)
                        X_test.dropna(inplace=True)
                        #print(set(y_train.unique()) == set(y_test.unique()))
                        #print(len(set(y_train.unique()))==suspect_set_size)
                        if X_train.shape[0]==y_train.shape[0] and X_test.shape[0]==y_test.shape[0]:
                            X_train.fillna(0, inplace=True)
                            X_test.fillna(0, inplace=True)
                            X_train = pd.DataFrame(scale(X_train))
                            X_test = pd.DataFrame(scale(X_test))
                            le = LabelEncoder()  # nEEDED FOR XGB
                            y_train = le.fit_transform(y_train)
                            y_test = le.transform(y_test)
                            acc, f1, y_pred, report, y_pred_proba, labels= run_classify(X_train, y_train, X_test, y_test, f"Nope", False, ctype=classifier)
                            y_test = le.inverse_transform(y_test)
                            y_pred = le.inverse_transform(y_pred)
                            #assert set(y_train.author.unique()) == set(y_test.author.unique())
                            #print(len(set(y_train.author.unique())))
                            #assert len(set(y_train.author.unique())) == suspect_set_size
                            y_t = pd.concat([pd.Series(y_test), df_test_subset_copy['created_utc']], axis=1)
                            with open(os.path.join(save_folder,
        f'edges_{hash(suspect_setty)}_unchunked_classifier{classifier}_{feature_function.__name__}.csv'), 'w') as file:
                                file.write("true,predicted\n")
                                for author_true, author_pred in zip(y_test, y_pred):
                                    file.write(f"{author_true},{author_pred}\n")
                                file.close()
                                # write rankings
                            df_pred = pd.DataFrame(y_pred_proba)
                            authors = sorted(suspect_set)
                            dict_columns = dict([(i, authors[i]) for i in range(len(authors))])
                            df_pred.rename(columns=dict_columns, inplace=True)
                            df_pred = pd.concat([pd.Series(y_test), df_pred], axis=1)
                            df_pred.to_csv(os.path.join(save_folder,
                                                            f'ranking_{hash(suspect_setty)}_unchunked_classifier{classifier}_{feature_function.__name__}.csv'),
                                               index=None)
                            with open(os.path.join(save_folder,
        f'times_{hash(suspect_setty)}_unchunked_classifier{classifier}_{feature_function.__name__}.csv'), 'w') as file:
                                file.write("y_test,utc\n")
                                for author_true, author_pred in zip(y_test, df_test_subset_copy['created_utc']):
                                    file.write(f"{author_true},{author_pred}\n")
                                file.close()

                        else:
                            with open(os.path.join(save_folder,'misc','log_error.txt'), 'a') as file:
                                file.writelines(str(hash(suspect_setty))+',EmptyEntries'+'\n')
    return None


def run_classify(X_train, y_train, X_test, y_test,filename_features, feature_importance=True, ctype='svm', CLASSES=8):
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
    labels = classifier.classes_
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    #print(report)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    if feature_importance:
        if ctype == 'xgb':
            importances = classifier.feature_importances_
            feature_names = X_train.columns.tolist()
            df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False)
            df.to_csv(filename_features, index=False)
        elif ctype == 'rf':
            importances = classifier.feature_importances_
            feature_names = X_train.columns.tolist()
            df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False)
            df.to_csv(filename_features, index=False)
        elif ctype == 'logR':
            result = permutation_importance(classifier, X_train, y_train, n_repeats=10)
            perm_sorted_idx = result.importances_mean.argsort()
            feature_name = X_train.columns[perm_sorted_idx]
            res=[]
            for a,b in zip(result.importances_mean, feature_name):
                res.append({'Feature': b, 'Mean_Importance': a})
            pd.DataFrame.from_dict(res).to_csv(filename_features, index=False)
        else:
            # Compute permutation importance
            result = permutation_importance(classifier, X_train, y_train, n_repeats=10)
            perm_sorted_idx = result.importances_mean.argsort()
            feature_name = X_train.columns[perm_sorted_idx]
            res=[]
            for a,b in zip(result.importances_mean, feature_name):
                res.append({'Feature': b, 'Mean_Importance': a})
            pd.DataFrame.from_dict(res).to_csv(filename_features, index=False)
    return accuracy, f1, y_pred, report, y_pred_proba, labels



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


def run_leave_out(train_test_location,save_location,no_of_experiments,suspect_set_size,year_range,classifiers=[], features=[],no_authors_to_test =10):
    test = os.path.join(train_test_location,"testing_data_good_10.csv")
    train = os.path.join(train_test_location,"training_data_good.csv")
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)
    df_train = df_train[df_train['pretreated_body'].notna()]
    df_test = df_test[df_test['pretreated_body'].notna()]
    good_authors = set(df_train.author.unique()).intersection(df_test["author"].unique())
    suspect_set_for_leave_out(index, good_authors, no_of_experiments, suspect_set_size, year_range, os.path.join(save_location, "result", f'mapping_{no_of_experiments}_{suspect_set_size}.csv'))
    mapping = pd.read_csv(os.path.join(save_location, "result", f'mapping_{no_of_experiments}_{suspect_set_size}.csv'))
    for row in tqdm(mapping.iterrows()):
        ha = row[1]["hash"]
        suspect_set = set(eval(row[1]["authors"]))
        test_authors = set(eval(row[1]["test_authors"]))
        test_authors = test_authors.difference(suspect_set)
        test_authors = set(random.sample(list(test_authors), no_authors_to_test))
        assert len(suspect_set)==suspect_set_size
        if len(set(suspect_set)) == suspect_set_size:
            # print(len(set(suspect_set)))
            df_train_temp = df_train[df_train['author'].isin(set(suspect_set))]
            df_test_temp = df_test[df_test['author'].isin(set(test_authors))]
            for classifier in classifiers:
                for feature_function in features:
                    # print(feature_function.__name__)
                    df_train_subset_copy = df_train_temp[df_train_temp['pretreated_body'].notna()]
                    df_test_subset_copy = df_test_temp[df_test_temp['pretreated_body'].notna()]
                    X_train, X_test = feature_function(df_train_subset_copy, df_test_subset_copy, 'pretreated_body')
                    y_test = df_test_subset_copy['author']
                    y_train = df_train_subset_copy['author']
                    X_train.dropna(inplace=True)
                    X_test.dropna(inplace=True)
                    # print(set(y_train.unique()) == set(y_test.unique()))
                    # print(len(set(y_train.unique()))==suspect_set_size)
                    if X_train.shape[0] == y_train.shape[0] and X_test.shape[0] == y_test.shape[0]:
                        X_train.fillna(0, inplace=True)
                        X_test.fillna(0, inplace=True)
                        X_train = pd.DataFrame(scale(X_train))
                        X_test = pd.DataFrame(scale(X_test))
                        le = LabelEncoder()  # nEEDED FOR XGB
                        y_train = le.fit_transform(y_train)
                        #y_test = le.transform(y_test)
                        y_pred, y_pred_proba, labels = run_classify_leave_out(X_train, y_train, X_test,
                                                                                      ctype=classifier)
                        y_pred = le.inverse_transform(y_pred)
                        with open(os.path.join(save_location,
                                               f'edges_{ha}_unchunked_classifier{classifier}_{feature_function.__name__}.csv'),
                                  'w') as file:
                            file.write("true,predicted\n")
                            for author_true, author_pred in zip(y_test, y_pred):
                                file.write(f"{author_true},{author_pred}\n")
                            file.close()
                            # write rankings
                        df_pred = pd.DataFrame(y_pred_proba)
                        authors = sorted(suspect_set)
                        dict_columns = dict([(i, authors[i]) for i in range(len(authors))])
                        df_pred.rename(columns=dict_columns, inplace=True)
                        df_pred = pd.concat([pd.Series(y_test), df_pred], axis=1)
                        df_pred.to_csv(os.path.join(save_location,
                                                    f'ranking_{hash(ha)}_unchunked_classifier{classifier}_{feature_function.__name__}.csv'),
                                       index=None)
                        with open(os.path.join(save_location,
                                               f'times_{hash(ha)}_unchunked_classifier{classifier}_{feature_function.__name__}.csv'),
                                  'w') as file:
                            file.write("y_test,utc\n")
                            for author_true, author_pred in zip(y_test, df_test_subset_copy['created_utc']):
                                file.write(f"{author_true},{author_pred}\n")
                            file.close()

                    else:
                        with open(os.path.join(save_location, 'misc', 'log_error.txt'), 'a') as file:
                            file.writelines(str(ha) + ',EmptyEntries' + '\n')
    return None

def suspect_set_for_leave_out(index, good_authors,no_of_experiments, suspect_set_size,year_range, save_file):
    """
    Conditions: 1. authors in suspect set must have different yob
                2. authors to test must share yob with authors in suspect set (+/-3 years)
    :param index: index file
    :param good_authors: set of authors in both train and test
    :param no_of_experiments: how many runs I want to have
    :return:
    """
    df = pd.read_csv(index)
    df = parse_yob(df)
    df = df[["author_name", "yob"]]
    df = df[df["author_name"].isin(good_authors)]
    dict_yob = dict()
    for index, row in df.iterrows():
        dict_yob[row["author_name"]] = row["yob"]
    res =[]
    hashes = set()
    suspect_set_pool = set(df.author_name.unique())
    while len(res)<no_of_experiments or len(suspect_set_pool)<suspect_set_size:
        df_temp = df.copy()
        all_authors = suspect_set_pool
        authors = set()
        for i in range(suspect_set_size):
            a = random.sample(list(all_authors), 1)[0]
            authors.add(a)
            df_temp = df_temp[df_temp["yob"]!=dict_yob[a]]
            all_authors = set(df_temp.author_name.unique())
        assert len(set(authors))==suspect_set_size
        y_o_b_interest = []
        for author in authors:
            y_o_b_interest.extend(range(int(dict_yob[author])-year_range,int(dict_yob[author])+year_range+1))
        y_o_b_interest = set([str(y) for y in y_o_b_interest])
        test_authors = list(set(df[df["yob"].isin(y_o_b_interest)].author_name.unique()))
        if hash(str(sorted(list(authors)))) not in hashes:
            res.append({"hash":hash(str(sorted(list(authors)))),
                    "authors":sorted(list(authors)),
                    "test_authors":test_authors}
                   )
            hashes.add(hash(str(sorted(list(authors)))))
            suspect_set_pool = suspect_set_pool.difference(authors)
    df = pd.DataFrame.from_dict(res)
    df.to_csv(save_file, index=False)
    return None

import datetime
import os
import re
import pandas as pd
import string
from iteration_utilities import intersperse
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
import shutil
#Global variables
punkt_translation = {'nl': 'dutch', 'en': 'english', 'fr': 'french', 'de':'german', 'it':'italian', 'pt': 'portuguese', 'es':'spanish', 'tr': 'turkish'}
# def index_merge(folder_old_index_file='index_files/old_indexfiles', folder_out_put='index_files', new_file_name='index_file.csv'):
#     """Function to manage index files"""
#     filenames = os.listdir(folder_old_index_file)
#     df = pd.DataFrame()
#     for filename in filenames:
#         df_1 = pd.read_csv(os.path.join(folder_old_index_file, filename))
#         df = pd.concat([df, df_1])
#         print(df.shape)
#     df.to_csv(os.path.join(folder_out_put, new_file_name), index=False)
#     return None
# #New authors to pipeline
# def new_authors(treated_authors_folder='0_users_comments', new_authors='0_users_comments1'):
#     """Function to create set of authors to pretreat"""
#     filenames_old = set(os.listdir(treated_authors_folder))
#     filenames_new = set(os.listdir(new_authors))
#     print(len(filenames_new.difference(filenames_old)))
#     print(filenames_new.difference(filenames_old))
#     return filenames_new.difference(filenames_old)

def find_URL(text: str):
    """Function to drop URL from text; Returns text without URL"""
    expression = '(?:http[s]?:\/\/.)?(?:www\.)?[-a-zA-Z0-9@%._\+~#=]{2,256}\.[a-z]{2,6}(?:[-a-zA-Z0-9@:%_\+.,~#?&\/\/=]*)'
    new_list = re.split(expression, text)
    new_text = ''.join(new_list)
    return new_text

def get_rid_of_reddit_quotes(text):
    text_new = []
    quotes = re.split('(>)', text)
    if len(quotes)>1:
        quotes = [ele for ele in quotes if ele != '']
        status = 0
        for ele in quotes:
            if ele == '>':
                status+=1
            else:
                if status<=0:
                    text_new.append(ele)
                else:
                    status -= ele.count('\n')
                    listed = [a for a in ele.split('\n') if a != '']
                    if status == 0:
                        if len(listed)>=1:
                            text_new.append(listed[-1])
                    if status < 0:
                        text_new.append(''.join(list(intersperse(listed[-1+status:], '\n'))))
        return ''.join(text_new)
    else:
        return text

def strip_emojis_by_alphanum(text: str):
    """Function to remove all emojis from the text; This is accomplished by only permitting alphanumeric text Returns the text without emojis and the count of how many items were removed"""
    emojis, text_new = [], []
    for character in text:
        if not character.isalpha() and character not in string.punctuation and character not in string.whitespace and not character.isdigit():
            emojis.append(character)
        else:
            text_new.append(character)
    return {'text': ''.join(text_new), 'emoji_count': len(emojis), 'emojis': emojis}

def helper_preatreatment(text:str):
    if type(text)==str:
        text = get_rid_of_reddit_quotes(text)
        text = find_URL(text)
        return strip_emojis_by_alphanum(text)
    else:
        return {'text': '', 'emoji_count': None, 'emojis': None}

def pretreatment(input_dir='0_users_comments', output_dir='1_users_comments_pretreated'):
    """Function to (1) get rid of reddit quotes, (2) get rid of urls and (3) get rid of emojis (by getting rid of non-whitespace, non-punctuation or non-alpha characters
    Calls Functions find_URL, get_rid_of_reddit_quotes, strip_emojis_by_alphanum"""
    #Get a list of all files_to_treat
    input_files = os.listdir(input_dir)
    output_files = set(os.listdir(output_dir))
    list_of_new_files = []
    for file in input_files:
        if file[-4:]=='.csv':
            if file not in output_files:
                print(file)
                try:
                    temp_df = pd.read_csv(os.path.join(input_dir, file), engine='python')
                except:
                    print("file: Emptyfile")
                try:
                    temp_df['temp'] = temp_df['body'].apply(helper_preatreatment)
                    temp_df['pretreated_body'] = temp_df['temp'].apply(lambda dict: dict['text'])
                    temp_df['emojis'] = temp_df['temp'].apply(lambda dict: dict['emojis'])
                    temp_df['emoji_count'] = temp_df['temp'].apply(lambda dict: dict['emoji_count'])
                    temp_df.drop(columns=['Unnamed: 0', 'temp'], inplace=True)
                    temp_df.to_csv(os.path.join(output_dir, file), index=False)
                    list_of_new_files.append(file)
                except KeyError:
                    print(file, " was deleted")
    return list_of_new_files

#___________ Now lets classify the languages
def get_comments_in_json(list_csv_directory=['1_users_comments_pretreated'], output_directory='2_users_comments_language_detection/json_before', minimal_number_of_comments =30, set_considered=set(), list_of_files=[]):
    for directory in list_csv_directory:
        filenames = os.listdir(directory)
        for filename in filenames:
            if filename[-4:]=='.csv':
                if filename[:-4] not in set_considered:
                    if filename in list_of_files:
                        df_temp = pd.read_csv(os.path.join(directory, filename))[['author',	'pretreated_body','created_utc']]
                        if df_temp.shape[0]>minimal_number_of_comments:
                            df_temp.to_json(os.path.join(output_directory, filename[:-4]+"beforepolyglot.json"))
                            set_considered.add(filename[:-4])
    return set_considered

def load_polyglot_output(polyglot_output_directory='2_users_comments_language_detection/json_output', directory_pretreated_comments='1_users_comments_pretreated', directory_to_save_files_in='2_users_comments_language_detection'):
    filenames = os.listdir(polyglot_output_directory)
    for filename in filenames:
        if filename[-4:]=='json':
            try:
                df_temp = pd.read_json(os.path.join(polyglot_output_directory, filename))
                df_temp[['name', 'code', 'confidence', 'read bytes']] = pd.json_normalize(df_temp['poly_obj'])#get infos out
                df_comment_info = pd.read_csv(os.path.join(directory_pretreated_comments, filename[:-4]+'csv'))
                df = df_temp.merge(right=df_comment_info, left_on='pretreated_body', right_on='pretreated_body')
                df.rename(columns={'author_x': 'author', 'created_utc_x': 'created_utc'}, inplace=True)
                df.drop(columns=['poly_obj', 'author_y', 'created_utc_y'], inplace=True)
                df.to_csv(os.path.join(directory_to_save_files_in, filename[:-4]+'csv'), index=False)
            except:
                print(filename)
    return None



def get_rid_of_certain_subreddits(df, list_of_subreddits=['languagelearning']):
    """Get rid of undesirable subreddits"""
    print(df.shape)
    for element in list_of_subreddits:
        df = df[df['subreddit']!=element]
    print(df.shape)
    return df

def unchunked_data_for_classification(dir_input, dir_output, min_comment_length=50, confidence=98,language_text = 'en' ,language_tokenize = 'english',excluded_subreddits=set(), training_words_needed=6000, margin=50, min_no_of_testing_comments=10):
    """
    Prepares comments for classification
    Makes sure they have a minimal length; are in the target language, and not from the excluded subreddits
    :param dir_input:
    :param dir_output:
    :param min_comment_length:
    :param confidence:
    :param language_text:
    :param language_tokenize:
    :param excluded_subreddits:
    :return:
    """
    #Look if its already done
    files_already_done = os.listdir(dir_output)
    for f in files_already_done:
        if f == 'training_data.csv':
            df_train = pd.read_csv(os.path.join(dir_output, f))
        elif f == 'testing_data.csv':
            df_test = pd.read_csv(os.path.join(dir_output, f))
        else:
            pass
    try:
        authors = set(df_train.author.unique()).intersection(set(df_test.author.unique()))
    except:
        authors = set()
    #
    list_files = os.listdir(dir_input)
    list_files = [file for file in list_files if os.path.isfile(os.path.join(dir_input, file))]
    list_files = [file for file in list_files if file[-4:]=='.csv']
    list_files = [file for file in list_files if file[:-4] not in authors]

    print(authors)
    if authors == set():
        training_data = pd.DataFrame()
        testing_data = pd.DataFrame()
    else:
        training_data = df_train
        testing_data = df_test
    enough_text =1
    for file in list_files:
        df = pd.read_csv(os.path.join(dir_input, file))
        df = df[df['code'].notna()]
        df = df[df['confidence'].notna()]
        df = df[df['pretreated_body'].notna()]
        df = df[df['subreddit'].notna()]
        df = df[df['code']==language_text]
        df = df[df['confidence'].astype(float)>confidence]
        df = df[~df['subreddit'].isin(excluded_subreddits)]
        df['word_count_pb'] = df['pretreated_body'].apply(lambda string: len(word_tokenize(string, language_tokenize)))
        df = df[df['word_count_pb']>=min_comment_length]
        if df['word_count_pb'].sum() > training_words_needed:
            df = df.sort_values(by='created_utc')
            enough_text +=1
            df = df.reset_index()
            df = df[['author', 'pretreated_body', 'word_count_pb', 'subreddit', 'code', 'confidence']]
            df = df.reset_index()
            df = df.rename(columns={'index': 'comment_id'})
            words =0
            potential_split=0
            too_many_words = training_words_needed+margin
            while words < training_words_needed-margin:
                df_temp = df[df['comment_id']<=potential_split]
                words = df_temp['word_count_pb'].sum()
                if too_many_words < words:
                    i = df[df['comment_id']==potential_split].index
                    df.drop(i, inplace=True)
                    df_temp = df[df['comment_id'] <= potential_split]
                    words = df_temp['word_count_pb'].sum()
                if df['word_count_pb'].sum()<training_words_needed:
                    potential_split == df.shape[0] #to make sure it gets rejected
                    break
                potential_split += 1
            df_temp_test = df[df['comment_id'] > potential_split]
            #df_temp_test = df_temp_test.head(min_no_of_testing_comments)#only take amount of testing data necessary
            if df_temp_test.shape[0]>min_no_of_testing_comments:
                print(f"training data: {df_temp['word_count_pb'].sum()} + testing data: {df_temp_test['word_count_pb'].sum()}")
                training_data = pd.concat([training_data, df_temp])
                testing_data = pd.concat([testing_data, df_temp_test])
                training_data.to_csv(os.path.join(dir_output, 'training_data.csv'))
                testing_data.to_csv(os.path.join(dir_output, 'testing_data.csv'))
        else:
            print(f"file {file} has not enough training text")
    print(enough_text)
    if os.path.isdir(dir_output):
        pass
    else:
        os.mkdir(die_output)
    training_data.to_csv(os.path.join(dir_output, 'training_data.csv'))
    testing_data.to_csv(os.path.join(dir_output, 'testing_data.csv'))
    return None

def filter_by_comment_length(comment_lengths, dest_folder, source, language_text='en', confidence=98, excluded_subreddits=set(), language_tokenize='english'):
    list_files = os.listdir(source)
    list_files = [file for file in list_files if os.path.isfile(os.path.join(source, file))]
    list_files = [file for file in list_files if file[-4:]=='.csv']
    for file in list_files:
        df = pd.read_csv(os.path.join(source, file))
        df = df[df['code'].notna()]
        df = df[df['confidence'].notna()]
        df = df[df['pretreated_body'].notna()]
        df = df[df['subreddit'].notna()]
        df = df[df['code']==language_text]
        df = df[df['confidence'].astype(float)>confidence]
        df = df[~df['subreddit'].isin(excluded_subreddits)]
        df['word_count_pb'] = df['pretreated_body'].apply(lambda string: len(word_tokenize(string, language_tokenize)))
        for min_comment_length in comment_lengths:
            df_temp = df[df['word_count_pb']>=min_comment_length]
            if df_temp.shape[0]>10:
                df_temp.to_csv(os.path.join(dest_folder, f'length_{min_comment_length}', file), index=None)
    return None


def unchunked_data_stats(train_file, test_file, comment_length, trainingwords, index_file ='index_files/index_file.csv', save_dir='data/3_training_testing'):
    """
    returns dict with results
    :param train_file:
    :param test_file:
    :return:
    """
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    no_authors = len(df_train.author.unique())
    if len(df_train.author.unique()) != len(df_test.author.unique()):
        raise ValueError
    df_index = pd.read_csv(index_file)
    df_index.drop_duplicates(subset=['author_name'], inplace=True)
    df_index = df_index[['author_name', 'Native', 'level_en']]
    df = df_train.merge(df_index, left_on='author', right_on='author_name')
    df.drop_duplicates(subset=['author_name'], inplace=True)#I want statistics on authors
    df = df.groupby(['Native']).count()
    # df.reset_index(inplace=True)
    # ax = sns.barplot(df, x='Native', y='author', ci=None)
    # plt.title(f"Authors if training data size is {trainingwords} words \n Minimal comment length: {comment_length}")
    # plt.ylabel("#Authors")
    # plt.xlabel("native language of user")
    # plt.xticks(rotation=45, ha='right')
    # for i in ax.containers:
    #     ax.bar_label(i, )
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'{trainingwords}_author_overview.png'))
    # plt.clf()
    # df = df[df['Native']!='en']
    # sns.barplot(df, x='Native', y='author', hue='level_en', ci=None)
    # plt.title(f"Authors if training data size is {trainingwords} words \n Minimal comment length: {comment_length}")
    # plt.ylabel("#Authors")
    # plt.xlabel("native language of user")
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'{trainingwords}_author_levels.png'))
    # plt.clf()
    return {'#authors':no_authors, '#comments_training':df_train.shape[0]/no_authors, '#comments_testing':df_test.shape[0]/no_authors, 'min_comment_length': comment_length, 'words_in_training':trainingwords,
            'en': df.at['en', 'author'], 'it': df.at['it', 'author'], 'es': df.at['es', 'author'], 'pt': df.at['pt', 'author'],
            'de': df.at['de', 'author'], 'fr': df.at['fr', 'author'], 'nl': df.at['nl', 'author']}

def make_shorter_testing(testing_file,name_testing_file, no_of_comments=10):
    df = pd.read_csv(testing_file)
    a = pd.DataFrame()
    for author in df.author.unique():
        df_temp = df[df['author']==author]
        df_temp = df_temp.head(no_of_comments)
        a = pd.concat([a, df_temp])
    a.to_csv(name_testing_file, index=False)
    return None

def unchunked_data_for_classification_with_comment_length_prep(dir_input, dir_output, min_comment_length=50, training_words_needed=5000, margin=50, min_no_of_testing_comments=10, counter = None):
    """
    Prepares comments for classification
    Makes sure they have a minimal length; are in the target language, and not from the excluded subreddits
    :param dir_input:
    :param dir_output:
    :param min_comment_length:
    :param confidence:
    :param language_text:
    :param language_tokenize:
    :param excluded_subreddits:
    :return:
    """
    #Look if its already done
    if counter != None:
        c = 0
    if os.path.isdir(dir_output):
        pass
    else:
        os.mkdir(dir_output)
    files_already_done = os.listdir(dir_output)
    for f in files_already_done:
        if f == 'training_data_good.csv':
            df_train = pd.read_csv(os.path.join(dir_output, f))
        elif f == 'testing_data_good.csv':
            df_test = pd.read_csv(os.path.join(dir_output, f))
        else:
            pass
    try:
        authors = set(df_train.author.unique()).intersection(set(df_test.author.unique()))
    except:
        authors = set()
    print(authors)

    if authors == set():
        training_data = pd.DataFrame()
        testing_data = pd.DataFrame()
    else:
        training_data = df_train
        testing_data = df_test

    enough_text =1
    list_files = os.listdir(dir_input)
    list_files = [file for file in list_files if os.path.isfile(os.path.join(dir_input, file))]
    list_files = [file for file in list_files if file[-4:]=='.csv']
    list_files = [file for file in list_files if file[:-4] not in authors]
    for file in list_files:
        if c>90:
            break
        df = pd.read_csv(os.path.join(dir_input, file))
        if df['word_count_pb'].sum() > training_words_needed+min_comment_length*min_no_of_testing_comments:
            df = df.sort_values(by='created_utc')
            enough_text +=1
            df = df.reset_index()
            df = df[['author', 'pretreated_body', 'word_count_pb', 'subreddit', 'code', 'confidence', 'parent_id', 'created_utc']]
            df = df.reset_index()
            df = df.rename(columns={'index': 'comment_id'})
            words =0
            potential_split=0
            too_many_words = training_words_needed+margin
            while words < training_words_needed-margin:
                df_temp = df[df['comment_id']<=potential_split]
                words = df_temp['word_count_pb'].sum()
                if too_many_words < words:
                    i = df[df['comment_id']==potential_split].index
                    df.drop(i, inplace=True)
                    df_temp = df[df['comment_id'] <= potential_split]
                    words = df_temp['word_count_pb'].sum()
                if df['word_count_pb'].sum()<training_words_needed:
                    potential_split == df.shape[0] #to make sure it gets rejected
                    break
                potential_split += 1
            bad_parents = set(df_temp.parent_id.unique())
            df_temp_test = df[df['comment_id'] > potential_split]
            parents = set(df_temp_test.parent_id.unique()).difference(bad_parents)
            df_temp_test = df_temp_test[df_temp_test['parent_id'].isin(parents)]
            #df_temp_test = df_temp_test.head(min_no_of_testing_comments)#only take amount of testing data necessary
            if df_temp_test.shape[0]>min_no_of_testing_comments:
                print(f"training data: {df_temp['word_count_pb'].sum()} + testing data: {df_temp_test['word_count_pb'].sum()}")
                training_data = pd.concat([training_data, df_temp])
                testing_data = pd.concat([testing_data, df_temp_test])
                training_data.to_csv(os.path.join(dir_output, 'training_data_good.csv'))
                testing_data.to_csv(os.path.join(dir_output, 'testing_data_good.csv'))
                c+=1
        else:
            print(f"file {file} has not enough training text")
    print(enough_text)
    training_data.to_csv(os.path.join(dir_output, 'training_data_good.csv'))
    testing_data.to_csv(os.path.join(dir_output, 'testing_data_good.csv'))

    return None


def unchunked_data_for_classification_without_comment_length_prep(dir_input, dir_output, min_comment_length=50, training_words_needed=5000, margin=50, min_no_of_testing_comments=10, counter=None):
    """
    Prepares comments for classification
    Makes sure they have a minimal length; are in the target language, and not from the excluded subreddits
    :param dir_input:
    :param dir_output:
    :param min_comment_length:
    :param confidence:
    :param language_text:
    :param language_tokenize:
    :param excluded_subreddits:
    :return:
    """
    #Look if its already done
    if counter!=None:
        c = 0
    if os.path.isdir(dir_output):
        pass
    else:
        os.mkdir(dir_output)
    files_already_done = os.listdir(dir_output)
    for f in files_already_done:
        if f == 'training_data_good.csv':
            df_train = pd.read_csv(os.path.join(dir_output, f))
        elif f == 'testing_data_good.csv':
            df_test = pd.read_csv(os.path.join(dir_output, f))
        else:
            pass
    try:
        authors = set(df_train.author.unique()).intersection(set(df_test.author.unique()))
    except:
        authors = set()
    print(authors)

    if authors == set():
        training_data = pd.DataFrame()
        testing_data = pd.DataFrame()
    else:
        training_data = df_train
        testing_data = df_test

    enough_text =1
    list_files = os.listdir(dir_input)
    list_files = [file for file in list_files if os.path.isfile(os.path.join(dir_input, file))]
    list_files = [file for file in list_files if file[-4:]=='.csv']
    list_files = [file for file in list_files if file[:-4] not in authors]
    for file in list_files:
        df = pd.read_csv(os.path.join(dir_input, file))
        if df['word_count_pb'].sum() > training_words_needed+min_comment_length*min_no_of_testing_comments:
            df = df.sort_values(by='created_utc')
            enough_text +=1
            df = df.reset_index()
            df = df[['author', 'pretreated_body', 'word_count_pb', 'subreddit', 'code', 'confidence', 'parent_id', 'created_utc']]
            df = df.reset_index()
            df = df.rename(columns={'index': 'comment_id'})
            words =0
            potential_split=0
            too_many_words = training_words_needed+margin
            while words < training_words_needed-margin:
                df_temp = df[df['comment_id']<=potential_split]
                words = df_temp['word_count_pb'].sum()
                if too_many_words < words:
                    i = df[df['comment_id']==potential_split].index
                    df.drop(i, inplace=True)
                    df_temp = df[df['comment_id'] <= potential_split]
                    words = df_temp['word_count_pb'].sum()
                if df['word_count_pb'].sum()<training_words_needed:
                    potential_split == df.shape[0] #to make sure it gets rejected
                    break
                potential_split += 1
            bad_parents = set(df_temp.parent_id.unique())
            df_temp_test = df[df['comment_id'] > potential_split]
            parents = set(df_temp_test.parent_id.unique()).difference(bad_parents)
            df_temp_test = df_temp_test[df_temp_test['parent_id'].isin(parents)]
            #df_temp_test = df_temp_test.head(min_no_of_testing_comments)#only take amount of testing data necessary
            if df_temp_test.shape[0]>min_no_of_testing_comments:
                print(f"training data: {df_temp['word_count_pb'].sum()} + testing data: {df_temp_test['word_count_pb'].sum()}")
                training_data = pd.concat([training_data, df_temp])
                testing_data = pd.concat([testing_data, df_temp_test])
                training_data.to_csv(os.path.join(dir_output, 'training_data_good.csv'))
                testing_data.to_csv(os.path.join(dir_output, 'testing_data_good.csv'))
                c+=1
        else:
            print(f"file {file} has not enough training text")
        if c>counter:
            return None
    print(enough_text)
    training_data.to_csv(os.path.join(dir_output, 'training_data_good.csv'))
    testing_data.to_csv(os.path.join(dir_output, 'testing_data_good.csv'))
    return None


def get_parent_id_in_traing_and_testing(training, testing, folder_comments, save_file):
    #First get parent_ids_of last
    comments_to_drop = pd.DataFrame()
    df_train = pd.read_csv(training)
    last_lines = df_train.drop_duplicates(subset=['author'],keep='last')
    df_test = pd.read_csv(testing)
    n = 200  # chunk row size
    list_df = [df_test[i:i + n] for i in range(0, df_test.shape[0], n)]
    del df_test

    for df in list_df:
        print(df.shape)
        for author in df.author.unique():
            df_test_temp = df[df['author']==author]
            df_train_temp = df_train[df_train['author']==author]
            author_file = pd.read_csv(os.path.join(folder_comments, f"{author}.csv"))
            df_temp = pd.merge(df_test_temp, author_file, on='pretreated_body')
            df_temp_t = pd.merge(df_train_temp, author_file, on='pretreated_body')
            bad_parent_ids = df_temp_t.parent_id.unique()
            if df_temp[df_temp['parent_id'].isin(bad_parent_ids)].shape[0]>0:
                print("Overlap!")
                comments_to_drop = pd.concat([comments_to_drop, df_temp[df_temp['parent_id'].isin(bad_parent_ids)][['author_x', 'comment_id']]])
                comments_to_drop.drop_duplicates(inplace=True)
                comments_to_drop.to_csv(save_file, index=None)
    return None

def retract_bad_test_comments(testing, badcomments, savefile):
    df_testing = pd.read_csv(testing)
    bad_comments = pd.read_csv(badcomments)
    index_t = []
    df = pd.merge(df_testing, bad_comments, left_on=['author', 'comment_id'],on=['author_x', 'comment_id'], how='outer', indicator=True)
    print(df.testing.shape, bad_comments.shape, df.shape)
    return df

if __name__ == '__main__':
    print("Hello")
    #index_merge()
    ## List new authors
    #set_author = new_authors()
    #data_folder=r"path\users_comments"
    #pretreatment_folder=r"path\pretreatment"
    ## First pretreatment
    #new_files = pretreatment(data_folder, os.path.join(pretreatment_folder,"step0"))
    ##Second language detection
    #get_comments_in_json([os.path.join(pretreatment_folder,"step0")], output_directory=r"C:\Users\jasmi\Documents\Data_Generations\pretreatment\step1\recollect", list_of_files=new_files) #This prepares the comments for language detection (on google collab due to issues with environment with azure
    # Third: Load results language detection
    #load_polyglot_output(polyglot_output_directory=r'path\pretreatment\step2\after1', directory_pretreated_comments=os.path.join(pretreatment_folder,"step0"), directory_to_save_files_in=os.path.join(pretreatment_folder,"step3"))
    # Forth: train_test split
    #res = []
    #filter_by_comment_length([64, 128, 256, 512, 1024, 2048, 4096],
                            # r'path\trainingdata',r'path\pretreatment\step3')
    #
    for comment_length in [64,128,256,512]:
         for training_length in [1000,3000,5000,7000,9000]:
            print(comment_length, training_length)
            unchunked_data_for_classification_without_comment_length_prep(dir_input=fr'path\trainingdata\length_{comment_length}', dir_output=fr'path\trainingdata\length_{comment_length}\train_length_{training_length}',
                                                                            min_comment_length=training_length,
                                                                            training_words_needed=training_length, margin=50,
                                                                          min_no_of_testing_comments=10, counter = 85)
            make_shorter_testing(rf"path\trainingdata\length_{comment_length}\train_length_{training_length}\testing_data_good.csv", rf"path\trainingdata\length_{comment_length}\train_length_{training_length}\testing_data_good_10.csv", no_of_comments=10)
        

    # #_________________make sure we didn't loose too many people
    # step00 = r'users_comments'
    # step0 = r'data_pretreatment\step_0'
    # step1 = r'data_pretreatment\step_1'
    # step2 = r'data_pretreatment\step_2'
    # step3 = r'data_pretreatment\step_3'
    #
    # # step00_a = set([a[:-4] for a in os.listdir(step00)])
    # step0_a = set([a[:-4] for a in os.listdir(step0)])
    # # print(f"lost between step 00 and 0: {len(step00_a.difference(step0_a))}")
    # # for file in step00_a:
    # #     if file in step00_a.difference(step0_a):
    # #         if os.path.getsize(os.path.join(step00, file+'.csv'))>4:
    # #             shutil.copy(os.path.join(step00, file+'.csv'),os.path.join(r'data_pretreatment\step_00',file+'.csv'))
    # #             print(file, os.path.getsize(os.path.join(step00, file+'.csv')))
    # #pretreatment(r'data_pretreatment\step_00', r'data_pretreatment\step_0')
    #
    # step1_a = set([a[:-19] for a in os.listdir(step1)])
    # print(f"lost between step 0 and 1: {len(step0_a.difference(step1_a))}")
    # #for file in step0_a:
    # #    if file in step0_a.difference(step1_a):
    # #        if os.path.getsize(os.path.join(step00, file+'.csv'))>4:
    # #            print(file, os.path.getsize(os.path.join(step00, file+'.csv')))
    # #            shutil.copy(os.path.join(step0, file + '.csv'), os.path.join(
    # #                r'data_pretreatment\step_00',
    # #                file + '.csv'))
    # #get_comments_in_json(
    # #    [r'data_pretreatment\step_00'],
    # #    output_directory=r'data_pretreatment\step_1')  # This prepares the comments for language detection (on google collab due to issues with environment with azure)
    #
    # step2_a = set([a[:-5] for a in os.listdir(step2)])
    # for file in step1_a:
    #     if file in step1_a.difference(step2_a):
    #         shutil.copy(os.path.join(step1, file + 'beforepolyglot.json'), os.path.join(
    #                 r'data_pretreatment\step_00',
    #                 file + 'beforepolyglot.json'))
    # #print(f"lost between step 1 and 2: {len(step1_a.difference(step2_a))}")
    # #step3_a = set([a[:-4] for a in os.listdir(step3)])
    # #print(f"lost between step 2 and 3: {len(step2_a.difference(step3_a))}")

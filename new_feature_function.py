import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
modelSentence = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
modelReddit = SentenceTransformer("gabrielloiseau/LUAR-CRUD-sentence-transformers")
import statistics
import spacy
#spacy.cli.download("en_core_web_sm")
from collections import Counter
#from sentence_transformers import SentenceTransformer, LoggingHandler
import warnings
warnings.filterwarnings("ignore")

pos_lang_trad = {'en':'english', 'de':'german', 'nl':'dutch', 'fr':'french', 'pt':'portuguese', 'it':'italian', 'es':'spanish', 'tr':'turkish'} #Todo extend for other languages
def sentence_features_general(train_data, test_data, column_text):
    X_trainsent = modelSentence.encode_document(list(train_data[column_text]))
    X_testsent =  modelSentence.encode_document(list(test_data[column_text]))
    return pd.DataFrame(X_trainsent), pd.DataFrame(X_testsent)

def sentence_features_trainedonreddit(train_data, test_data, column_text):
    X_trainsent = modelReddit.encode_document(list(train_data[column_text]))
    X_testsent = modelReddit.encode_document(list(test_data[column_text]))
    return pd.DataFrame(X_trainsent), pd.DataFrame(X_testsent)


def char_n_grams_normalized_new(X_train, X_test, name_of_columns_with_text, min_char_n_gram=1, max_char_n_gram=3, max_no_of_features = 1000, stop_word_for_n_gram=None):
    """
    Feature extraction: get char_n_grams out of text / normalized by length of text (length of text #no of characters)
    :param X_train: training dataframe with text data_for_genAI_bias
    :param X_test: testing dataframe with text data_for_genAI_bias
    :param name_of_columns_with_text: column name in dataframe with text to treat
    :param min_char_n_gram: minimal -ngram to make (1-gram, 2-gram...), default 1
    :param max_char_n_gram: maximal -ngram to make (1-gram, 2-gram...), default 3
    :param max_no_of_features: How many columns to add at max
    :param stop_word_for_n_gram: List of stopwords to consider; Should be None (due to limited comparability between languages)
    :return: dataframe with the new features added
    """
    try:
        X_train.reset_index(inplace=True)
        X_test.reset_index(inplace=True)
    except:
        pass
    if set(X_train.code.unique())==set(X_test.code.unique()) and len(set(X_train.code.unique()))==1:
        language = list(X_train.code.unique())[0]
    else:
        print("Multiple languages used!")
        raise ValueError
    res_train = pd.DataFrame()
    res_test = pd.DataFrame()
    for no in range(min_char_n_gram, max_char_n_gram+1):
        char_vectorizer = CountVectorizer(analyzer='char', stop_words=stop_word_for_n_gram, ngram_range=(no, no))  # max_features considers only the top x features ordered by term frequency across the corpus
        #Apply Char_n_frams
        charngrams_vectorizer_train = char_vectorizer.fit_transform(X_train[name_of_columns_with_text])
        charngrams_vectorizer_test = char_vectorizer.transform(X_test[name_of_columns_with_text])
    #Get it into DF
        df_charngrams_train = pd.DataFrame(charngrams_vectorizer_train.toarray(), columns=char_vectorizer.get_feature_names_out())
        df_charngrams_test = pd.DataFrame(charngrams_vectorizer_test.toarray(), columns=char_vectorizer.get_feature_names_out())
    #Rename_columns
        df_charngrams_train.columns = ['char_ngram_' + col for col in df_charngrams_train.columns]
        df_charngrams_test.columns = ['char_ngram_' + col for col in df_charngrams_test.columns]
    # Fill empty
        df_charngrams_train.fillna(0, inplace=True)
        df_charngrams_test.fillna(0, inplace=True)
    # Normalize by number of possible combinations given length
        df_charngrams_train["possible_combinations"] = X_train[name_of_columns_with_text].apply(lambda string: len(string)-(no-1))
        df_charngrams_test["possible_combinations"] = X_test[name_of_columns_with_text].apply(
            lambda string: len(string) - (no - 1)) # does it make sense to normalize these according to this number? Given that it is depended on train?
        df_charngrams_train[df_charngrams_train.columns] = df_charngrams_train[df_charngrams_train.columns].div(df_charngrams_train["possible_combinations"],axis=0)
        df_charngrams_test[df_charngrams_test.columns] = df_charngrams_test[df_charngrams_test.columns].div(df_charngrams_test["possible_combinations"], axis=0)
        df_charngrams_train.drop(columns=["possible_combinations"], inplace=True)
        df_charngrams_test.drop(columns=["possible_combinations"], inplace=True)
    # Concatenate both DFs
        res_train = pd.concat([res_train, df_charngrams_train], axis=1)
        res_test = pd.concat([res_test, df_charngrams_test], axis=1)
    # Pick columns
    summary = res_train.describe().T
    #summary.reset_index()
    summary = summary.nlargest(max_no_of_features, "std").T #take the ones with most variability
    columns_to_keep = summary.columns
    #Only keep columns of interest
    res_train = res_train[columns_to_keep]
    res_test = res_test[columns_to_keep]
    #get everything back together
    return res_train, res_test

def word_n_grams_normalized_new(X_train, X_test, name_of_columns_with_text, min_word_n_gram=1, max_word_n_gram=3, max_no_of_features = 1000, stop_word_for_n_gram=None, language='en'):
    """
    Feature extraction: get char_n_grams out of text / normalized by length of text (length of text #no of characters)
    :param X_train: training dataframe with text data_for_genAI_bias
    :param X_test: testing dataframe with text data_for_genAI_bias
    :param name_of_columns_with_text: column name in dataframe with text to treat
    :param min_word_n_gram: minimal -ngram to make (1-gram, 2-gram...), default 1
    :param max_word_n_gram: maximal -ngram to make (1-gram, 2-gram...), default 3
    :param max_no_of_features: How many columns to add at max
    :param stop_word_for_n_gram: List of stopwords to consider; Should be None (due to limited comparability between languages)
    :param language: language text is in
    :return: dataframe with the new features added
    """
    try:
        X_train.reset_index(inplace=True)
        X_test.reset_index(inplace=True)
    except:
        pass
    if set(X_train.code.unique())==set(X_test.code.unique()) and len(set(X_train.code.unique()))==1:
        language = list(X_train.code.unique())[0]
    else:
        print("Multiple languages used!")
        raise ValueError
    res_train = pd.DataFrame()
    res_test = pd.DataFrame()
    for no in range(min_word_n_gram, max_word_n_gram+1):
        word_vectorizer = CountVectorizer(analyzer='word', stop_words=stop_word_for_n_gram, ngram_range=(no, no))  # max_features considers only the top x features ordered by term frequency across the corpus
        #Apply Char_n_frams
        wgrams_vectorizer_train = word_vectorizer.fit_transform(X_train[name_of_columns_with_text])
        wgrams_vectorizer_test = word_vectorizer.transform(X_test[name_of_columns_with_text])
    #Get it into DF
        df_wngrams_train = pd.DataFrame(wgrams_vectorizer_train.toarray(), columns=word_vectorizer.get_feature_names_out())
        df_wngrams_test = pd.DataFrame(wgrams_vectorizer_test.toarray(), columns=word_vectorizer.get_feature_names_out())
    #Rename_columns
        df_wngrams_train.columns = ['word_ngram_' + col for col in df_wngrams_train.columns]
        df_wngrams_test.columns = ['word_ngram_' + col for col in df_wngrams_test.columns]
    # Fill empty
        df_wngrams_train.fillna(0, inplace=True)
        df_wngrams_test.fillna(0, inplace=True)
    # Normalize by number of possible combinations given length
        df_wngrams_train["possible_combinations"] = X_train[name_of_columns_with_text].apply(lambda string: len(word_tokenize(text=string, language=pos_lang_trad[language]))-(no-1))
        df_wngrams_test["possible_combinations"] = X_test[name_of_columns_with_text].apply(lambda string: len(word_tokenize(text=string, language=pos_lang_trad[language])) - (no - 1)) # does it make sense to normalize these according to this number? Given that it is depended on train?
        df_wngrams_train[df_wngrams_train.columns] = df_wngrams_train[df_wngrams_train.columns].div(df_wngrams_train["possible_combinations"],axis=0)
        df_wngrams_test[df_wngrams_test.columns] = df_wngrams_test[df_wngrams_test.columns].div(df_wngrams_test["possible_combinations"], axis=0)
        df_wngrams_train.drop(columns=["possible_combinations"], inplace=True)
        df_wngrams_test.drop(columns=["possible_combinations"], inplace=True)
    # Concatenate both DFs
        res_train = pd.concat([res_train, df_wngrams_train], axis=1)
        res_test = pd.concat([res_test, df_wngrams_test], axis=1)
    # Pick columns
    summary = res_train.describe().T
    #summary.reset_index()
    summary = summary.nlargest(max_no_of_features, "std").T #take the ones with most variability
    columns_to_keep = summary.columns
    #Only keep columns of interest
    res_train = res_train[columns_to_keep]
    res_test = res_test[columns_to_keep]
    return res_train, res_test

def Yules_K_function(text: str, language='en'):
    """
    Helper function for features, to detrmine Yules K based on
    "Stylistic Constancy and Change across Literary Corpora: Using Measures of Lexical Richness to Date Works, Author(s): J. A. Smith and C. Kelly,Source: Computers and the Humanities , Nov., 2002, Vol. 36, No. 4 (Nov., 2002), pp. 411-430"

    """
    words = word_tokenize(text=text, language=pos_lang_trad[language])
    N = len(words)
    #1 Count occurances of each word in text
    if N>0:
        dict_counts = Counter(words) #which word occurs how many times
        how_many_times = dict(Counter(list(dict(dict_counts).values())))
        res = -(1/N)
        for element in how_many_times.keys():
            res += (how_many_times[element]*how_many_times[element]*element/(N**2)) #how_many_times[element] is equivalent to i, element is equivalent to V(i, N)
        return 10000*res
    else:
        return None

def TTR_function(text: str, language='en'):
    """Helper function for features to determine a texts Type-Token Ratio
    Function taken from https://doi.org/10.1007/978-3-030-53360-1, Eq. 2.3"""
    total_world = len(word_tokenize(text=text, language=pos_lang_trad[language]))
    unique_words = len(set(word_tokenize(text=text, language=pos_lang_trad[language])))
    if total_world:
        return unique_words/total_world
    else:
        return 0

def POS_TAG_helper(nlp, text):
  try:
    doc = nlp(text)
    return ' '.join([w.pos_ for w in doc])
  except Exception as ex:
    print(text)
    raise ex

def pos_n_grams_normalized_new(X_train, X_test, column_text, pos_lang='en', min_pos_n_gram=1, max_pos_n_gram =3, pos_tag_n_gram_max=1000,
                            stop_word_for_n_gram=None, normalize=True):
    try:
        X_train.reset_index(inplace=True)
        X_test.reset_index(inplace=True)
    except:
        pass
    if set(X_train.code.unique()) == set(X_test.code.unique()) and len(set(X_train.code.unique())) == 1:
        language = list(X_train.code.unique())[0]
    else:
        print("Multiple languages used!")
        raise ValueError
    res_train = pd.DataFrame()
    res_test = pd.DataFrame()

    X_train['total_w'] = X_train.apply(
    lambda x: len(word_tokenize(text=x[column_text], language=pos_lang_trad[pos_lang])), axis=1)
    X_test['total_w'] = X_test.apply(
    lambda x: len(word_tokenize(text=x[column_text], language=pos_lang_trad[pos_lang])), axis=1)
    db_dict = {'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm', 'de': 'de_core_news_sm', 'es': 'es_core_news_sm',
     'it': 'it_core_news_sm', 'pt': 'pt_core_news_sm', 'nl': 'nl_core_news_sm'}
    nlp = spacy.load(db_dict[pos_lang])
    X_train['pos_tags'] = X_train[column_text].apply(lambda x: POS_TAG_helper(nlp, x))
    X_test['pos_tags'] = X_test[column_text].apply(lambda x: POS_TAG_helper(nlp, x))
    for no in range(min_pos_n_gram, max_pos_n_gram + 1):
        # Extract POS n-grams
        pos_vectorizer = CountVectorizer(analyzer='word', stop_words=stop_word_for_n_gram, ngram_range=(no, no))
        pos_vectorizer_train = pos_vectorizer.fit_transform(X_train['pos_tags'])
        pos_vectorizer_test = pos_vectorizer.transform(X_test['pos_tags'])
        df_train = pd.DataFrame(pos_vectorizer_train.toarray(),
                        columns=pos_vectorizer.get_feature_names_out())  # convert into a DF
        df_test = pd.DataFrame(pos_vectorizer_test.toarray(),
                       columns=pos_vectorizer.get_feature_names_out())
        # Rename_columns
        df_train.columns = ['POS_ngram_' + col for col in df_train.columns]
        df_test.columns = ['POS_ngram_' + col for col in df_test.columns]
        # Fill empty
        feature_cols = df_train.columns
        df_train = pd.concat([df_train, X_train["total_w"]], axis=1)
        df_test = pd.concat([df_test, X_test["total_w"]], axis=1)# convert into a DF
        # normalize
        if normalize:
            df_train[feature_cols] = df_train[feature_cols].div(df_train['total_w']-(no-1), axis=0)
            df_test[feature_cols] = df_test[feature_cols].div(df_test['total_w']-(no-1), axis=0)
        # drop unwanted columns
        df_train.drop(columns=['total_w'], inplace=True)
        df_test.drop(columns=['total_w'], inplace=True)
        df_train.fillna(0)
        df_test.fillna(0)
        control=df_train.sum(axis=1)
        res_train = pd.concat([res_train, df_train], axis=1)
        res_test = pd.concat([res_test, df_test], axis=1)
    # Pick columns
    summary = res_train.describe().T
    #summary.reset_index()
    summary = summary.nlargest(pos_tag_n_gram_max, "std").T #take the ones with most variability
    columns_to_keep = summary.columns
    #Only keep columns of interest
    res_train = res_train[columns_to_keep]
    res_test = res_test[columns_to_keep]
    return res_train, res_test

def get_length_list(text, language):
    """Helperfunction for features"""
    tokenized_comment_chunk = word_tokenize(text=text, language=language)
    len_list = []
    for word in tokenized_comment_chunk:
        len_list.append(len(word))
    return len_list
def style_features_normalized_new(train_data, test_data, column_text, chunkified=False, round_value=3, word_level_total_words=False, average_word_length=True, median_word_length=True, stan_dev_word_length=True, word_length_distribution=True,
                                    min_range_w_len_dist=1, max_range_w_len=20, char_level_total_char=True, char_n_grams=True, char_n_gram_max_f=1000, word_n_grams=True, word_n_gram_max_f=1000,
                                    stop_word_for_n_gram=None, n_gram_char=3, n_gram_word=3, TTR=True, yules_K=True, POS_Tags=True, n_gram_pos=3, pos_tag_n_gram_max=1000, pos_lang='en',
                                    emoji_count=False):
    """
    Extracts features for analysis, needs multiple helper functions! Only works for monolinguistic train/test sets
    :param train_data: Training data_for_genAI_bias (df)
    :param test_data: Testing Data (df)
    :param column_text: column name in dataframe with text to analyze
    :param chunkified: if chunkified, feature emoji_count is not possible!
    :param round_value: To which decimal value should floatvalues be rounded?
    :param word_level_total_words: Is the feature "total number of words" required? Doesn't make sense for chunkified text True/false (this doesn't work if chunked)
    :param average_word_length: Should average length of words in text be included as a feature True/false
    :param median_word_length: Should median length of words in text be included as a feature True/false
    :param stan_dev_word_length: Should standart deviation of words in text be included as a feature True/false
    :param word_length_distribution: Should the distribution of the length of words in text be included as a feature True/false
    :param min_range_w_len_dist: Minimal length of words the word_length_distribution
    :param max_range_w_len: Maximal length of words the word_length_distribution
    :param char_level_total_char: Is the feature "total number of characters" required? True/false
    :param char_n_grams: Should char_ngrams be made True/false
    :param char_n_gram_max_f: int; maximum number of features to consider for char_n_gram
    :param word_n_grams: Should word_grams be made True/false
    :param word_n_gram_max_f: int; maximum number of features to consider for word_n_gram
    :param stop_word_for_n_gram: List of stopwords to consider; Should be None (due to limited comparability between languages)
    :param n_gram_char: Upper range of character-n-grams; N-grams are constructed for range(1, n_gram_char); In writeprints to trigrams
    :param n_gram_word: Upper range of word-n-grams; N-grams are constructed for range(1, n_gram_word); In writeprints to trigrams
    :param TTR: Should Type-Token Ratio be calculated; True/false
    :param yules_K: Should Yules K be calculated; True/false
    :param POS_Tags: Should POS_ngrams be made True/false
    :param n_gram_pos: Upper range of pos-tag-n-grams; N-grams are constructed for range(1, n_gram_pos); In writeprints to trigrams
    :param pos_tag_n_gram_max: int; maximum number of features to consider for pos_n_gram
    :param pos_lang: two letter language code of language of training/testing data_for_genAI_bias, str
    :param emoji_count: bool
    :return: train_df, test_df
    """
    #Make a safety copy =) without empty rows
    df_train = train_data[train_data[column_text].notna()]
    df_test = test_data[test_data[column_text].notna()]
    columns_to_drop_train = df_train.columns
    columns_to_drop_test = df_test.columns
    # LEXICAL FEATURES
    # Total words
    df_train['total_w'] = df_train.apply(lambda x: len(word_tokenize(text=x[column_text], language=pos_lang_trad[pos_lang])), axis=1)
    df_test['total_w'] = df_test.apply(lambda x: len(word_tokenize(text=x[column_text], language=pos_lang_trad[pos_lang])), axis=1)
    #Average length of words
    if average_word_length:
        df_train['avg_w_len'] = df_train.apply(lambda x: round(statistics.mean(get_length_list(x[column_text], pos_lang_trad[pos_lang])) if x['total_w'] >0 else None), axis=1)
        df_test['avg_w_len'] = df_test.apply(lambda x: round(statistics.mean(get_length_list(x[column_text], pos_lang_trad[pos_lang])) if x['total_w'] >0  else None), axis=1)
    #Median length of words
    if median_word_length:
        df_train['med_w_len'] = df_train.apply(lambda x: round(statistics.median(get_length_list(x[column_text], pos_lang_trad[pos_lang])) if x['total_w'] >0  else None), axis=1)
        df_test['med_w_len'] = df_test.apply(lambda x: round(statistics.median(get_length_list(x[column_text], pos_lang_trad[pos_lang])) if x['total_w'] >0  else None), axis=1)
    if stan_dev_word_length:
        df_train['stdev_w_len'] = df_train.apply(lambda x: round(statistics.stdev(get_length_list(x[column_text],pos_lang_trad[pos_lang])), round_value) if x['total_w'] >2 else None, axis=1)
        df_test['stdev_w_len'] = df_test.apply(lambda x: round(statistics.stdev(get_length_list(x[column_text], pos_lang_trad[pos_lang])), round_value) if x['total_w'] >2 else None, axis=1)
    #Distribution of word length; Normalized by length of comment
    if word_length_distribution:
        for length in range(min_range_w_len_dist, max_range_w_len + 1):
            df_train[f"#w_len_{length}"] = df_train.apply(lambda x: get_length_list(x[column_text], language=pos_lang_trad[pos_lang]).count(length)/x['total_w'] if x['total_w'] >0 else 0, axis=1) #Normalized for the length of comment in case it's not chunkified
            df_test[f"#w_len_{length}"] = df_test.apply(lambda x: get_length_list(x[column_text], language=pos_lang_trad[pos_lang]).count(length)/x['total_w'] if x['total_w'] >0 else 0, axis=1) #Normalized for the length of comment in case it's not chunkified
    if char_level_total_char:
        df_train['total_char'] = df_train.apply(lambda x: len(x[column_text]), axis=1)
        df_test['total_char'] = df_test.apply(lambda x: len(x[column_text]), axis=1)
    #ngrams (character n-grams are considered to be lexical; word-n-grams are considered to be content-related features)
    #Special charcter 1-grams, as well as special charcters are considered to be taken into account with the characterr 1-grams
    if char_n_grams:
        # # Extract character n-grams (includes character freq. extraction), TfidfVectorizer = equivalent to CountVectorizer followed by TfidfTransformer and includes auto-normalizing
        df_train_char_ngram, df_test_char_ngrams = char_n_grams_normalized_new(df_train, df_test, column_text, min_char_n_gram=1, max_char_n_gram=n_gram_char, max_no_of_features=char_n_gram_max_f, stop_word_for_n_gram=stop_word_for_n_gram)
        df_train_char_ngram.fillna(0)
        df_test_char_ngrams.fillna(0)
        df_train = pd.concat([df_train, df_train_char_ngram], axis=1)
        df_test = pd.concat([df_test, df_test_char_ngrams], axis=1)
    if word_n_grams:
        # Extract word n-grams (includes BoW feature extraction)
        df_train_temp = train_data[train_data[column_text].notna()]
        df_test_temp = test_data[test_data[column_text].notna()]

        df_wordngrams_train_final, df_wordngrams_test_final = word_n_grams_normalized_new(df_train_temp, df_test_temp, column_text, min_word_n_gram=1, max_word_n_gram=n_gram_word, max_no_of_features=word_n_gram_max_f, stop_word_for_n_gram=stop_word_for_n_gram, language=pos_lang)
        # Replace empty values
        df_wordngrams_train_final.fillna(0)
        df_wordngrams_test_final.fillna(0)
        # Concatenate both DFs
        df_train = pd.concat([df_train, df_wordngrams_train_final], axis=1)
        df_test = pd.concat([df_test, df_wordngrams_test_final], axis=1)
    #POS-TAGs, POS-TAG-ngrams
    if POS_Tags:
        db_dict = {'en':'en_core_web_sm', 'fr':'fr_core_news_sm', 'de':'de_core_news_sm', 'es':'es_core_news_sm', 'it':'it_core_news_sm', 'pt':'pt_core_news_sm', 'nl':'nl_core_news_sm'}
        if pos_lang not in db_dict:
          raise ValueError(f"Error in POS-TAG compution: {pos_lang} isn't available use 'en', 'fr', 'de', 'es', 'it', 'pt' and 'nl")
        pos_train, pos_test = pos_n_grams_normalized_new(df_train, df_test, column_text,)
        df_train = pd.concat([df_train, pos_train], axis=1)
        df_test = pd.concat([df_test, pos_test], axis=1)
    #Vocabulary Richness Measures
    #TTR (Type-Token Ratio)
    if TTR:
        df_train['TTR'] = df_train[column_text].apply(lambda x: TTR_function(str(x)))
        df_test['TTR'] = df_test[column_text].apply(lambda x: TTR_function(str(x)))
    #Yules K
    if yules_K:
        df_train['yules_K'] = df_train[column_text].apply(lambda x: Yules_K_function(str(x)))
        df_test['yules_K'] = df_test[column_text].apply(lambda x: Yules_K_function(str(x)))
    if chunkified or not emoji_count:
        try:
            df_train= df_train.drop(columns=['emoji_count'])
            df_test= df_test.drop(columns=['emoji_count'])
        except:
            pass
    else:
        df_train['emoji_count'] = df_train.apply(lambda x: x.emoji_count/x.total_w if x.total_w != 0 else 0, axis=1)
        df_test['emoji_count'] = df_test.apply(lambda x: x.emoji_count/x.total_w if x.total_w != 0 else 0, axis=1)
    if not word_level_total_words and 'total_w' in df_train.columns:
        df_train.drop(columns=['total_w'], inplace=True)
    if not word_level_total_words and 'total_w' in df_test.columns:
        df_test.drop(columns=['total_w'], inplace=True)
    df_train.drop(columns=columns_to_drop_train, inplace=True)
    df_test.drop(columns=columns_to_drop_test, inplace=True)
    if 'index' in df_train.columns:
        df_train.drop(columns='index', inplace=True)
    if 'index' in df_test.columns:
        df_test.drop(columns='index', inplace=True)
    df_train.fillna(0)
    df_test.fillna(0)
    if 'level_0' in df_train.columns:
        df_train.drop(columns=['level_0'], inplace=True)
    if 'level_0' in df_test.columns:
        df_test.drop(columns=['level_0'], inplace=True)
    return df_train, df_test

if __name__ == "__main__":
    print("Hello")
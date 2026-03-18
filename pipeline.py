### Author: J. Wyss and R. Overdorf
import datetime
import re
import wayback
import re
import time
import pandas as pd
from tqdm import tqdm
import os
import csv
import re

import reddit_API_caller as rapi
    
### Step 1: Collect urls from wayback API
def get_reddit_thread_ids_from_wayback(subreddit, threads_file, logfile, from_date=datetime.datetime(2000, 1, 1), to_date=datetime.date.today()):
    client = wayback.WaybackClient()
    qs = ['https://www.reddit.com/r/%s', 'https://old.reddit.com/r/%s', 'https://www.reddit.com/r/%s/new', 'https://old.reddit.com/r/%s/new', 'https://www.reddit.com/r/%s/top', 'https://old.reddit.com/r/%s/top']
    print(from_date)
    print(to_date)
    for q in qs:
        print(q)
        records = []
        results = client.search(q%subreddit, from_date=from_date, to_date=to_date)
        print(qs[0]%subreddit)
        for record in results:
            records.append(record)
        
        for record in tqdm(records):
            try:
                response = client.get_memento(record)
                content = response.content.decode()
                res = set(re.findall('\/r\/%s\/comments\/\w+\/\w+\/' % subreddit, content))
                with open(threads_file, 'a') as log:
                    for element in res:
                        log.writelines(f"{response.timestamp.isoformat()},{element.split('/')[4]},{element.split('/')[5]}\n")
                    log.close()
            except Exception as ex:
                with open(logfile, 'a') as log:
                    log.writelines(f"{ex}\n")
                    log.close()
            time.sleep(1)

def reduce_threads_file(threads_file_complete, name_new_file, languages_of_interest={'es', 'pt', 'nl', 'de', 'it', 'fr'}, threads_per_year_and_language=100, confidence_min=99):
    """
    The problem is we have way too much threads to collect users from
    (the program would run 153 days)
    Thus we need to be strategic about it:
    1. we identify the language of the thread name
    2. for all thread names in one of the target languages we subsample 100 per language per year ( we don't control for origin of subreddit)
    3. we save this new reduced set of threads to collect for further operation

    :param threads_file_complete: threads id file with language detection results
    :param name_new_file: where result is saved
    :return: None
    """
    df = pd.read_csv(threads_file_complete)
    print(df.shape)
    df['year'] = df['timestamp'].apply(lambda timstamp: timstamp.split('-')[0])
    df['confidence']=pd.to_numeric(df['confidence'], errors='coerce')
    df.dropna(subset=['confidence'], inplace=True)
    df = df[df['confidence']>confidence_min] #filter
    df = df[df['code'].isin(languages_of_interest)]
    df = df[['timestamp', 'thread_id', 'raw_text', 'code', 'confidence', 'year']]
    new_df = pd.DataFrame()
    for year in df.year.unique():
        for language in df.code.unique():
            df_temp = df[df['year']==year]
            df_temp = df_temp[df_temp['code']==language]
            print(f"For year {year} we have {df_temp.shape[0]} entries in language {language}")
            if df_temp.shape[0]<=threads_per_year_and_language:
                new_df = pd.concat([new_df, df_temp])
            else:
                df_temp = df_temp.sample(n=threads_per_year_and_language, random_state=42)
                new_df = pd.concat([new_df, df_temp])
    new_df.to_csv(name_new_file, index=False)
    return



def get_users_from_threads(threads_file, users_file):
    num_breaks = 0
    waittime = 5
    df = pd.read_csv(threads_file)[['timestamp','thread_id','raw_text']]
    #ca c'est si read file ne marche pas
    # d = {}
    # with open(threads_file) as f:
    #     reader = csv.reader(f, delimiter=',', quotechar='"')
    #     index = 0
    #     for row in reader:
    #         if len(row)==3:
    #             d[index] = row
    #             index+=1
    #         elif len(row) ==5:
    #             list_1 = row[:3]
    #             list_1[2] = list_1[2].split('20')[0]
    #             d[index] = list_1
    #             index += 1
    #             list_2 = row[2:]
    #             z = re.match("20\d\d-[0,1]\d-[0,1,2,3]\dT[0,1,2]\d:[0,1,2,3,4,5]\d:[0,1,2,3,4,5]\d\+\d\d:\d\d", row[2])
    #             if z:
    #                 list_2[0] = z.group()
    #                 d[index] = list_2
    #                 index += 1
    #                 print(list_1, list_2)
    #         else:
    #             print("row skipped:", row)
    # df = pd.DataFrame(d).T
    #
    df.rename(columns={'timestamp':'timestamp', 'thread_id':'id', 'raw_text':'thread_name'}, inplace=True)
    df = df.dropna()
    df = df.drop_duplicates(subset=['id'])
    thread_ids = list(df['id'])
    i = 0
    df_res = pd.DataFrame()
    for thread_id in tqdm(thread_ids):
        try:
            d = rapi.collect_users_from_post(thread_id, filter_by_flair=False)
            if d==False:
                pass #do nothing
            elif len(d) > 1:
                if i == 0:
                    df = pd.DataFrame(d)
                    df_res = pd.concat([df_res, df])
                    df_res.to_csv(users_file, index=None)
                    i = i + 1
                else:
                    df = pd.read_csv(users_file)
                    df = pd.concat([df, pd.DataFrame(d)])
                    df_res = pd.concat([df_res, df])
                    df_res.to_csv(users_file, index=None)
                    i = i + 1
            time.sleep(waittime)
        except Exception as ex:
            #waittime = waittime + 5
            print(f"Exception: {ex}")
            if ex != "'Redditor' object has no attribute 'id'":
                num_breaks = num_breaks + 1
                print(f"Number of exceptions caught:  {num_breaks}")
            else:
                print("redditor id exception caught")
            if num_breaks > 2:
                quit()

def get_post_histories(users_file_labeled, save_loc):
    tqdm.pandas()
    df = pd.read_csv(users_file_labeled)
    df['author_name'].progress_apply(lambda x:  rapi.request_comments(x, f'{save_loc}/{x}.csv'))

if __name__ == '__main__':
    subreddit_file  = 'subreddits_to_collect.txt' #file with subnames seperated by comma
    threads_file = 'path/reddit_thread_ids_from_wayback.csv'
    threads_file_2= 'path/reddit_thread_ids_from_wayback_lang_detect_1.csv'
    threads_file_reduced = 'path/reddit_thread_ids_from_wayback_lang_detect_reduced.csv'
    logfile = 'path/different_languages_data_collection/data/logfile.csv'
    users_file = 'path/user_ids_from_threads_reduced.csv'
    users_file_final = 'path/user_ids_from_threads_reduced_noduplicates.csv'
    save_loc = 'path/users_comments/'
    with open(subreddit_file, 'r') as subreddit_f:
        sub_list = subreddit_f.read().split(',')
        subreddit_f.close
    for subreddit in sub_list:
        ### Step 1: Collect urls from wayback API
       get_reddit_thread_ids_from_wayback(subreddit, threads_file, logfile) #, from_date=datetime.datetime(2020, 1, 1),
    reduce_threads_file(threads_file_2, threads_file_reduced, confidence_min=95)
        ### Setp 2: Collect users from threads collected in step 1
    get_users_from_threads(threads_file_reduced, users_file)
        ### Step 3: Collect user post histories
    #get rid of duplicates
    df = pd.read_csv(users_file)
    print(f"Number of entries before: {df.shape}")
    df.drop_duplicates(inplace=True)
    print(f"Number of entries after: {df.shape}")
    df.to_csv(users_file_final, index=None)
    get_post_histories(users_file_final, save_loc)

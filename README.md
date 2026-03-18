# UnFairMistakes
Code Used for "(Un)fair Mistakes on Social Media: How Demographic Characteristics Influence Authorship Attribution"

The code is split into multiple .py files

The code used for **collecting the dataset** is in the following files:

- **pipeline.py**
  
  The code in this file is used to construct the dataset
  
  This file contains the following functions:
  
    **get_reddit_thread_ids_from_wayback** A function that accesses the wayback API and searches for all saved reddit frontpages of a subreddit of interest in a given timeframe. It saves the different thread_ids displayed on those pages in a csv file

    **reduce_threads_file** A function that takes the csvfile resulting from calling **get_reddit_thread_ids_from_wayback** and reduces it to contain only discussions where the title is in a language of interst

    **get_users_from_threads** A function that collects all user names associated with a reddit post (collects user names of the people who commented on the post as well as the original author).

    **get_post_histories** A function that collects the entire accessible post history of a user.
  
- **reddit_API_caller.py**

    This file contains the helper code to access the reddit api. It also contain the instructions on how to get reddit api credentials.
  

The code used to **clean the data and prepare it** for the experiments is in the following files:

- **pretreatment.py** This file contains all functions necessary to clean the data as well as doing a time aware train-test split

- **PolyglotTest.ipynb** This jupyter Notebook was used to run polgylot. 

The code used to extract features from the data

- **new_feature_function.py** Contains all feature functions used in the experiments




The code used for additional analysis

- cosinedistance.py Code used to compare the cosine distance between training and testing data.

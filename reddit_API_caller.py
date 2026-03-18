import os
import praw
import pandas as pd
from datetime import datetime
import json
import prawcore.exceptions as praw_ex
import time

"""
This files is meant to be imported by another and its methods used to collected data. 
This file contains methods to collect the most recent posts on a subreddit, collect 
the usernames of everyone who has commented on the post, and collect post history from a user
"""

def log(text):
    with open('log.txt', mode='a', encoding='utf-8') as log_file:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{timestamp}] {text}\n')
        print(f'[{timestamp}] {text}')
        log_file.close()


with open('credentials.txt', 'r') as f:
    """
    Info on obtaining credentials: https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
    Reddit Collection Setup (runs on "import reddit_collector")
    credentials.txt should look like (IN THIS ORDER): 
    client_id = client_id_here
    client_secret = client_secret_here
    user_agent = name_of_app
    """
    client_id = next(f).split(' = ')[1].strip()
    client_secret = next(f).split(' = ')[1].strip()
    user_agent = next(f).split(' = ')[1].strip()
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    f.close()


def collect_most_recent_post_ids(subreddit, limit=1000):
    """
    Queries Reddit for the most recent posts on a subreddit
    :param subreddit: string, the name of the subreddit to collect
    :param limit: how many posts to collect (max 1000)
    :return: df: post_id, post_timestamp
    """
    new_posts = reddit.subreddit(subreddit).new(limit=limit)  # send the request to reddit, max 1000 posts
    data = []
    for post in new_posts:  # reformat the data into a list of dicts with only the info we want (id, datetime)
        data.append({'postid': str(post), 'num_comments': post.num_comments,'created_utc': post.created_utc, 'collection_date': datetime.now().strftime('%d-%m-%y')})  # <-- the line to edit if you want more than just the post ids and timestamps
    df = pd.DataFrame(data, index=None)  # throw it into a df so that it's easy to save and play with
    return df


def collect_users_helper(comment, filter_by_flair):
    if comment.author == None:
        return False
    if filter_by_flair:
        if comment.author_flair_text == None or comment.author_flair_text == '':
            return False
#        in_flair = False
#        for el in filter_by_flair:
#            if el in comment.author_flair_text:
#                in_flair = True
#        if not in_flair:
#            return False
    try:
        if comment.author.id is not None:
            return {'author_name': comment.author.name, 'author_id': comment.author.id, 'thread_id': comment.link_id[3:], 'comment_id': comment.id, 'flair': comment.author_flair_text}
        else:
            return None
    except Exception as ex:
        log(str(ex))
        return False


def collect_users_from_post(post, limit=1000, filter_by_flair=False):
    """
    Collects all users (and their flair) who have commented in the post (so far).
    :param post: string, the id of the post to collect
    :param limit: how many comments to collect (max 1000 as of nov 2022)
    :return: a df containing all the users who commented on the post
    """

#    print('On post: %s\r' % post, end="")
#    time.sleep(1) # don't get rate limited
    submission = reddit.submission(id=post)  # send the request to reddit
    submission.comments.replace_more(limit=limit)  # idk, the docs say to do this to deal with the more button stuff
    data = []  # the array that holds all comments
    for comment in submission.comments:  # for each comment in the post
        d = collect_users_helper(comment, filter_by_flair)  # make a dict of the things we want to keep
        if d:
            d['post'] = post
            data.append(d)  # add it to the array that holds everything
        for reply in comment.replies:  # for each reply to the comment above...
            d = collect_users_helper(reply, filter_by_flair)  # make a dict of the things we want to keep
            if d:
                d['post'] = post
                data.append(d)  # add it to the array that holds everything
    return data


def request_comments(username, save_loc, total=None, limit=None):
    """
    Collects the full history of comments of username and saves them in save_loc
    :param username: the user to collect
    :param total: optional, how many users there are to collect (used to print progress)
    :param save_loc: location to save the comments
    :return: None
    """
#    if '%s' in save_loc:
#        save_loc = save_loc % username
#    num_collected = len([name for name in os.listdir('data/language/user_comments')])

#    if os.path.exists(save_loc):
#        pass
#    else:
    if total is not None:
        pass
    if save_loc is not None and os.path.exists(save_loc):
        return None
#            print('Collected: %d out of %d\tOn user: %s' % (num_collected, total, username), end="")
#            print('On user: %s' % (username), end="")
    try:
        comments = []
        for c in reddit.redditor(username).comments.new(limit=None):
            d = {'author': c.author.name, 'body': c.body, 'created_utc': c.created_utc, 'edited': c.edited, 'id': c.id, 'is_submitter': c.is_submitter, 'link_id': c.link_id, 'parent_id': c.parent_id, 'score': c.score, 'subreddit': c.subreddit, 'subreddit_id': c.subreddit_id}
            for key, val in d.items():
                d[key] = str(val)  # string everything
            comments = comments + [d]
    except Exception as ex:
        if '403' in str(ex):
            print('403')
            comments = ['suspended']
        elif '404' in str(ex):
            print('404')
            comments = ['deleted']
        else:
            comments = [str(ex)]  # even if we don't recognize the error, let's not kill the script. Record it and move on.
#        f = pd.read_json(comments, orient='list')
    if save_loc == None:
        return comments
    else:
        df = pd.DataFrame(comments)
        df.to_csv(save_loc)
#        with open(save_loc, "w") as o:
#            json.dump(comments, o)
#            o.close()

def get_new_posts(n):
    subreddit = reddit.subreddit('all')
    posts = subreddit.new(limit=n)
    return [post for post in posts]


def get_upvotes_for(post_id):
    post = reddit.submission(id=post_id)
    return post.score


def get_upvotes(post_ids):

    posts = reddit.info(fullnames=['t3_%s' % i for i in post_ids])
    # Get the posts from the Reddit API
    # posts = [reddit.submission(id=post_id) for post_id in post_ids]

    # Get the number of upvotes for each post
    upvotes = [post.ups for post in posts]

    return upvotes


def get_flair_from_comment(comment_id):
    comment = reddit.comment(comment_id)
    return comment.author_flair_text
    

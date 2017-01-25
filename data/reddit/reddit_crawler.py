import json
import os
import time

import praw

r = praw.Reddit("CMV_Comments")


def get_discussion(link, folder):
    if link is None:
        link = 'https://www.reddit.com/r/changemyview/comments/5312p1/cmv_apathetic_parents_should_be_held_accountable/'
    if folder is None:
        folder = './'
    discussion = r.get_submission(link)
    discussion.replace_more_comments(limit=None, threshold=0)

    # parent_id : parent comment/discussion name (discussion do not have parent_id)
    # fullname : comment/discussion name
    # created_utc : time created
    # author._case_name : author id
    # body : stripped text (selftext for discussion)

    data = []

    footnote = "*Hello, users of CMV! This is a footnote from your moderators. We'd just like to remind you of a couple of things. Firstly, please remember to* ***[read through our rules](http://www.reddit.com/r/changemyview/wiki/rules)***. *If you see a comment that has broken one, it is more effective to report it than downvote it. Speaking of which,* ***[downvotes don't change views](http://www.reddit.com/r/changemyview/wiki/guidelines#wiki_upvoting.2Fdownvoting)****! If you are thinking about submitting a CMV yourself, please have a look through our* ***[popular topics wiki](http://www.reddit.com/r/changemyview/wiki/populartopics)*** *first. Any questions or concerns? Feel free to* ***[message us](http://www.reddit.com/message/compose?to=/r/changemyview)***. *Happy CMVing!*"

    # main post
    data.append({'parent': 0,
                 'id': discussion.fullname,
                 'utc': discussion.created_utc,
                 'author': discussion.author._case_name,
                 'text': discussion.selftext.replace(footnote, '')})

    def traverse_comments(comments):
        for comment in comments:
            data.append({'parent': comment.parent_id,
                         'id': comment.fullname,
                         'utc': comment.created_utc,
                         'author': 'None' if comment.author is None else comment.author._case_name,
                         'text': comment.body})
            traverse_comments(comment.replies)

    traverse_comments(discussion.comments)

    with open(os.path.join(folder, (link.split('/')[-2] if link.endswith('/') else link.split('/')[-1])) + ".json",
              'w+') as f:
        f.write(json.dumps(data, indent=2, sort_keys=True))


cmv_list = ['https://www.reddit.com/r/todayilearned/comments/56xqeq/til_baby_otters_cant_swim_at_first_but_their/']

for i, cmv in enumerate(cmv_list):
    time.sleep(1)
    get_discussion(cmv, './')
    print('[' + ('x' * int((i + 1) / len(cmv_list) * 20)) + ('-' * int(20 - (i + 1) / len(cmv_list) * 20)) + ']',
          end='\r')

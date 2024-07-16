from tables import (IsDescription,
                    StringCol,
                    Int32Col,
                    UInt32Col,
                    UInt8Col,
                    UInt16Col,
                    Time32Col)
import tables as tb
import xml.etree.ElementTree as ET
import string
from datetime import datetime
import numpy

CREATION_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
CREATION_TIME_FORMAT_VOTE = "%Y-%m-%d"


class User(IsDescription):
    id = Int32Col()
    name = StringCol(16)
    age = UInt8Col()
    reputation = UInt16Col()
    creation_date = Time32Col()
    views = UInt16Col()
    upvotes = UInt16Col()
    downvotes = UInt16Col()


class Vote(IsDescription):
    id = Int32Col()
    post_id = Int32Col()
    vote_type_id = UInt8Col()
    creation_date = Time32Col()


class Post(IsDescription):
    id = Int32Col()
    post_type_id = UInt8Col()
    parent_id = Int32Col()
    accepted_answer_id = Int32Col()
    creation_date = Time32Col()
    score = UInt32Col()
    viewcount = UInt32Col()
    owner_user_id = Int32Col()
    answer_count = UInt16Col()
    favorite_count = UInt16Col()
    tags = StringCol(itemsize=25, shape=(5,))


# event_type:
# 0 = user
# 1 = vote
# 2 = post
class Event(IsDescription):
    id = Int32Col()
    timestamp = Time32Col()
    event_type = UInt8Col()
    event_id = Int32Col()


def fast_iter(xml, table, func):
    tree_iter = ET.iterparse(xml, events=('end', 'start'))
    tree_iter = iter(tree_iter)
    occurence = 0
    event, root = tree_iter.next()
    for event, elem in tree_iter:
        if event == "end" and elem.tag == "row":
            func(elem, event, table)
            root.clear()
        occurence += 1
        if occurence > 100000:
            occurence = 0
            table.flush()
            return
    del tree_iter


def process_users(e, event, table):
    user = table.row
    user['id'] = e.get("Id", -111)
    user['name'] = e.get("DisplayName").encode('ascii', 'ignore')
    user['age'] = e.get("Age", -1)
    user['reputation'] = e.get("Reputation", 0)
    user['creation_date'] = datetime.strptime(
        e.get("CreationDate"),
        CREATION_TIME_FORMAT).strftime("%s")
    user['views'] = e.get("Views", 0)
    user['upvotes'] = e.get("UpVotes", 0)
    user['downvotes'] = e.get("DownVotes", 0)
    user.append()


def process_posts(e, event, table):
    post = table.row
    post['id'] = e.get("Id", -111)
    post['post_type_id'] = e.get("PostTypeId", 0)
    post['parent_id'] = e.get("ParentId", -111)
    post['accepted_answer_id'] = e.get("AcceptedAnswerId", -111)
    post['creation_date'] = datetime.strptime(
        e.get("CreationDate"),
        CREATION_TIME_FORMAT).strftime("%s")
    post['score'] = e.get("Score", 0)
    viewcount = e.get("ViewCount", 0)
    if (viewcount) == '':
        viewcount = 0
    post['viewcount'] = viewcount
    post['owner_user_id'] = e.get("OwnerUserId", -111)
    post['answer_count'] = e.get("AnswerCount", 0)
    post['favorite_count'] = e.get("FavoriteCount", 0)
    # Parse tags
    s = e.get("Tags", "")
    num = 0
    i = 0
    arS = []
    while True:
        i = string.find(s, '<', i)
        if i == -1:
            break
        end = string.find(s, '>', i)
        # Add the tag into the tag array
        arS.append(s[i+1:end])
        # Skip past the gt
        i = end + 1
        num += 1
    while len(arS) < 5:
        arS.append('')
    post['tags'] = numpy.array(arS)
    post.append()


def process_votes(elem, event, table):
    vote = table.row
    vote['id'] = elem.get("Id")
    vote['post_id'] = elem.get("PostId")
    vote['vote_type_id'] = elem.get("VoteTypeId")
    vote['creation_date'] = datetime.strptime(
        elem.get("CreationDate"),
        CREATION_TIME_FORMAT_VOTE).strftime("%s")
    vote.append()


def add_events(event, table, event_type, uid):
    table_id = table.cols.id
    table_date = table.cols.creation_date
    for i in xrange(len(table)):
        event['id'] = uid
        event['timestamp'] = table_date[i]
        event['event_type'] = event_type  # user type
        event['event_id'] = table_id[i]
        event.append()
        uid += 1
    return uid


def calculate_events(event_table, user_table, post_table, vote_table):
    event = event_table.row
    uid = 0
    uid = add_events(event, user_table, 0, uid)
    uid = add_events(event, post_table, 2, uid)
    uid = add_events(event, vote_table, 1, uid)


def preprocess_to_hdf5(directory):
    h5file = tb.openFile("overflow.h5",
                         mode="w",
                         title="InfoOverflow Visualization Data ")
    user_table = h5file.createTable("/", 'users', User, 'User information')
    post_table = h5file.createTable("/", 'posts', Post, 'Post information')
    vote_table = h5file.createTable("/", 'votes', Vote, 'Vote information')
    event_table = h5file.createTable("/", 'events', Event,
                                     'Summation of all other tables')
    event_table.cols.timestamp.createCSIndex()
    print "Start processing users"
    fast_iter(directory + "/users.xml", user_table, process_users)
    user_table.flush()
    print "Finished processing users"
    print "Start processing posts"
    fast_iter(directory + "/posts.xml", post_table, process_posts)
    post_table.flush()
    print "Finished processing posts"
    print "Start processing votes"
    fast_iter(directory + "/votes.xml", vote_table, process_votes)
    vote_table.flush()
    print "Finished processing votes"
    print "Calculating events..."
    calculate_events(event_table, user_table, post_table, vote_table)
    event_table.flush()
    print "Finished calculating events"
    h5file.close()
    print "Finished preprocessing, data stored in overflow.h5"

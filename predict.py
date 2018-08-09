# General IMPORTS --------------------------------------------------------------------------------------------------#
import re
import os
import sys
import time
import pickle
import tweepy
import operator
import subprocess
import collections
from datetime import datetime

# NLTK IMPORTS -----------------------------------------------------------------------------------------------------#
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# PYSPARK PATH SETUP AND IMPORTS -----------------------------------------------------------------------------------#
os.environ['SPARK_HOME'] = "/Users/path/to/spark-1.6.1-bin-hadoop2.6"  # Path to source folder

# Append pyspark  to Python Path
sys.path.append("/Users/path/to/spark-1.6.1-bin-hadoop2.6/python")
sys.path.append("/Users/path/to/spark-1.6.1-bin-hadoop2.6/python/lib/py4j-0.9-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import Row
    from pyspark.sql import SQLContext
    from pyspark.mllib.linalg import SparseVector
    from pyspark.accumulators import AccumulatorParam
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


# GLOBAL VARIABLES -------------------------------------------------------------------------------------------------#

# Spark Context object
sc = SparkContext('local[4]', 'EU_Tweet_Sentiment_Analyser')        # Instantiate a SparkContext object
sqlContext = SQLContext(sc)                                         # Instantiate a sqlContext object

# Twitter API credentials

# APP_0
consumer_key_0 = "to be completed"
consumer_secret_0 = "to be completed"
access_key_0 = "to be completed"
access_secret_0 = "to be completed"

# APP_1
consumer_key_1 = "to be completed"
consumer_secret_1 = "to be completed"
access_key_1 = "to be completed"
access_secret_1 = "to be completed"

consumer_key = [consumer_key_0, consumer_key_1]
consumer_secret = [consumer_secret_0, consumer_secret_1]
access_key = [access_key_0, access_key_1]
access_secret = [access_secret_0, access_secret_1]


# -- SUB-FUNCTIONS ----------------------------------------------------------------------------------------------------
def filter_tweet(tweet):

    tweet = re.sub("(htt.* ?)", " ", tweet)    # captures all occurences of "http" followed or not followed by a space
    tweet = re.sub("(www.* ?)", " ", tweet)    # same
    tweet = re.sub("RT ", "", tweet)           # removes leading RTs
    tweet = re.sub("([@|#].*?)", " ", tweet)   # removes handles/hastags
    tweet = re.sub("([/| |'|(|+|-]\d+[\.| |/|;|?|%|:|,|'|(|)|+|-]\d*.?)", " ", tweet)  # removes floating point numbers
    tweet = re.sub("( \d+.? )", " ", tweet)     # removes numbers!

    # further abnormalities
    tweet = re.sub("([ |.]\d+[-|\.]\d*.? )", " ", tweet)
    tweet = re.sub("(\d+-\d+.?)", "", tweet)

    return tweet


def lemmatize(tweet_words):

    # Instantiate lemmatization-object
    wordnet_lemmatizer = WordNetLemmatizer()

    # Lemmatize: lowering stinr is necessary in this step - unfortunately it strips words of semantics
    # (e.g CAPITALS are usually used for shouting!)
    for i in range(len(tweet_words)):
        tweet_words[i] = wordnet_lemmatizer.lemmatize(tweet_words[i].lower())

    return tweet_words


def negation_tokenizer(tweet_words):

    # regex to match negation tokens
    negation_re = re.compile("""(?x)(?:^(?:never|no|nothing|nowhere|
    noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't""")

    alter_re = re.compile("""(?x)(?:^(?:but|however|nevertheless|still|though|tho|yet)$)""")

    neg_tweed_words = []
    append_neg = False  # stores whether to add "_NEG"

    for token in tweet_words:

        # If append_neg is False
        if append_neg == False:

            # Check if the current token is a negation
            if negation_re.match(token):
                append_neg = True

        # but if a negation has been previously identified, check if this is an  alteration
        elif alter_re.match(token):
            append_neg = False

        # or if another negation appears
        elif negation_re.match(token):
            append_neg = False

        # and if not then append the suffix
        else:
            token += "_NEG"

        # append the new token in the return list
        neg_tweed_words.append(token)

    return neg_tweed_words


def filter_stop_words(tweet_words):

    return [word for word in tweet_words if word not in stop_words_bv.value]


def tf(tokens):
    """ Compute TF
    Args:
        tokens (list of str): input list of tokens from tokenize
    Returns:
        dictionary: a dictionary of tokens to its TF values
    """
    counts = {}
    length = len(tokens)
    for t in tokens:
        counts.setdefault(t, 0.0)
        counts[t] += 1
    return {t: counts[t] / length for t in counts}


def tfidf(tokens, idfs):
    """ Compute TF-IDF
    :param tokens: tokens (list of str): input list of tokens from tokenize.
    :param idfs: record to IDF value.
    :return: dictionary: a dictionary of records to TF-IDF values
    """

    tfidfs = {}
    tfs = tf(tokens)
    for t in tfs:
        tfidfs.setdefault(t, 0.0)
        tfidfs[t] += 1
    return {t: tfs[t] * idfs[t] for t in tfidfs if t in idfs}


def featurize(tokens_kv):
    """
    :param tokens_kv: list of tuples of the form (word, tf-idf score)
    :param dictionary: list of n words
    :return: sparse_vector of size n
    """

    # MUST sort tokens_kv by key
    tokens_kv = collections.OrderedDict(sorted(tokens_kv.items()))

    vector_size = len(Dictionary_BV.value)
    non_zero_indexes = []
    index_tfidf_values = []

    for key, value in tokens_kv.iteritems():
        index = 0
        for word in Dictionary_BV.value:
            if key == word:
                non_zero_indexes.append(index)
                index_tfidf_values.append(value)
            index += 1

    return SparseVector(vector_size, non_zero_indexes, index_tfidf_values)


def get_all_tweets_for_prediction(screen_name, sameModel, tokenizer, C_KEY):

    # Set up local api variable
    auth = tweepy.OAuthHandler(consumer_key[C_KEY], consumer_secret[C_KEY])
    auth.set_access_token(access_key[C_KEY], access_secret[C_KEY])
    api = tweepy.API(auth)

    # Twitter only allows access to a users most recent 3240 tweets with this method
    print("Accumulating tweets for %s:" % screen_name)

    # make a single request for most recent tweets (200 is the maximum allowed count)
    resume = True
    while resume:
        resume = False
        try:
            tweets = api.user_timeline(screen_name=screen_name, count=200)
        except tweepy.TweepError, e:
            if e.reason == "Not authorized.":
                print("Stop exception: %s." % e.reason)
                print "------------------------"
                return 0, 0, C_KEY
            elif e.reason == "Sorry, that page does not exist.":
                print("Stop exception: %s." % e.reason)
                print "------------------------"
                return 0, 0, C_KEY
            else:
                print "Tweepy Exception: %s" % e.reason
                print "Rate limit exceeded--> Switching APP keys. Date/Time: %s." % str(datetime.now())

                C_KEY = (C_KEY + 1) % 2

                # authorize twitter, initialize tweepy
                auth = tweepy.OAuthHandler(consumer_key[C_KEY], consumer_secret[C_KEY])
                auth.set_access_token(access_key[C_KEY], access_secret[C_KEY])
                api = tweepy.API(auth)

                raw_input("Changed keys. Press any key to continue...")
                resume = True
        except StopIteration:
            print 'ERROR: Failed because of %s' % e.reason
            return 0, 0, C_KEY

    # If tweets have been accumulated:
    if len(tweets) > 0:
        # filter out non-EU-related tweets
        eu_tweets = []
        list_words = ['European union', 'European Union', 'european union', 'EUROPEAN UNION',
                      'Brexit', 'brexit', 'BREXIT',
                      'euref', 'EUREF', 'euRef', 'eu_ref', 'EUref',
                      'leaveeu', 'leave_eu', 'leaveEU', 'leaveEu',
                      'borisvsdave', 'BorisVsDave',
                      'StrongerI', 'strongerI', 'strongeri', 'strongerI',
                      'votestay', 'vote_stay', 'voteStay',
                      'votein', 'voteout', 'voteIn', 'voteOut', 'vote_In', 'vote_Out',
                      'referendum', 'Referendum', 'REFERENDUM',
                       ' EU ', ' eu ']

        for tweet in tweets:
            for word in list_words:
                if tweet.text.encode("utf-8").find(word) != -1:
                    eu_tweets.append(tweet.text.encode("utf-8"))
                    break

        if len(eu_tweets) > 0:
            # TOKENIZE COLLECTED TWEETS -----------------------------------------------------------------------------#
            eu_tweets_sample_RDD = sc.parallelize(eu_tweets, 4)

            # Tokenize through Spark Transformations
            eu_wordsByTweet_RDD = (eu_tweets_sample_RDD.map(lambda tweet: tweet.decode("ascii", "ignore").encode("ascii"))
                            .map(filter_tweet)
                            .map(tokenizer.tokenize)
                            .map(lemmatize)
                            .map(filter_stop_words)
                            .map(negation_tokenizer)
                            .cache())

            # SHOW WORDS BY TWEET -----------------------------------------------------------------------------------#
            sample_RDD = eu_wordsByTweet_RDD.collect()
            print '\n'.join(map(lambda x: '{0}'.format(x), sample_RDD))

            # COMPUTE TF-IDF SCORES ---------------------------------------------------------------------------------#
            TFsIDFs_Vector_Weights_RDDs = (eu_wordsByTweet_RDD
                                           .map(lambda tokens: (tfidf(tokens, IDFS_weights_BV.value)))
                                           .cache())

            # CREATE FEATURE VECTORS --------------------------------------------------------------------------------#
            vectors = TFsIDFs_Vector_Weights_RDDs.map(lambda (tokens): featurize(tokens))

            # MAKE PREDICTIONS
            predictions = vectors.map(lambda v: sameModel.predict(v))
            prop = predictions.filter(lambda label: label == 1).count()
            opp = predictions.filter(lambda label: label == 0).count()
            return prop, opp, C_KEY
        else:
            print "No eu-related tweets..."
            return 0, 0, C_KEY
    else:
        print "No tweets to collect..."
        return 0, 0, C_KEY

    print "------------------------"


def get_friends(user_handle, api):

    # Create empty users array with user's scree_name as first input
    users = [user_handle]

    # Get user's friends
    for user in api.friends(screen_name=user_handle, count=200):
        print user.screen_name
        users.append(user.screen_name)

    return users


# -- MAIN CODE --------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Choose App_0
    C_KEY = 1

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key[C_KEY], consumer_secret[C_KEY])
    auth.set_access_token(access_key[C_KEY], access_secret[C_KEY])
    api = tweepy.API(auth)

    # Instantiate tokenizer-object
    tokenizer = RegexpTokenizer(r'(\w+)')  # Removes "RT @" and keeps only [a-zA-Z0-9] and '_'

    # Stop_Words set
    delete = {'should', 'don', 'again', 'not'}  # Make NLTK's stopwords list more sentiment-aware
    stopWords = set(stopwords.words('english')).difference(delete)
    stop_words_bv = sc.broadcast(stopWords)

    # Read dictionary_RDD_IDFs_Weights dict back from the file
    pkl_file = open('/Users/path/to/dictionary_RDD_IDFs_Weights.pkl',
                    'rb')
    dictionary_RDD_IDFs_Weights = pickle.load(pkl_file)
    pkl_file.close()

    # Create dictionary_RDD_IDFs_Weights broadcast variable
    IDFS_weights_BV = sc.broadcast(dictionary_RDD_IDFs_Weights)

    # Read dictionary of the N first words with respect to the most important ones (sorting based on IDFs)
    sorted_dict = sorted(dictionary_RDD_IDFs_Weights.items(), key=operator.itemgetter(1))

    # Set to max of N words for corresponding number of features for which the model is trained
    Dictionary = []
    for key, value in sorted_dict:
        Dictionary.append(key)

    print len(Dictionary)

    # Create a broadcast variable for the Dictionary
    Dictionary_BV = sc.broadcast(sorted(Dictionary))

    # Load Naive Bayes Model
    model_path = "/Users/path/to/twitter_analytics/NB_model"
    sameModel = NaiveBayesModel.load(sc, model_path)

    # Start intro Video -  make sure to first run "chmod a+x play.sh" otherwise --> permission denied exception
    video = "Users:path:to:vids:intro.mp4"
    video_1 = subprocess.Popen("osascript runner.scpt " + "'" + video + "'", shell=True)

    # Get user twitter-handle
    x = int(input("Do you have a twitter account? \n(1) Yes \n(2) No \nYour choice: "))
    if x == 1:
        user_handle = raw_input("Please provide user twitter handle: ")
        friends = get_friends(user_handle, api)

        # Collect EU related tweets and predict position for each
        poss = 0
        negs = 0

        timeout = time.time() + 60 * 5  # 5 minutes and a half from now
        for friend in friends:
            (prop, opp, C_KEY) = get_all_tweets_for_prediction(
                friend, sameModel, tokenizer, C_KEY)
            poss += prop
            negs += opp
            print("Tweets count --------------------------")
            print("Number of stay tweets: " + str(poss) + ".")
            print("Number of leave tweets: " + str(negs) + ".")

            if time.time() > timeout:
                break
    else:
        poss = 1
        negs = 1

    # QUERIES -------------------------------------------------------------------------------------------------------#
    print('------------------------------------- QUESTIONAIRE -------------------------------------')
    print "Please complete the following questionaire: "

    pos_q = 0
    neg_q = 0

    # Newspaper -------------------------------------------------
    scores = [82, -8, -18, -42, -46, -54, 0]
    choice_0 = int(input('You read: '
                         '\n(1) The guardian'
                         '\n(2) Mirror/Record'
                         '\n(3) Telegraph'
                         '\n(4) Sun'
                         '\n(5) Mail'
                         '\n(6) Express'
                         '\n(7) None of the above'
                         '\nYour choice: '))

    if scores[choice_0 - 1] > 0:
        pos_q += scores[choice_0 - 1]
    else:
        neg_q += -scores[choice_0 - 1]

    # AGE -------------------------------------------------------
    age = int(input('Age:'))
    if 18 <= age <= 29:
        pos_q += 46
    elif 30 <= age <= 39:
        pos_q += 24
    elif 40 <= age <= 49:
        neg_q += 4
    elif 50 <= age <= 59:
        neg_q += 10
    else:
        neg_q += 26

    # UNIVERSITY ------------------------------------------------
    choice_1 = int(input('University graduate?'
                         '\n(1) Yes'
                         '\n(2) No'
                         '\nYour choice: '))
    if choice_1 == 1:
        pos_q += 40

    # SEX -------------------------------------------------------
    choice_2 = int(input('Sex?'
                         '\n(1) Male'
                         '\n(2) Female'
                         '\nYour choice: '))
    if choice_2 == 2:
        pos_q += 2

    # PERMANENT ADDRESS IN -------------------------------------
    scores = [30, 26, 16, 10, 2, -2, -2, -2, -10, -10, -12, -18, 0]
    choice_3 = int(input('You are originally from:'
                         '\n(1) Nortern Ireland'
                         '\n(2) Scotland'
                         '\n(3) London'
                         '\n(4) Wales'
                         '\n(5) North East Egland'
                         '\n(6) North West Egland'
                         '\n(7) South East Egland'
                         '\n(8) South West Egland'
                         '\n(9) Yorkshire & Humberside'
                         '\n(10) East Anglia'
                         '\n(11) West Midlands'
                         '\n(12) East Midlands'
                         '\n(13) Other'
                         '\nYour choice: '))

    if scores[choice_3 - 1] > 0:
        pos_q += scores[choice_3 - 1]
    else:
        neg_q += -scores[choice_3 - 1]

    # CALCULATE DEMOGRAPHIC SCORES ----------------------------------------------------------------------------------#

    print('--------------------------------------- RESULTS ---------------------------------------')

    total_Demographic_units = pos_q + neg_q
    pos_q_percent = pos_q/float(total_Demographic_units) * 100
    neg_q_percent = neg_q / float(total_Demographic_units) * 100

    print "Stay demographic score: " + str(pos_q) + ". Percentage: " \
          + str(pos_q/float(total_Demographic_units) * 100) + "%"
    print "Leave demographic score: " + str(neg_q) + ". Percentage: " \
          + str(neg_q / float(total_Demographic_units) * 100) + "%"

    # CALCULATE TWEET SENTIMENT ANALYSIS SCORES ---------------------------------------------------------------------#
    total_eu_tweets = poss + negs
    pos_percent = poss/float(total_eu_tweets) * 100
    neg_percent = negs/float(total_eu_tweets) * 100

    print "Stay tweets: " + str(poss) + ". Percentage: " + str(poss/float(total_eu_tweets) * 100) + "%"
    print "Leave tweets: " + str(negs) + ". Percentage: " + str(negs/float(total_eu_tweets) * 100) + "%"

    # PRESENT FINAL RESULTS -----------------------------------------------------------------------------------------#
    video_1.kill()
    video = 'Users:path:to:vids:processing.mp4'
    video_2 = subprocess.Popen("osascript runner.scpt " + "'" + video + "'", shell=True)
    time.sleep(3)

    final_pos = pos_q_percent * (25/float(100)) + pos_percent * (75/float(100))
    final_neg = neg_q_percent * (25 / float(100)) + neg_percent * (75 / float(100))

    print "----------------------------------------------"
    print "Final percentages:"
    print "Stay: " + str(final_pos) + "%"
    print "Leave: " + str(final_neg) + "%"

    if final_pos > final_neg and (final_pos - final_neg) > 20:
        video_2.kill()
        video = 'Users:path:to:vids::vids:in.mp4'
        video_3 = subprocess.Popen("osascript runner.scpt " + "'" + video + "'", shell=True)
    elif final_pos < final_neg and (final_neg - final_pos) > 20:
        video_2.kill()
        video = 'Users:path:to:vids:vids:out.mp4'
        video_4 = subprocess.Popen("osascript runner.scpt " + "'" + video + "'", shell=True)
    else:
        video_2.kill()
        video = 'Users:path:to:vids:undec.mp4'
        video_5 = subprocess.Popen("osascript runner.scpt " + "'" + video + "'", shell=True)

# END OF FILE -------------------------------------------------------------------------------------------------------#

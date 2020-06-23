# General IMPORTS --------------------------------------------------------------------------------------------------#
import os
import re
import sys
import pickle
import pandas
import random
import itertools
import collections
import matplotlib.pyplot as plt


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
sc = SparkContext('local[4]', 'EU_Tweet_Sentiment_Analyser')        # Instantiate a SparkContext object
sqlContext = SQLContext(sc)                                         # Instantiate a sqlContext object


# SUB-FUNCTIONS-----------------------------------------------------------------------------------------------------#
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


def populate_with(number_of_tweets, tweets, label, selected_tweets, labels):

    for i in range(number_of_tweets):

        r = random.randint(0, len(tweets)-1)
        selected_tweets.append(tweets[r])

        if label == "STAY":
            labels.append(1)
        else:
            labels.append(0)

        del tweets[r]

    return selected_tweets, labels


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


def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus
    Returns:
        RDD: a RDD of (token, IDF value)
    """

    N = corpus.count()

    # The result of the next line will be a list with distinct tokens...
    unique_tokens = corpus.flatMap(lambda x: list(set(x)))  # No more records! FLATMAP --> unique_tokens is ONE SINGLE LIST
    token_count_pair_tuple = unique_tokens.map(lambda x: (x, 1))  # every element in the list will become a pair!
    token_sum_pair_tuple = token_count_pair_tuple.reduceByKey(lambda a, b: a + b)  # same elements in lists are aggregated
    return token_sum_pair_tuple.map(lambda x: (x[0], float(N) / x[1]))  # compute weight


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


def final_form_4_training(SVs, labels):
    """
    :param SVs: List of Sparse vectors.
    :param labels: List of labels
    :return: list of labeledpoint objects
    """

    to_train = []
    for i in range(len(labels)):
        to_train.append(LabeledPoint(labels[i], SVs[i]))
    return to_train


# MAIN -------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":

    # LOAD TWEETS --------------------------------------------------------------------------------------------------#

    # Load Corpus using Pandas
    dataFrame = pandas.read_csv('/Users/path/to/corpus.csv',
                                header=None,
                                names=['name', 'screen_name', 'id', 'created_at', 'text', 'label'])

    # Load Columns as Arrays (Notice: first element = column name)
    tweets = dataFrame['text']
    del tweets[0]
    labels = dataFrame['label']
    del labels[0]

    # CREATE TRAINING CORPUS / CROSS-VALIDATION SET / TEST SET  ----------------------------------------------------#

    # Instantiate Tweet RDDS
    labels_RDD = sc.parallelize(labels, 4)

    total_tweets = labels_RDD.count()
    print "Total tweets: %d" % total_tweets
    pos_tweets = labels_RDD.filter(lambda x: x == "Stay").count()
    print "Pos tweets: %d" % pos_tweets
    neg_tweets = pos_tweets = labels_RDD.filter(lambda x: x == "Leave").count()
    print "Neg tweets: %d" % pos_tweets

    # Break tweets between positive and negative
    pos_tweets = []
    neg_tweets = []
    for (tweet, label) in itertools.izip(tweets, labels):
        if label == "Stay":
            pos_tweets.append(tweet)
        else:
            neg_tweets.append(tweet)

    # Divide respectively to 85%-7.5%-7.5%
    training_no = int(min(len(pos_tweets), len(neg_tweets)) * 85 / 100)
    cross_validation_no = int(min(len(pos_tweets), len(neg_tweets)) * 7.5 / 100)
    test_no = min(len(pos_tweets), len(neg_tweets)) - training_no - cross_validation_no

    # Training Set
    training_set = []
    training_labels = []

    (training_set, training_labels) = populate_with(training_no, pos_tweets, "STAY", training_set, training_labels)
    (training_set, training_labels) = populate_with(training_no, neg_tweets, "LEAVE", training_set, training_labels)

    # Cross-Validation Set
    cross_validation_set = []
    cross_validation_labels = []

    (cross_validation_set, cross_validation_labels) = populate_with(cross_validation_no, pos_tweets, "STAY",
                                                                    cross_validation_set, cross_validation_labels)
    (cross_validation_set, cross_validation_labels) = populate_with(cross_validation_no, neg_tweets, "LEAVE",
                                                                    cross_validation_set, cross_validation_labels)
    # Test Set
    test_set = []
    test_labels = []

    (test_set, test_labels) = populate_with(cross_validation_no, pos_tweets, "STAY", test_set, test_labels)
    (test_set, test_labels) = populate_with(cross_validation_no, neg_tweets, "LEAVE",test_set, test_labels)

    # TOKENIZE TRAINING SET ----------------------------------------------------------------------------------------#

    # Instantiate Training RDD
    training_RDD = sc.parallelize(training_set, 4)

    # Instantiate tokenizer-object
    tokenizer = RegexpTokenizer(r'(\w+)')  # Removes "RT @" and keeps only [a-zA-Z0-9] and '_'

    # Stop_Words set
    delete = {'should', 'don', 'again', 'not'}  # Make NLTK's stopwords list more sentiment-aware
    stopWords = set(stopwords.words('english')).difference(delete)
    stop_words_bv = sc.broadcast(stopWords)

    # Tokenize through Spark Transformations
    wordsByTweet = (training_RDD.map(lambda tweet: tweet.decode("ascii", "ignore").encode("ascii"))
                    .map(filter_tweet)
                    .map(tokenizer.tokenize)
                    .map(lemmatize)
                    .map(filter_stop_words)
                    .map(negation_tokenizer)
                    .cache())

    # SHOW CORPUS -------------------------------------------------------------------------------------------------#
    corpus_RDD = wordsByTweet.collect()

    print '\n'.join(map(lambda x: '{0}'.format(x), corpus_RDD))

    # CREATE A DICTIONARY ------------------------------------------------------------------------------------------#
    print("---------------------------------------------------------------------------------------------------------")
    raw_input("Produce TF-IDF scores...")

    dictionary_RDD_IDFs = idfs(wordsByTweet)
    unique_token_count = dictionary_RDD_IDFs.count()
    print 'There are %s unique tokens in the dataset.' % unique_token_count

    IDFS_Tokens_Sample = dictionary_RDD_IDFs.takeOrdered(25, lambda s: s[1])
    print("This is a dictionary sample of 25 words:")
    print '\n'.join(map(lambda (token, idf_score): '{0}: {1}'.format(token, idf_score), IDFS_Tokens_Sample))

    # Create a broadcast variable for the weighted dictionary (sorted)
    dictionary_RDD_IDFs_Weights = dictionary_RDD_IDFs.sortBy(lambda (token, score): score).collectAsMap()
    IDFS_weights_BV = sc.broadcast(dictionary_RDD_IDFs_Weights)

    # Write IDFS_weights_BV as python dictionary to a file
    output = open('/Users/path/to/dictionary_RDD_IDFs_Weights.pkl', 'wb')
    pickle.dump(dictionary_RDD_IDFs_Weights, output)
    output.close()

    print IDFS_weights_BV.value

    # CREATE A HISTOGRAM -------------------------------------------------------------------------------------------#
    print("--------------------------------------------------------------------------------------------------------")
    raw_input("Create an IDF-scores histogram...")
    IDFs_values = dictionary_RDD_IDFs.map(lambda s: s[1]).collect()
    fig = plt.figure(figsize=(8, 3))
    plt.hist(IDFs_values, 50, log=True)
    plt.show()

    # PRE-COMPUTE TF-IDF WEIGHTS: Build Weight Vectors -------------------------------------------------------------#
    print("--------------------------------------------------------------------------------------------------------")
    raw_input("Produce the TF-IDF scores...")
    TFsIDFs_Vector_Weights_RDDs = wordsByTweet.map(lambda tokens: (tfidf(tokens, IDFS_weights_BV.value))).cache()
    print '\n'.join(map(lambda words: '{0}'.format(words), TFsIDFs_Vector_Weights_RDDs.take(10)))

    # BEGIN CREATION OF FEATURE VECTOR ------------------------------------------------------------------------------#
    print("--------------------------------------------------------------------------------------------------------")
    raw_input("Create an ordered dictionary for feature extraction...")

    # Create an ordered dictionary of the N first words
    Dictionary = (dictionary_RDD_IDFs
                  .sortBy(lambda (token, score): score)
                  .map(lambda (token, score): token)
                  .collect())       # N = all-->collect(), otherwise use take(N)
    print("This is the complete dictionary, ordered based on idf scores:")
    print '\n'.join(map(lambda token: '{0}'.format(token), Dictionary))
    print("--------------------------------------------------------------------------------------------------------")

    # Create a broadcast variable for the Dictionary
    # Dictionary MUST be sorted. If not sparse-vectors in the featurize function will throw exception.
    Dictionary_Sorted = sorted(Dictionary)
    Dictionary_BV = sc.broadcast(Dictionary_Sorted)

    # Save ordered Dictionary
    output = open("/Users/path/to/Dictionary.txt", "wb")
    output.write("\n".join(map(lambda x: str(x), Dictionary_Sorted)))
    output.close()

    # Feature Extraction
    Training_Set_Vectors = (TFsIDFs_Vector_Weights_RDDs
                            .map(lambda (tokens): featurize(tokens))
                            .collect())

    # GENERATE LABELEDPOINT PARAMETER TO LOAD TO THE TRAIN METHOD ---------------------------------------------------#
    print("--------------------------------------------------------------------------------------------------------")
    raw_input("Generate the LabeledPoint parameter... ")
    labelled_training_set_RDD = sc.parallelize(final_form_4_training(Training_Set_Vectors, training_labels))

    # TRAIN MODEL ---------------------------------------------------------------------------------------------------#
    print("--------------------------------------------------------------------------------------------------------")
    raw_input("Train the model... ")
    model = NaiveBayes.train(labelled_training_set_RDD, 1.0)

    # CROSS-VALIDATE MODEL ------------------------ -----------------------------------------------------------------#
    print("--------------------------------------------------------------------------------------------------------")
    raw_input("Cross-Validate the model... ")
    # Instantiate Cross_Validation RDD
    CV_RDD = sc.parallelize(cross_validation_set, 4)

    # Tokenize through Spark Transformations
    CV_wordsByTweet = (CV_RDD.map(lambda tweet_2: tweet_2.decode("ascii", "ignore").encode("ascii"))
                       .map(filter_tweet)
                       .map(tokenizer.tokenize)
                       .map(lemmatize)
                       .map(filter_stop_words)
                       .map(negation_tokenizer)
                       .cache())
    print("Cross Validation set loaded and tokenised... ")

    # Compute TF-IDF scores
    raw_input("Produce the TF-IDF scores for Cross-Validation Set...")
    CV_TFsIDFs_Vector_Weights_RDDs = (CV_wordsByTweet
                                      .map(lambda tokens: (tfidf(tokens, IDFS_weights_BV.value)))
                                      .cache())

    # Feature Extraction
    raw_input("Extract Features for Cross-Validation Set...")
    CV_Set_Vectors = (CV_TFsIDFs_Vector_Weights_RDDs
                      .map(lambda (tokens): featurize(tokens))
                      .collect())

    # Generate labelledppoint parameter...
    raw_input("Generate the LabeledPoint parameter... ")
    labelled_CV_set_RDD = sc.parallelize(final_form_4_training(CV_Set_Vectors, cross_validation_labels))

    # Compute Accuracy
    print("--------------------------------------------------------------------------------------------------------")
    raw_input("Compute model CV-accuracy...")
    predictionAndLabel = labelled_CV_set_RDD.map(lambda x: (model.predict(x.features), x.label))
    accuracy = 100.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / labelled_CV_set_RDD.count()

    print ("Model Accuracy is: {0:.2f}%".format(accuracy))

    print("--------------------------------------------------------------------------------------------------------")

    # TEST MODEL ---------------------------------------------------------------------------------------------------#
    print("--------------------------------------------------------------------------------------------------------")
    raw_input("Test the model... ")
    # Instantiate Cross_Validation RDD
    test_RDD = sc.parallelize(test_set, 4)

    # Tokenize through Spark Transformations
    test_wordsByTweet = (test_RDD.map(lambda tweet_4: tweet_4.decode("ascii", "ignore").encode("ascii"))
                         .map(filter_tweet)
                         .map(tokenizer.tokenize)
                         .map(lemmatize)
                         .map(filter_stop_words)
                         .map(negation_tokenizer)
                         .cache())
    print("Test set loaded and tokenised... ")

    # Compute TF-IDF scores
    raw_input("Produce the TF-IDF scores for Test Set...")
    test_TFsIDFs_Vector_Weights_RDDs = (test_wordsByTweet
                                        .map(lambda tokens: (tfidf(tokens, IDFS_weights_BV.value)))
                                        .cache())

    # Feature Extraction
    raw_input("Extract Features for Cross-Validation Set...")
    test_Set_Vectors = (test_TFsIDFs_Vector_Weights_RDDs
                        .map(lambda (tokens): featurize(tokens))
                        .collect())

    # Generate labelledppoint parameter...
    raw_input("Generate the LabeledPoint parameter... ")
    labelled_test_set_RDD = sc.parallelize(final_form_4_training(test_Set_Vectors, test_labels))

    # Compute Accuracy
    print("--------------------------------------------------------------------------------------------------------")
    raw_input("Compute model Test-accuracy...")
    predictionAndLabel = labelled_test_set_RDD.map(lambda x: (model.predict(x.features), x.label))
    accuracy = 100.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / labelled_test_set_RDD.count()

    print ("Model Accuracy is: {0:.2f}%".format(accuracy))

    print("--------------------------------------------------------------------------------------------------------")

    # SAVE MODEL ----------------------------------------------------------------------------------------------------#
    model_path = "/Users/path/to/twitter_analytics/NB_model"
    model.save(sc, model_path)

# END OF FILE -------------------------------------------------------------------------------------------------------#

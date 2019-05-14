
# Mining Public Opinion on Twitterr

This post elaborates on a simple opinion mining POC I was recently asked to develop as part of an internal company event. The general idea is to combine supervised machine learning with sentiment analysis to develop an opinion mining tool. The developed tool is based on python and builds on the nltk library, tweepy, and spark. For readability, the post is broken into short subsections.

The basic idea
At the core of this application lies a Naïve Bayes prediction model. The intention is to train the model to identify the polarity of a Brexit related tweet—leave or stay—and then use the model to classify person A's tweets as well as the tweets of the people that person A is following on twitter, in order to effectively classify person A.

## What is a Naïve Bayes classifier?
In brief, it is a classifier that applies Bayes’ Theorem for identifying a tweet’s class participation (e.g. positive/negative), based on the number of previously classified tweets of ‘similar type’. Simply put:

![Bayes Theorem](https://wikimedia.org/api/rest_v1/media/math/render/svg/882f4d436804a1d0dc76ca047bb9318b60f8e26f)
 where "A: label" and "B: tweet", which is interpreted as:
* P(label/tweet): The probability of a label given a tweet (the result of which we want to compute).
* P(tweet/label): The probability of a tweet given a label (which is based on previously gathered information).
* P(label): The probability of the label (which is independent of all other probabilities, e.g. 50% in the case of two labels).
* P(tweet): The probability of the tweet (also independent from all other probabilities).

## Why Naïve Bayes?
The “Naïve” Bayes' Classifier is characterised as such because it assumes, in our case, that a tweet’s tokens are independent random events, which greatly simplifies calculations, speeding up the prediction process. Some argue that this may lead to a reduced accuracy, but this is very rarely the case (this assumption is context dependent, it could be that a model requires that tokens are perceived as dependent random events in order to achieve accurate predictions), while, in addition, the model is extremely fast.
Step one
The first step of this process is training the model. To do so I had to first:

1.     accumulate a corpus of Brexit related tweets, and;
2.     label them as either “Leave” or “Stay”.

To that end a BBC article that listed close to 440 MPs and where they stand proved to be very helpful.
Getting a list of @MP-handles using the twitter API
Using the twitter API was an essential part of this project. Everyone who intends to do so must first create a new Twitter Application by following the steps here.

To my good fortune, a twitter-handles list with all the UK MPs with active twitter accounts subscribed to it, was available on twitter and I was able to get my hands on it. I used python for the whole project. Below you can see the full code I used to download the list.
```python
import tweepy
import csv

# -- Twitter API credentials --
consumer_key = "<Your Consumer Key>"
consumer_secret = "<Your Consumer Secret>"
access_key = "<Your Access Key>"
access_secret = "<Your Access Secret>j"

# authorize twitter app
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


# -- SUB-FUNCTIONS --
def get_list_members(owner_screen_name, slug):
    twitter_list = api.list_members(owner_screen_name=owner_screen_name, slug=slug, count=700)

    # transform the tweepy tweets into a 2D array that will populate the csv
    users = [[user.name.encode("utf-8")
                  .replace(" MP", "")
                  .replace(" MEP", "")
                  .replace(" MdEP", "")
                  .replace(" ASE/MEP", ""),
              user.screen_name.encode("utf-8")]
             for user in twitter_list]

    # write the csv
    with open('%s_list.csv' % slug, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Screen_Name"])
        writer.writerows(users)

    return twitter_list

# -- MAIN CODE --
if __name__ == '__main__':

    # UK MPs owner_screen_name:"tweetminster" slug:"ukmps"
    print("Instantiating UK-MPs's List...")
    ukMPs = get_list_members("tweetminster", "ukmps")
    print("List created successfully...
```

Once I had both lists in my possession (BBC list and twitter handles), it was a simple matter of matching MP names with their respective handles. In order to effectively do so, I had to strip all the prefixes from the registered MP names as they appear in the twitter list I downloaded, hence the use of:

replace(" prefix", "")
Note that for running the .get_list_members() method you will need both the owner screen name as well as the list slug. The latter is this:

https://twitter.com/tweetminster/lists/ukmps/members?lang=en-gb
## Accumulating the corpus
This part was generally easy. Once more, using the tweepy library, I was able to iterate through the list of labelled MPs, get their handles, access their timelines and download as much as 3240 (maximum number based on twitter restrictions) of their latest tweets. For most accounts, this number was more than enough to collect their full twitter history. A couple of things are worth noting however.

The main method used here is:
new_tweets = api.user_timeline(screen_name=screen_name, count=200)
The count parameter takes a maximum value of 200 which means that one can only collect as many as 200 tweets in one go. If you want more you need to iterate this method for as many times as you are allowed (that is, while not exceeding the maximum number of 3240 tweets) over and over again, collecting tweets from where you left things from the last collection. This means that you will have to identify the last tweet you collected in the previous call and use it as a parameter in your next call. This is done with:
```python
oldest = new_tweets [-1].id – 1
```
Hence, every next iterative call should be of the form:
```python
new_tweets=api.user_timeline(screen_name=screen_name,count=200,max_id=oldest)
```
Needless to say, you need to store every batch of 200 tweets in a separate list prior to iterating.

In addition to this, you need to encapsulate all your calls within try:catch statements to handle exceptions. There are generally 3 exceptions you need to worry about:

1.     Not authorised: Some users won’t allow you to collect their tweets.
2.     Page does not exist: It may simply be the case that you have misspelled a twitter handle.
3.     Rate limit exception: Twitter generally allows uninterrupted collection of tweets for roughly 10 minutes. You should be able to download more than 50k tweets within that time.  After that, you will get a “Rate limit exceeded” exception and you will be blocked from accessing any more twitter timelines for as long as 15 minutes (the temporary ban is consumer key based so there is a way to bypass it—I’ll get to this later in the article).

Furthermore, in case you are only concerned with collecting tweets up to a given point in the past, e.g. not before 2015, a simple way is to break your loop if the year of the last tweet collected is smaller than that of your choice. To get the year of creation of the last tweet collected you will need to run the code bellow:
```python
year = int(str(new_tweets[-1].created_at)[0:4])
```
where [0:4] simply refers to the first 4 characters of the `.created_at` value which refer to the year.

Finally, for every batch of 200 tweets collected, I had to filter out the non-brexit-related ones, encode every tweet based on “utf-8” and label it in accordance to the label associated with the MP whose account I accessed. I initially used relatively uncommon terms like:
```python
list_words = ['European union', 'European Union', 'european union', 'EUROPEAN UNION',
              'Brexit', 'brexit', 'BREXIT',
              'euref', 'EUREF', 'euRef', 'eu_ref', 'EUref',
              'leaveeu', 'leave_eu', 'leaveEU', 'leaveEu',
              'borisvsdave', 'BorisVsDave',
              'StrongerI', 'strongerI', 'strongeri', 'strongerI',
              'votestay', 'vote_stay', 'voteStay',
              'votein', 'voteout', 'voteIn', 'voteOut', 'vote_In', 'vote_Out',
              'referendum', 'Referendum', 'REFERENDUM']
```
to make sure that the tweets would definitely be related to the referendum (identification of word participation in the tweet was done using the .find(word) function). However, the corpus I was able to accumulate was no bigger than 19k tweets. This climbed to 56k once I added: ' eu ' and ' EU '. It is true that this resulted in including some tweets about the European Union that were unrelated to the referendum but after a good inspection of the data the proportion of those tweets was insignificant and should not materially affect the accuracy of the model. Finally note that accumulating this many tweets can take as long as a whole day so you will need to plan ahead in case you need to meet some deadline.

## Dividing the Corpus: Training/Cross-Validation/Test sets
The corpus (labelled tweets) was saved in a *.csv file which I loaded using pandas dataframes. So, I used two dataframes, one for the tweets and another one for the corresponding tweet-labels.
```python
# Load Corpus using Pandas
dataFrame = pandas.read_csv('/Users/username/PycharmProjects/twitter_analytics/corpus.csv',header=None, names=['name', 'screen_name', 'id', 'created_at', 'text', 'label'])

# Load Columns as Arrays (Notice: first element = column name)
tweets = dataFrame['text']
del tweets[0]
labels = dataFrame['label']
del labels[0]
```
The del dataframe[0] is simply used to remove the labels at the top of the dataframe.

The model was developed using the Apache Spark MLlib. The first step of the training process was to break the corpus into a training set (60%), cross-validation set (20%) and test set (20%). Unfortunately, an additional restriction required that the original set of 56k tweets had to be cut down to 18202 * 2 = 36404 tweets, where 18202 is the number of the “Leave” labels (the smaller of the two sets).

This had to be done in order to balance the training set between the “Leave” and the “Stay” tweets. You may be wondering why this was necessary, i.e. why sacrifice close to 20k tweets for it?

The answer here is not as simple as one would think so allow me to elaborate a bit on this matter. It is reasonable to expect that words (tokens) with significant semantic interpretation like “voteIn” or “voteLeave” will appear mostly (even exclusively in some cases) in stay-, leave-labelled tweets respectively. However, other, less semantically significant, words like “say” or “support” are very likely to appear in both sets in equal proportions. If one, dismisses this as not important then it is very likely that such words will be quantified in favour of one result over the other when they shouldn’t and this will surely cause a form of bias in the prediction model.

So, first things first, I began the process by creating a Spark context objet:
```python
sc = SparkContext('local[4]', 'EU_Tweet_Sentiment_Analyser')
and started with producing an RDD composed of the tweet labels, on which I applied simple transformations and actions to count respectively the “leave” and “stay” labels.

# Instantiate Tweet RDDS
labels_RDD = sc.parallelize(labels, 4)

total_tweets = labels_RDD.count()
pos_tweets = labels_RDD.filter(lambda x: x == "Stay").count()
neg_tweets = pos_tweets = labels_RDD.filter(lambda x: x == "Leave").count()
This was followed by splitting the tweets into two sets: positives and negatives (leave and stay respectively).

# Break tweets between positive and negative
pos_tweets = []
neg_tweets = []
for (tweet, label) in itertools.izip(tweets, labels):
    if label == "Stay":
        pos_tweets.append(tweet)
    else:
        neg_tweets.append(tweet)
```

This process was followed by calculating the number of tweets to include in every one of the three sets and calling the populate_with() function which chose random tweets from the negatives’ and the positives’ sets to populate each one of them (example below concerned with just the training set).
```python
# Divide respectively to 60%-20%-20%
training_no = int(min(len(pos_tweets), len(neg_tweets)) * 60 / 100)
cross_validation_no = int(min(len(pos_tweets), len(neg_tweets)) * 20 / 100)
test_no = min(len(pos_tweets), len(neg_tweets)) - training_no - cross_validation_no

# Training Set
training_set = []
training_labels = []

(training_set, training_labels) = populate_with(training_no, pos_tweets, "STAY", training_set, training_labels)
(training_set, training_labels) = populate_with(training_no, neg_tweets, "LEAVE", training_set, training_labels)
```
## Tokenize
Next comes tokenizing each tweet in the training set. To do so using Spark transformations, I converted the training list of tweets into an RDD:
```python
training_RDD = sc.parallelize(training_set, 4)
```
For the tokenizing part I used the tokenize method from the nltk library:
```python
from nltk.tokenize import RegexpTokenizer
```
However, just before I engaged in the tokenizing process I had to first worry about a couple of things with the first one being encoding. If you want to get your mind off of worrying about encoding issues the simplest thing you can to is run an map transformation that, essentially, encodes everything into ascii. This can be done with:
```python
training_RDD.map(lambda tweet: tweet.decode("ascii", "ignore").encode("ascii"))
```
In addition to encoding, one has to also get rid of a lot of gibberish in every tweet. Things that are not helpful like leading “RT”s, web-links, numbers, hashtags etc. For this part I used a function I created to reflect the requirements of this project (.map(filter_tweetRDD)), using regular expressions. I won’t provide the function here but I will simply refer you to regex101.com which will be much more than just your right hand in this filtering process.

## Lemmatize
Lemmatization in linguistics is the process of grouping together the different inflected forms of a word so they can be analysed as a single term. For example, dogs to dog, cats to cat, playing to play, played to play and so on. The idea here is that since words with the same root bare the same semantical meaning they should be perceived as the same thing. nltk accounts for this as well. All you need is to import the WordNetLemmatizer.
```python
from nltk.stem import WordNetLemmatizer
```
## Removing stopwords
In the world of Natural Language Processing (NLP) and specifically that of sentiment analysis, stopwords refers to a txt file that contains what are perceived as semantically indifferent. For example, words like: "them, his, very, they, not, during, now" bear no significant semantic value and are therefore disregarded. Again, this can be easily done through the nltk library by importing stopwords:
```python
from nltk.corpus import stopwords
```
## The Negation_tokenizer
Finally, one last step which I found to be quite interesting as well as very helpful in terms of increasing the model’s accuracy, is the negation tokenizer. Intuition behind the employment of this approach lies in the fact that sentiment words behave very differently when under the semantic scope of negation. Take for example “bad” and “not bad” or “support” and “not support”. To capture this effect, one should differentiate between the two words—bad with no negation and bad following after a negation–which can be easily achieved with appending a “_NEG“ suffix in the end of a word if that word is preceded by some form of negation. This can be done using the following function which also accounts for double/triple negation etc.
```python
def negation_tokenizer(tweet_words):

    # regex to match negation tokens
    negation_re = re.compile("""(?x)(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|
            hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|
            doesnt|didnt|isnt|arent|aint)$)|n't""")

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
```
The whole process described up to this point is effectively summarised in the bellow snippet of code:
```python
wordsByTweet = (training_RDD.map(lambda tweet: tweet.decode("ascii", "ignore").encode("ascii"))
                .map(filter_tweet)
                .map(tokenizer.tokenize)
                .map(lemmatize)
                .map(filter_stop_words)
                .map(negation_tokenizer)
                .cache())
```
## Creating a Dictionary (TF-IDF scores)
Once the tokenization process was complete the next step was to create a dictionary of words to use for feature extraction. To do so I had to come up with a way of determining the importance of every unique word in the training corpus and one relatively good choice was TF-IDF scoring. Respectively, the abbreviations refer to Term Frequency and Inverse Document Frequency. The first term simply concerns the number of times, in our case, a token appears within the same tweet while the second one refers to a formula that is primarily based on Document Frequency (DF) and refers to the the number of tweets in which a token appears, i.e.:

![Term Frequency](https://wikimedia.org/api/rest_v1/media/math/render/svg/ac67bc0f76b5b8e31e842d6b7d28f8949dab7937)

where t is the term (token) and D is the corpus of Documents (tweets). In essence, IDF is a measure of the value of information a word (token) provides in terms of whether that word is common or rare across all documents. This is reflected by the logarithmically scaled inverse fraction of the documents (tweets) that contain the word, obtained by dividing the total number of documents (tweets) by the number of tweets that contain the word, and then taking the logarithm of that quotient.

To get the TF-IDF score for every tweet one has to simply multiply the two:
![TF-IDF](https://wikimedia.org/api/rest_v1/media/math/render/svg/10109d0e60cc9d50a1ea2f189bac0ac29a030a00)

where d is a single document (tweet). I won’t go much into the details of these terms but I will, however, comment on two things worth noting. In reality, because of how short a tweet is, the chances are that the TF score that a token receives will be the same for every tweet it appears in since it is very likely that it won’t appear more than once in the same tweet. This means that the same words will very likely receive the same score in the distinct tweets they appear. However, as far as creating a dictionary goes, the IDF score is very helpful for identifying the most important words from the perspective of their frequency amongst different tweets within the corpus. In the case where one is concerned with different modelling objectives, this may not be the optimal approach. In general, however, mapping IDF scores with every word, thus creating an ordered dictionary based on those scores, can help one filter out the less significant words which will assist in further speeding up the prediction time, trading off an insignificant portion of the model’s accuracy.

The process is summarised in the following snippet of code:
```python
def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus
    Returns:
        RDD: a RDD of (token, IDF value)
    """

    N = corpus.count()

    # The result of the next line will be a flatmap with distinct tokens...
    unique_tokens = corpus.flatMap(lambda x: list(set(x)))
    token_count_pair_tuple = unique_tokens.map(lambda x: (x, 1))
    token_sum_pair_tuple = token_count_pair_tuple.reduceByKey(lambda a, b: a + b)
    return token_sum_pair_tuple.map(lambda x: (x[0], float(N) / x[1]))  # compute weight
```
followed by:
```python
dictionary_RDD_IDFs = idfs(wordsByTweet)
dictionary_RDD_IDFs_Weights = dictionary_RDD_IDFs.sortBy(lambda (token, score): score).collectAsMap()
```
Once, the dictionary has been created one should instantiate a broadcast variable in order to send the dictionary to the workers for the calculation of the final TF-IDF score:
```python
import pickle

# Write IDFS_weights_BV as python dictionary to a file
output = open('/Users/username/PycharmProjects/twitter_analytics/dictionary_RDD_IDFs_Weights.pkl', 'wb')
pickle.dump(dictionary_RDD_IDFs_Weights, output)
output.close()
In my case, I was able to produce an ordered dictionary of 27k words, which means that an equivalent number of features need to be created for every tweet should one decide to include the whole dictionary. The final step of the process is to produce pairs of token:TFIDF score which can be done as follows:

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
followed by:

TFsIDFs_Vector_Weights_RDDs = (wordsByTweet.map(lambda tokens: (tfidf(tokens,IDFS_weights_BV.value)))
            .cache())
```
## Feature Extraction
This section is a bit tricky. Generally, the process is simple and intends to convert every tweet which is, at this point a map of words and TF-IDF scores, into a vector that will be used for training the model. Our intention here however is to use sparse vectors rather than actual vectors and this is because, as we have already mentioned, tweets are far shorter than documents and thus if we assume that every feature will represent a unique word in the dictionary (27k words) we will end up with more than 99% of features with null values, thus unnecessarily wasting memory space.  Fortunately, sparse vectors take much fewer space and they are faster to process.

One thing you should have in mind before you start working with sparse vectors is that they are instantiated in the following way:
```python
SparseVector(vector_size, non_zero_indexes, index_tfidf_values)
```
where the non_zero_indexes is a list of integers of the form [0,4,67,1003,25001,27000] which must be ordered (ascending), followed by another list of floating point numbers that represent the values stored in the corresponding index positions of the first list.

The way to extract the feature vectors is fundamentally simple. One needs to iterate through the dictionary of words (each word representing a feature) and mark the index position where a match is found between a word in a tweet and a word in the dictionary, storing the TFIDF score of that word at that position.

A pseudo image of a vector of TFIDF scores for the indexes where a match has been identified between a tweet word and a dictionary word.&nbsp;

A pseudo image of a vector of TFIDF scores for the indexes where a match has been identified between a tweet word and a dictionary word.

Two things need to precede this process. One is ordering of the dictionary and the second is ordering of the words in the tweets prior to running the matching process. This guarantees that the indexes of the matched words will appear in ascending order in the non_zero_indexes list making sure that no exceptions will be thrown. This process appears in the bellow snippet:
```python
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

# Create an ordered dictionary of the N first words
Dictionary = (dictionary_RDD_IDFs
              .sortBy(lambda (token, score): score)
              .map(lambda (token, score): token)
              .collect())       # N = all-->collect(), otherwise use take(N)

# Create a broadcast variable for the Dictionary
# Dictionary MUST be sorted. If not sparse-vectors in the featurize function will throw exception.
Dictionary_Sorted = sorted(Dictionary)
Dictionary_BV = sc.broadcast(Dictionary_Sorted)

# Feature Extraction
Training_Set_Vectors = (TFsIDFs_Vector_Weights_RDDs
                        .map(lambda (tokens): featurize(tokens))
                        .collect())
```
## Training the model
One last step needs to precede the “training the model” part. I must admit at this point that there is a better alternative to this which wouldn’t require that I run a .collect()) action in the end of the last section. That is to keep the association of each tweet with its label throughout the whole process which I haven’t done. I have kept however a list with the respective labels of each tweet and I can create a (sparse-vector, label) tuple at this point anyway, even if this means that I have to collect everything back to the master.

Note that after the creation and processing of an RDD, unless the user forcibly alters the order of the contents in the RDD (e.g. by using sortBy), Spark maintains the original order of the file. Judging by the resulting accuracy of the produced model I think I can corroborate this guarantee.

So in this final step, I re-instantiated a new RDD from the collected Training_Set_Vectors and their corresponding labels as follows:

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
followed by:
```python
labelled_training_set_RDD = sc.parallelize(final_form_4_training(Training_Set_Vectors, training_labels))
```
## Training the model
I guess that there couldn’t be a better proof that 80% of data science is preparing the data than this next line of code:
```python
model = NaiveBayes.train(labelled_training_set_RDD, 1.0)
```
And just like that you have have trained a model.

## Cross-Validation and Testing of the model’s accuracy
The above process was repeated numerous times in an attempt to cross-validate the model’s accuracy and choose the appropriate number of features (the number of words to include from the dictionary). After repeating the cross validation process, I wasn’t able to achieve a better accuracy than 84% and that was when including the whole dictionary. The iterative training trials yielded the accuracy graph below, where on the x-axis one sees the number of most important words kept from the dictionary (i.e. number of features for each tweet) and the model’s respective accuracy for that number.

Given that I was using sparse-vectors, computation time was not significantly affected in the case of reducing the number of features even up to 1/10th of the dictionary’s original size. The accuracy however was significantly diminished, thus I opted to keep the whole dictionary in order to produce the final model.

Both for the cross-validation process as well as for the testing process the routine is the same:
·      tokenize;
·      featurize, and;
·      predict

and it is detailed as follows:
```python
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

# Compute TF-IDF scores
test_TFsIDFs_Vector_Weights_RDDs = (test_wordsByTweet
                                    .map(lambda tokens: (tfidf(tokens, IDFS_weights_BV.value)))
                                    .cache())

# Feature Extraction
raw_input("Extract Features for Cross-Validation Set...")
test_Set_Vectors = (test_TFsIDFs_Vector_Weights_RDDs
                    .map(lambda (tokens): featurize(tokens))
                    .collect())

# Generate labelledppoint parameter...
labelled_test_set_RDD = sc.parallelize(final_form_4_training(test_Set_Vectors, test_labels))

# Compute Accuracy
predictionAndLabel = labelled_test_set_RDD.map(lambda x: (model.predict(x.features), x.label))
accuracy = 100.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / labelled_test_set_RDD.count()

print ("Model Accuracy is: {0:.2f}%".format(accuracy))
The same process is followed in the case of the test set which yielded an accuracy score equal to 84.43%.

Saving the model
Saving the model is as easy as running the following command:

model_path = "/Users/username/PycharmProjects/twitter_analytics/NB_model"
model.save(sc, model_path)
```
Note, however, that if the model is already saved on your machine you will get an exception when the code is run again as you won’t be able to overwrite the existing model. In other words make sure you have saved a previously produced model somewhere else before you try to train a new one.

## Predicting Your Stance on the referendum

Back to the twitter API. The first step of this process is, given a twitter handle, to collect the set of users that one follows. Note that the generalised assumption here is that people tend to share similar opinions with most of the people they follow. Of course, this is not 100% correct but it was good enough to yield reasonable results for my purposes.

So, for accumulating the set of users one follows, you simply need to use the api.friend() method as follows:
```python
def get_friends(user_handle, api):

    # Create empty users array with user's scree_name as first input
    users = [user_handle]

    # Get user's friends
     for user in api.friends(screen_name=user_handle, count=200):
        print user.screen_name
        users.append(user.screen_name)

    return users
friends = get_friends(user_handle, api)
```
Once more, remember from the beginning of the post, that you have to use your application credentials and instantiate a tweepy api object before you can run this code.

After the list of friends has been accumulated all you need to do is to once more start accessing user timelines based on the users found in the friends’ list, filter out those that are not related to the “EU”, tokenize and featurize the rest and use the prediction model to classify them.

To do so you will need three things:

1.     The IDF_Weights_Scores you produced in the modelling process.
2.     An order dictionary (just the words).
3.     The prediction model.

Assuming that your previous jobs are sound, you should be able to retrieve these three things by running the following code:
```python
# Read dictionary_RDD_IDFs_Weights dict back from the file
pkl_file = open('/Users/user/PycharmProjects/twitter_analytics/dictionary_RDD_IDFs_Weights.pkl','rb')
dictionary_RDD_IDFs_Weights = pickle.load(pkl_file)
pkl_file.close()

# Create dictionary_RDD_IDFs_Weights broadcast variable
IDFS_weights_BV = sc.broadcast(dictionary_RDD_IDFs_Weights)

# Read dictionary of the N first words with respect to the most important ones (IDFs sorted)
sorted_dict = sorted(dictionary_RDD_IDFs_Weights.items(), key=operator.itemgetter(1))

# Set to max of N words for corresponding number of features for which the model is trained
Dictionary = []
for key, value in sorted_dict:
    Dictionary.append(key)

print len(Dictionary)

# Create a broadcast variable for the Dictionary
Dictionary_BV = sc.broadcast(sorted(Dictionary))

# Load Naive Bayes Model
model_path = "/Users/christoshadjinikolis/PycharmProjects/twitter_analytics/NB_model"
sameModel = NaiveBayesModel.load(sc, model_path)
The next and final step is to run the get_all_tweets_for_prediction() function below which collects only the first 200 tweets from a person's timeline (simply to reduce the processing time at the event), categorises them and returns their respective counts that are summed up with the overall ones.

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
```
Part of the code is purposely omitted at this point. However, you can find the complete code here. The function is, of course, called within a loop:
```python
timeout = time.time() + 60 * 10  # 2 minutes and a half from now
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
```
The resulting ratio can then be used for categorising the person whose handle you used. A better alternative is to produce a leave-/stay ratio by categorising all the people a person follows based on the ratio of leave/stay tweets they posted, but it was more fun to refer to the categorised tweets in this process, even if, for example, in the case of Nigel Farage who follows Cameron, Cameron’s tweets significantly increased Farage’s relevance with stay tweets (still Farage was labelled “Leave”).

This last part is a bit big to include in this post. So I made sure to upload the whole code in this repo. In the repo you will also find the videos we used on the day that are launched using a simple apple script in the `prediction.py` file.

I must stress at this point that this is by no means a bullet-proof, sophisticated approach to opinion mining and should not be perceived as such. It was simply a fun project intended for internal purposes. In this respect, as a final part of the categorisation process, apart from the tweet categorisation part, I also factored in some demographics, weighing them to affect the overall result by 25%, by asking users to answer a few simple questions.  You can find these here.

The last thing I want to comment about is that the C_KEY parameter you can see in the get_all_tweets_for_prediction function.  On the day I was demoing the project I wanted to avoid hitting a “Rate limit exception“ blocking me from accessing more tweets and so I had to come up with a way to counter this problem. After a little bit of research, I found that the Rate Limit ban is application-directed. Specifically, the ban is not on your IP but on your application. So if you get yourself a set of 4-5 application credentials and cycle through them every time you get a Rate limit exception you will basically be able to collect tweets non-stop! Make sure however to change the api object to a local object rather than a global one for this to work (you will see what I mean when and if you try this). That’s a DoS attack right there Twitter!

## Last words
All in all, this was a fun project which I very much enjoyed. It has a lot of potential and can be easily generalised to satisfy other needs while at the same time I am sure that there is a lot of room for improvement and would be happy to see anyone, who has the time to do so, contributing to this project.

© Copyright 2016 Reply Limited. ALL RIGHTS RESERVED.

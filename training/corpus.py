from datetime import datetime
import itertools
from typing import BinaryIO

import tweepy
import pandas
import time
import csv
import os

# Twitter API credentials
consumer_key = "to be completed"
consumer_secret = "to be completed"
access_key = "to be completed"
access_secret = "to be completed"

# authorize twitter, initialize tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


def get_all_tweets_for_corpus(screen_name, name, label, corpus_directory):
    # Twitter only allows access to a users most recent 3240 tweets with this method
    print("Accumulating tweets for %s:" % screen_name)

    # initialize a list to hold all the tweepy Tweets
    all_tweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    resume = True
    while resume:
        resume = False
        try:
            new_tweets = api.user_timeline(screen_name=screen_name, count=200)
            # If there are tweets to accumulate then:
            if len(new_tweets) > 0:
                # save most recent tweets
                all_tweets.extend(new_tweets)

                # save the id of the oldest tweet less one
                oldest = all_tweets[-1].id - 1

                # stop collecting tweets if stop_year is reached
                stop_year = 2015
                year = int(str(all_tweets[-1].created_at)[0:4])  # for first loop

                # keep grabbing tweets until there are no tweets left to grab
                while (len(new_tweets) > 0) & (year >= stop_year):
                    print(f"getting tweets before {oldest}")

                    # all subsequent requests use the max_id param to prevent duplicates
                    try:
                        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)
                        # save most recent tweets
                        all_tweets.extend(new_tweets)

                        # update the id of the oldest tweet less one
                        oldest = all_tweets[-1].id - 1

                        # get year of last tweet
                        year = int(str(all_tweets[-1].created_at)[0:4])

                        print(all_tweets[-1].created_at)
                        print(f"...{len(all_tweets)} tweets downloaded so far, latest tweet collected is within {year}.")

                    except tweepy.TweepError as e:
                        if e.reason == "Not authorized.":
                            print("Stop exception: %s." % e.reason)
                            print("------------------------")
                            return
                        elif e.reason == "Sorry, that page does not exist.":
                            print("Stop exception: %s." % e.reason)
                            print("------------------------")
                            return
                        else:
                            print("Tweepy Exception: %s" % e.reason)
                            print("Rate limit exceeded--> sleeping for 15 minutes from: %s." % str(datetime.now()))
                            time.sleep(60 * 15)
                            continue
                    except StopIteration as e:
                        print('ERROR: Failed because of %s' % e.value)
                        return  # exit if something goes wrong

                print("Finished procesing tweets for %s" % (screen_name))

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

                for tweet in all_tweets:
                    for word in list_words:
                        if tweet.text.encode("utf-8").find(word) != -1:
                            print(tweet.text.encode("utf-8"))
                            eu_tweets.append(tweet)
                            break

                # transform the tweepy tweets into a 2D array that will populate the csv
                out_tweets = [[name, screen_name, tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"), label]
                              for tweet in eu_tweets]

                # write the csv
                with open(os.path.join(corpus_directory, 'labelled__tweets.csv'), 'a') as f:
                    writer = csv.writer(f)
                    for row in out_tweets:
                        writer.writerow(row)
            else:
                print("No tweets to collect...")

            print("------------------------")
        except tweepy.TweepError as e:
            if e.reason == "Not authorized.":
                print("Stop exception: %s." % e.reason)
                print("------------------------")
                return
            else:
                print("Tweepy Exception: %s" % e.reason)
                print("Rate limit exceeded--> sleeping for 15 minutes from: %s." % str(datetime.now()))
                time.sleep(60 * 15)
                resume = True
        except StopIteration as e:
            print('ERROR: Failed because of %s' % e.value)
            return  # exit if something goes wrong


if __name__ == '__main__':

    # Load twitter handles
    print("Load twitter MPs")
    dataFrame = pandas.read_csv('../data/raw/labelled_MPs.csv', header=None, names=['name', 'screen_name', 'party', 'label'])

    screenNames = dataFrame['screen_name']
    names = dataFrame['name']
    labels = dataFrame['label']

    # Skip First item ('screen_name')
    iterScreenName = iter(screenNames)
    next(iterScreenName)

    # Skip First item ('name')
    iterName = iter(names)
    next(iterName)

    # Skip First item ('label')
    iterLabel = iter(labels)
    next(iterLabel)

    # Create Corpus CSV
    corpus_directory = "/path/to/corpus"
    f: BinaryIO
    with open(os.path.join(corpus_directory, 'labelled__tweets.csv'), 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["name", "screen_name", "id", "created_at", "text", "label"])

    # Iterate through screen_names and collect tweets
    for (screen_name, name, label) in itertools.iizip(iterScreenName, iterName, iterLabel):
        get_all_tweets_for_corpus(screen_name, name, label, corpus_directory)

    print("Finished collecting tweets for the corpus!")

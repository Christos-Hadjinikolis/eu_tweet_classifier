import tweepy
import csv

# Twitter API credentials
consumer_key = "to be completed"
consumer_secret = "to be completed"
access_key = "to be completed"
access_secret = "to be completed"

# authorize twitter, initialize tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


# -- SUB-FUNCTIONS ----------------------------------------------------------------------------------------------------
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

# -- MAIN CODE --------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # UK MPs owner_screen_name:"tweetminster" slug:"ukmps"
    print("Instantiating UK-MPs's List...")
    ukMPs = get_list_members("tweetminster", "ukmps")
    print("List created successfully... ")

    # EU MPs owner_screen_name:"Europarl_EN" slug:"all-meps-on-twitter"
    print("Instantiating EU-MPs's List...")
    euMPs = get_list_members("Europarl_EN", "all-meps-on-twitter")
    print("List created successfully... ")

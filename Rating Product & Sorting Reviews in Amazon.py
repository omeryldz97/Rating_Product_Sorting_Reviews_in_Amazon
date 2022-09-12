
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df=pd.read_csv("....")#The data set is not shared because it is private.
df.head()
df.shape
df.isnull().sum()
#Task 1: Calculate the Average Rating according to the current comments and compare it with the existing average rating.
#Step1: Calculate the average score of the product.
df["overall"].mean()
#4.58

#Step 2: Calculate the weighted average score by date.
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = pd.to_datetime(str(df['reviewTime'].max()))
df["day_diff"] = (current_date - df['reviewTime']).dt.days

# determination of time-based average weights
def time_based_weighted_average(dataframe, w1=24, w2=22, w3=20, w4=18, w5=16):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.2), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.2)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.4)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.4)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.6)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.6)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w4 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w5 / 100

time_based_weighted_average(df)


#Task 2: Determine 20 reviews for the product to be displayed on the product detail page.
#Step 1: Generate the variable helpful no.
df["helpful_no"]=df["total_vote"] - df["helpful_yes"]
df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

#Step 2: Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores and add them to the data.
def score_up_down(up,down):
    return up - down

def score_average_rating(up,down):
    if up + down == 0:
        return 0
    return up/(up+down)

# Wilson Lower Bound Score
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


#score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down(x["helpful_yes"], x["helpful_no"]), axis=1)

#score_average_rating
df["score_average_rating"]=df.apply(lambda x:score_average_rating(x["helpful_yes"],x["helpful_no"]),axis=1)

#wilson_lower_bound
df["wilson_lower_bound"]=df.apply(lambda x:wilson_lower_bound(x["helpful_yes"],x["helpful_no"]),axis=1)

#Identify and rank the top 20 comments to wilson_lower_bound.
df.sort_values("wilson_lower_bound",ascending=False).head(20)

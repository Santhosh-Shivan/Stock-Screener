import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import math
import numpy as np
import matplotlib.pyplot as plt


# Average dollar Volume over a period
def avg_dollar_vol(df, len):
    df_temp = df.iloc[-1 * len :]
    avg_dollar_vol = 0
    for index, _ in df_temp.iterrows():
        avg_dollar_vol += (
            df_temp.loc[index, "Volume"] * df_temp.loc[index, "Close"]
        ) / len
    return avg_dollar_vol


# Average ADR% over a period
def ADR(df, len):
    df_temp = df.iloc[-1 * len :]
    sum = 0
    for index, _ in df_temp.iterrows():
        sum += (df_temp.loc[index, "High"] / df_temp.loc[index, "Low"]) / len

    ADR = 100 * (sum - 1)
    return ADR


# Relative strength over a period
def RS_line(df, df_index, len):

    df_index_temp = df_index.iloc[-1 * len :]
    df_temp = df.iloc[-1 * len :]
    # Method 1
    """RS = df_temp["Close"].values / index_df_temp["Close"].values"""

    # Method 2
    s3 = (
        df_temp["Close"].values / df_index_temp["Close"].values * 100
    )  # https://www.tradingview.com/script/nFTOmmXU-IBD-Relative-strengtH/

    mult = df["Close"].values[-60] / s3[-60]
    s4 = s3 * mult * 0.85

    # Min-Max Normalization
    """min_val = np.min(s4)
    max_val = np.max(s4)
    normalized_s4 = (s4 - min_val) / (max_val - min_val) * 100"""

    return s4


# RS_score
"""Yearly performance of stock divided by INDEX performance during the same period(SPY/QQQ).

RS Score for Stocks = 40% * P3 + 20% * P6 + 20% * P9 + 20% * P12

RS Score for SPY/QQQ = 40% * P3 + 20% * P6 + 20% * P9 + 20% * P12

With P3 the performance of the 3 last month. (P3 = Close/Close[63], for 63 days back)

Formula: RS Score = (1 + RS Score for Stocks) / (1 + RS Score for SPY/QQQ)

Then all stocks are ranked from largest to smallest and a percentile is assigned from 99 to 0."""


def RS_score(closes: pd.Series, closes_ref: pd.Series):
    rs_stock = Absolute_strength(closes)
    rs_ref = Absolute_strength(closes_ref)
    rs_score = (1 + rs_stock) / (1 + rs_ref) * 100
    rs_score = int(rs_score * 100) / 100  # round to 2 decimals
    return rs_score


def Absolute_strength(closes: pd.Series):
    """Calculates the Absolute_strength of the stock last year (most recent quarter is weighted double)"""
    # Quarters method
    quarters1 = perf(closes, 1 * int(252 / 4) + 1)
    quarters2 = perf(closes, 2 * int(252 / 4) + 1)
    quarters3 = perf(closes, 3 * int(252 / 4) + 1)
    quarters4 = perf(closes, 4 * int(252 / 4) + 1)
    return 0.4 * quarters1 + 0.2 * quarters2 + 0.2 * quarters3 + 0.2 * quarters4


def perf(closes: pd.Series, n):
    length = min(len(closes), n)
    prices = closes.tail(length)
    pct_chg = prices.pct_change().dropna()
    perf_cum = (pct_chg + 1).cumprod() - 1
    return perf_cum.tail(1).item()


def RS_rating(RS_scores: dict):
    percentile = {}
    for idx, ticker in enumerate(RS_scores):
        percentile[ticker] = int((idx + 1) / len(RS_scores) * 100)

    return percentile


def linear_regression(y):

    x = []
    for i in range(len(y)):
        x.append(i)

    slope, intercept, r, p, std_err = stats.linregress(x, y)
    # print(slope, intercept)

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, x))
    # print(mymodel)

    # plt.scatter(x, y)
    # plt.plot(x, mymodel)
    # plt.show()
    angle = math.degrees(math.atan(slope))
    return slope


# Both this anf above func does the same
def calc_slope(
    src, length
):  # https://www.tradingview.com/script/MMcT4hbU-IBD-Relative-Strength-Linear-Regression/
    # Check if there are enough data points

    sum_x = 0.0
    sum_y = 0.0
    sum_x_sqr = 0.0
    sum_xy = 0.0

    for i in range(length):
        val = src[i]
        per = i + 1.0
        sum_x += per
        sum_y += val
        sum_x_sqr += per * per
        sum_xy += val * per

    slope = (length * sum_xy - sum_x * sum_y) / (length * sum_x_sqr - sum_x * sum_x)
    average = sum_y / length
    intercept = average - slope * sum_x / length + slope
    # print(slope, intercept)
    return slope, average, intercept

import yfinance as yf
import pandas as pd
import os
import datetime


# get data from yahoo finance
# search symbols here: https://finance.yahoo.com/lookup
def __download_data(ticker, start, end, xchange, write_to_file=True):
    directory = "yahoo_data/" + xchange
    if not os.path.exists(directory):
        os.makedirs(directory)

    fileName = directory + "/" + ticker[0] + str(start) + str(end) + ".csv"

    allow_reading_file = True
    if os.path.isfile(fileName) and allow_reading_file:
        # print("reading from file:  " + fileName)
        df = pd.read_csv(
            fileName, index_col="Date", parse_dates=True, na_values=["nan"]
        )

        return df
    else:
        print("fetching data from API for " + ticker[0])
        df = yf.download(ticker[0], start=start, end=end)
        # df = yf.download(ticker[0], period = "1")

        if write_to_file:
            df.to_csv(fileName)

    return df


if __name__ == "__main__":
    startDate = datetime.date(2024, 1, 1)
    endDate = datetime.date(2024, 4, 30)
    print(__download_data(["AAPL"], startDate, endDate, "US").head())
    # get_data_for_symbol(RELIANCE.NS', startDate, endDate)

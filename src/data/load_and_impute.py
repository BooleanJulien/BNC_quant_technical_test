import pandas as pd
import numpy as np
import openpyxl

# funds and their corresponding underlying asset types
FUND_INDEX_LOOKUP = {
        "Fund A": "Canada equities",
        "Fund B": "Canada equities",
        "Fund C": "US equities",
        "Fund D": "US equities",
        "Fund E": "EAFE equities",
        "Fund F": "EAFE equities",
        "Fund G": "EM equities",
        "Fund H": "EM equities",
        "Fund I": "Canada Broad FI",
        "Fund J": "Canada Broad FI",
        "Fund K": "Canada Corps FI",
        "Fund L": "Canada Corps FI"
    }

def read_sheets():
    """
    Loads data from excel file in data folder

    Returns
    -------
    pd.DataFrame
        4 dataframes corresponding to each sheet in
        raw data
    """
    benchmark_weights = pd.read_excel(
        io="../../data/raw/Technical Test - Portfolio Attribution.xlsm",
        sheet_name="Benchmark Weights",
        header=[0, 1],
        index_col=0)

    saa_weights = pd.read_excel(
        io="../../data/raw/Technical Test - Portfolio Attribution.xlsm",
        sheet_name="SAA Weights",
        header=[0, 1],
        index_col=0)

    manager_weights = pd.read_excel(
        io="../../data/raw/Technical Test - Portfolio Attribution.xlsm",
        sheet_name="Manager Weights",
        header=[0, 1, 2],
        index_col=0)

    returns = pd.read_excel(
        io="../../data/raw/Technical Test - Portfolio Attribution.xlsm",
        sheet_name="Returns",
        header=[0, 1],
        index_col=0)

    return benchmark_weights, saa_weights, manager_weights, returns


def single_fill_conditionally_with_geometric_average(t_0,
                                                     t_plus_1,
                                                     fund,
                                                     fund_returns,
                                                     index_returns,
                                                     fund_index_lookup):
    """imputes one Fund and Date combination
    of a returns df assuming it is not part of
    a consecutive set of missing values

    Parameters
    ----------
    t_0 : Timestamp
        current date
    t_plus_1 : Timestamp
        date 1 timestep ahead
    fund : str
        name of fund
    fund_returns : pd.DataFrame
        returns df to be modified
    index_returns : pd.DataFrame
        index returns for reference
    fund_index_lookup : dict
        funds and their corresponding
        underlying asset types

    Returns
    -------
    pd.DataFrame
        returns df with the fund/date combo modified
    """

    # if underlying is not traded, assume fund return is 0
    if index_returns.loc[t_0, fund_index_lookup[fund]] == 0:
        fund_returns.at[t_0, fund] = 0
    else:
        geometric_average = ((1 + fund_returns.loc[t_plus_1, fund]) ** (1/2)) - 1
        fund_returns.at[t_0, fund] = geometric_average
        fund_returns.at[t_plus_1, fund] = geometric_average

    return fund_returns


def double_fill_conditionally_with_geometric_average(t_0,
                                                     t_plus_1,
                                                     t_minus_1,
                                                     fund,
                                                     fund_returns,
                                                     index_returns,
                                                     fund_index_lookup):
    """imputes two Fund and Date combinations
    of a returns df assuming t_0 is the latter part
    of a consecutive set of two missing values

    Parameters
    ----------
    t_0 : Timestamp
        current date
    t_plus_1 : Timestamp
        date 1 timestep ahead
    t_minus_1 : Timestamp
        date 1 timestep back
    fund : str
        name of fund
    fund_returns : pd.DataFrame
        returns df to be modified
    index_returns : pd.DataFrame
        index returns for reference
    fund_index_lookup : dict
        funds and their corresponding
        underlying asset types

    Returns
    -------
    pd.DataFrame
        returns df with the fund/date combos modified
    """
    # if underlying is not traded, assume fund return is 0
    # stretch goal - TODO niche case handle where previous date
    # is traded but current date is not
    if index_returns.loc[t_0, fund_index_lookup[fund]] == 0:
        fund_returns.at[t_0, fund] = 0
        fund_returns.at[t_minus_1, fund] = 0
    else:
        geometric_average = ((1 + fund_returns.loc[t_plus_1, fund]) ** (1/3)) - 1
        fund_returns.at[t_minus_1, fund] = geometric_average
        fund_returns.at[t_0, fund] = geometric_average
        fund_returns.at[t_plus_1, fund] = geometric_average

    return fund_returns


def main():
    """Writes a csv of cleaned returns data to the
    interim data folder.
    """

    benchmark_weights, saa_weights, manager_weights, returns = read_sheets()

    fund_returns = returns['Fund Total Returns (CAD)']
    index_returns = returns['Index Total Returns (CAD)']

    # todo find better way to suppress copy warning
    pd.set_option('mode.chained_assignment', None)
    fund_returns['Fund F'].fillna(0, inplace=True)

    # save dates that we need to modify
    dates_to_modify = list(fund_returns[fund_returns.isna().any(axis=1)].index)

    fund_returns_copy = fund_returns.copy()

    # get the necessary time stamps for each row
    fund_returns_copy["t_0"] = list(fund_returns_copy.reset_index()["index"])
    fund_returns_copy["t_minus_1"] = fund_returns_copy["t_0"].shift(-1)
    fund_returns_copy["t_plus_1"] = fund_returns_copy["t_0"].shift(1)

    # add flags for type of missing value
    fund_returns_copy["one_value_missing_flag"] = fund_returns_copy.apply(
        lambda row: 1 if (row["t_0"] in dates_to_modify) & (
                        row["t_minus_1"] not in dates_to_modify) &(
                        row["t_plus_1"] not in dates_to_modify)
        else 0, axis=1)

    fund_returns_copy["two_values_missing_flag"] = fund_returns_copy.apply(
        lambda row: 1 if (row["t_0"] in dates_to_modify) & (
                        row["t_minus_1"] in dates_to_modify) & (
                        row["t_plus_1"] not in dates_to_modify) 
        else 0, axis=1)

    assert (fund_returns_copy["one_value_missing_flag"].sum() +
            fund_returns_copy["two_values_missing_flag"].sum() * 2) == len(
        dates_to_modify)

    # seems ok not to vectorize this as holidays make up small proportion of data
    for date in dates_to_modify:
        for fund in fund_returns.columns:

            if fund_returns_copy.loc[date, "one_value_missing_flag"] == 1:
                fund_returns = single_fill_conditionally_with_geometric_average(
                    t_0=date,
                    t_plus_1=fund_returns_copy.loc[date, "t_plus_1"],
                    fund=fund,
                    fund_returns=fund_returns,
                    index_returns=index_returns,
                    fund_index_lookup=FUND_INDEX_LOOKUP)

            elif fund_returns_copy.loc[date, "two_values_missing_flag"] == 1:
                fund_returns = double_fill_conditionally_with_geometric_average(
                    t_0=date,
                    t_plus_1=fund_returns_copy.loc[date, "t_plus_1"],
                    t_minus_1=fund_returns_copy.loc[date, "t_minus_1"],
                    fund=fund,
                    fund_returns=fund_returns,
                    index_returns=index_returns,
                    fund_index_lookup=FUND_INDEX_LOOKUP)

    assert fund_returns.isna().any().sum() == 0

    # set the fund returns to our cleaned data
    returns['Fund Total Returns (CAD)'] = fund_returns

    assert returns.isna().any().sum() == 0

    returns.to_csv("../../data/interim/returns.csv")

if __name__ == '__main__':

    main()
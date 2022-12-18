import pandas as pd
import numpy as np

class ReturnSeriesCalculator:
    """
    ReturnSeriesCalculator is a class for calculating fund and benchmark returns based on given weightings and return data.

    Attributes:
    benchmark_weights (pd.DataFrame): Dataframe of benchmark weights with datetime index.
    saa_weights (pd.DataFrame): Dataframe of strategic asset allocation weights with datetime index.
    manager_weights (dict): Dictionary with keys of return categories and values of dataframes of fund allocations for that asset type.
    index_returns (pd.DataFrame): Dataframe of index returns with datetime index.
    fund_returns (pd.DataFrame): Dataframe of fund returns with datetime index.
    benchmark_returns (pd.DataFrame): Dataframe of calculated benchmark returns with datetime index.
    saa_returns (pd.DataFrame): Dataframe of calculated SAA returns with datetime index.
    manager_returns (dict): Dictionary with keys of return categories and values of dataframes of calculated fund returns for that asset type.

    Methods:
    get_benchmark_returns(): Calculates and sets benchmark_returns attribute.
    get_saa_returns(): Calculates and sets saa_returns attribute.
    get_manager_returns(): Calculates and sets manager_returns attribute.
    get_manager_returns_with_drift(): Calculates and returns manager returns as if only rebalanced on target dates.
    get_all_returns(): Calls all return calculation methods to populate all return attributes.
    create_return_comparison(): Creates a comparison df of all calculated returns.
    write_csv(file_name): Writes all calculated returns to a CSV file with the given file name.
    return_gen(return_df, allocation_df): Generator function for calculating returns based on given return and allocation data.
    """

    def __init__(self, benchmark_weights, saa_weights, manager_weights, returns):
        returns.index = pd.to_datetime(returns.index)

        # in general here dropping multi-indexes and renaming columns
        # for consistency and ease of use/readablity
        self.index_returns = returns["Index Total Returns (CAD)"]
        self.fund_returns = returns["Fund Total Returns (CAD)"]

        self.benchmark_weights = benchmark_weights.droplevel(0, axis=1)
        self.benchmark_weights = self.benchmark_weights.rename(
            columns=dict(zip(self.benchmark_weights.columns,
                             self.index_returns.columns[0:5])))

        self.saa_weights = saa_weights.droplevel(0, axis=1)
        self.saa_weights = self.saa_weights.rename(
            columns=dict(zip(self.saa_weights.columns,
                             self.index_returns.columns)))

        manager_weights = manager_weights.droplevel(0, axis=1)
        man_sub_cats = list(
            manager_weights.columns.get_level_values(0).unique())

        # switching order to make sure US and Canadian equity funds
        # get assigned correctly
        man_sub_cats[0], man_sub_cats[1] = man_sub_cats[1], man_sub_cats[0]

        # slightly heavy looking code here but we are creating a dictionary
        # in which the keys are the return categories of the index and the
        # values are dataframes of the fund allocations for that particular
        # asset type. Each dataframe's rows in this dict add up to 1. It is
        # convenient to have these dfs in this format for fund_allocation_gen()
        self.manager_weights = {
            return_category: fund_df for (return_category, fund_df) in zip(
                self.index_returns.columns,
                [manager_weights[man_sub_cats[i]] for i in
                 range(0, len(man_sub_cats))])
        }

    def return_gen(self, return_df, allocation_df):
        """
        return_gen is a generator function for calculating returns based on given return and allocation data for a given date.

        Attributes:
        return_df (pd.DataFrame): Dataframe of returns with datetime index.
        allocation_df (pd.DataFrame): Dataframe of allocation weights with datetime index.

        Yields:
        pd.DataFrame: Dataframe of calculated returns for a given date and allocation.

        Raises:
        AssertionError: If the columns of return_df and allocation_df are not equal.
        AssertionError: If the sum of allocation values is not equal to 1.
        AssertionError: If the date of the allocation decision is not before the date it is used.
        """
        np.testing.assert_array_equal(return_df.columns, allocation_df.columns)

        for date, returns in return_df.iterrows():

            # the searchsorted part gets the index location of the most recent
            # allocation decision given our date being passed in the for loop
            # we use side=left since allocations are end of day
            # we have to use the len() - [::-1].search trick because the
            # dateindex of the data is all in descending order
            allocation = allocation_df.iloc[
                len(allocation_df.index) -
                allocation_df.index[::-1].searchsorted(
                    date, side='left')
            ]

            assert np.isclose(allocation.sum(), 1), f"{allocation} sum is not 1"
            # the allocation decision date should
            # always be behind the date it is used given EOD rebalance
            assert date > allocation.name

            yield pd.DataFrame(returns).T * allocation

    def get_benchmark_returns(self):
        """
        get_benchmark_returns is a method for calculating and setting the benchmark_returns attribute based on the benchmark_weights attribute and the index_returns attribute.

        Attributes:
        self.index_returns (pd.DataFrame): Dataframe of index returns with datetime index.
        self.benchmark_weights (pd.DataFrame): Dataframe of benchmark weights with datetime index.

        Sets:
        self.benchmark_returns (pd.DataFrame): Dataframe of calculated benchmark returns with datetime index.

        Raises:
        AssertionError: If the sum of the benchmark weights * their returns at the first date does not match the benchmark_returns value at the same date.
        """
        self.benchmark_returns = pd.concat(

            [df for df in self.return_gen(
                self.index_returns.loc[:, self.benchmark_weights.columns],
                self.benchmark_weights)]

        ).sum(axis=1)

        assert np.isclose(
            (self.index_returns.iloc[0, 0:5] * self.benchmark_weights.iloc[0]).sum(),
            self.benchmark_returns[0])

    def get_saa_returns(self):
        """
        get_saa_returns is a method for calculating and setting the saa_returns attribute based on the saa_weights attribute and the index_returns attribute.

        Attributes:
        self.index_returns (pd.DataFrame): Dataframe of index returns with datetime index.
        self.saa_weights (pd.DataFrame): Dataframe of strategic asset allocation weights with datetime index.

        Sets:
        self.saa_returns (pd.DataFrame): Dataframe of calculated SAA returns with datetime index.

        Raises:
        AssertionError: If the sum of the SAA weights * their returns at the first date does not match the saa_returns value at the same date.
        AssertionError: If the sum of the SAA weights * their returns at the last date does not match the saa_returns value at the same date.
        """

        self.saa_returns = pd.concat(

            [df for df in self.return_gen(
                self.index_returns,
                self.saa_weights)]

        ).sum(axis=1)

        # make sure manual calculations checks out for first and last row
        assert np.isclose(
            (self.index_returns.iloc[-1] * self.saa_weights.iloc[-1]).sum(),
            self.saa_returns[-1])

        assert np.isclose(
            (self.index_returns.iloc[0] * self.saa_weights.iloc[0]).sum(),
            self.saa_returns[0])

    def fund_allocation_gen(self, asset_fund_df, saa_weights, asset_name):
        """
        fund_allocation_gen is a generator function for calculating fund allocations taking
        into account asset class and SAA weights.

        Attributes:
        asset_fund_df (pd.DataFrame): Dataframe of fund allocations within a given asset class with datetime index.
        saa_weights (pd.DataFrame): Dataframe of strategic asset allocation weights with datetime index.
        asset_name (str): Name of the asset class.

        Yields:
        pd.DataFrame: Dataframe of calculated fund allocations for a given asset class for a given date.
        """
        # generates allocations per fund taking into account the SAA
        saa_asset_index = list(saa_weights.columns).index(asset_name)

        for date, within_asset_fund_allocation in asset_fund_df.iterrows():

            asset_allocation = saa_weights.iloc[

                # in this case we use right side because both
                # of these weights are being used at EOD
                len(saa_weights.index) -
                saa_weights.index[::-1].searchsorted(
                    date, side='right'),

                saa_asset_index]

            yield pd.DataFrame(within_asset_fund_allocation).T * asset_allocation

    def fund_return_gen(self, fund_returns_df, fund_weights):
        """
        fund_return_gen is a generator function for calculating returns for a given set of funds based on fund weights.

        Attributes:
        fund_returns_df (pd.DataFrame): Dataframe of fund returns with datetime index.
        fund_weights (pd.DataFrame): Dataframe of fund weights with datetime index.

        Yields:
        pd.DataFrame: Dataframe of calculated returns for a given set of funds for a given date.

        Raises:
        AssertionError: If the columns of fund_returns_df and fund_weights are not equal.
        AssertionError: If the date of the fund weight decision is not before the date it is used.
        """
        np.testing.assert_array_equal(fund_returns_df.columns, fund_weights.columns)

        for date, fund_returns in fund_returns_df.iterrows():

            fund_allocation = fund_weights.iloc[

                len(fund_weights.index) -
                fund_weights.index[::-1].searchsorted(
                    date, side='left')]

            assert date > fund_allocation.name

            yield pd.DataFrame(fund_returns).T * fund_allocation

    def get_manager_returns(self):
        """
        get_manager_returns is a method for calculating and setting the manager_returns attribute based on the manager_weights attribute, the saa_weights attribute, and the fund_returns attribute.

        Attributes:
        self.manager_weights (dict): Dictionary of dataframes of fund allocations for each asset class.
        self.saa_weights (pd.DataFrame): Dataframe of strategic asset allocation weights with datetime index.
        self.fund_returns (pd.DataFrame): Dataframe of fund returns with datetime index.

        Sets:
        self.fund_saa_allocation (pd.DataFrame): Dataframe of fund allocations with datetime index, taking into account SAA weights.
        self.manager_returns (pd.DataFrame): Dataframe of calculated manager returns with datetime index.

        Raises:
        AssertionError: If the sum of the fund_saa_allocation dataframe is not equal to 1 for all dates.
        AssertionError: If the sum of the fund weights * their returns at the first date does not match the corresponding fund_saa_allocation value at the same date.
        AssertionError: If the sum of the fund weights * their returns at the last date does not match the corresponding fund_saa_allocation value at the same date.
        """

        temp_fund_asset_dict = {}

        for asset in self.manager_weights.keys():

            temp_fund_asset_dict[asset] = pd.concat(

                [df for df in self.fund_allocation_gen(
                    asset_fund_df=self.manager_weights[asset],
                    saa_weights=self.saa_weights,
                    asset_name=asset
                )]
            )

            # make sure manual calc checks out for first and last rows
            np.testing.assert_array_almost_equal(
                self.manager_weights[asset].iloc[0] * self.saa_weights.iloc[0][asset],
                temp_fund_asset_dict[asset].iloc[0])

            np.testing.assert_array_almost_equal(
                self.manager_weights[asset].iloc[-1] * self.saa_weights.iloc[-1][asset],
                temp_fund_asset_dict[asset].iloc[-1])

        temp_fund_allocation_list = [
            temp_fund_asset_dict[asset] for asset in temp_fund_asset_dict.keys()]

        # switch back order of colnames to match fund returns
        temp_fund_allocation_list[0], temp_fund_allocation_list[1] = temp_fund_allocation_list[1], temp_fund_allocation_list[0]

        self.fund_saa_allocation = pd.concat(temp_fund_allocation_list, axis=1)

        # weights should sum to 1 across all assets now instead of within asset
        np.testing.assert_array_almost_equal(
            self.fund_saa_allocation.sum(axis=1),
            np.ones(len(self.fund_saa_allocation)))

        self.manager_returns = pd.concat(

            [df for df in
             self.fund_return_gen(fund_returns_df=self.fund_returns,
                                  fund_weights=self.fund_saa_allocation)],
            axis=0

        ).sum(axis=1)

    def get_manager_returns_with_drift(self):
        """Simulate the manager's returns by applying daily returns and rebalancing at target dates.

        This method simulates the returns of the manager's allocation decisions
        if they were only rebalanced on target dates.
        It does this by applying each day's return to the previous day's portfolio value
        and rebalancing the portfolio when the date matches a target date in the `fund_saa_allocation` dataframe.
        
        Parameters:
            None
            
        Returns:
            self.manager_returns_with_drift (pandas.Series): The simulated returns of the manager's allocation decisions, with dates as the index.
        """

        # utility functions for this process
        def apply_one_day_return(start_value, daily_return):
            return start_value.set_index(daily_return.index) * daily_return

        def daily_fund_return_gen(fund_return_df, fund_saa_allocation):
            # yields daily returns in ascending date order
            # adding 1 makes returns ready for multiplication
            for date, fund_return in (fund_return_df[::-1] + 1).iterrows():
                yield(pd.DataFrame(fund_return).T)

        def rebalance(current_value, fund_weights):
            total_value = current_value.sum(axis=1)[0]
            return fund_weights * total_value

        fund_value = pd.DataFrame(self.fund_saa_allocation.iloc[-1]).T
        start_value = fund_value

        for daily_return in daily_fund_return_gen(self.fund_returns, self.fund_saa_allocation):

            fund_value = pd.concat(
                [apply_one_day_return(start_value, daily_return),
                    fund_value])

            start_value = pd.DataFrame(fund_value.iloc[0]).T

            if start_value.index[0] in self.fund_saa_allocation.index:

                start_value = rebalance(
                    start_value,
                    pd.DataFrame(self.fund_saa_allocation.loc[start_value.index[0]]).T)

        portfolio_value = fund_value.sum(axis=1)

        self.manager_returns_with_drift = portfolio_value[::-1].pct_change()[::-1].dropna()

    def get_all_returns(self):
        self.get_benchmark_returns()
        self.get_saa_returns()
        self.get_manager_returns()
        self.get_manager_returns_with_drift()
        self.create_return_comparison_df()

    def create_return_comparison_df(self):

        self.return_comparison_df = pd.DataFrame(
            {"Benchmark Returns": self.benchmark_returns,
             "SAA Returns": self.saa_returns,
             "Manager Returns": self.manager_returns,
             "Manager Returns With Drift": self.manager_returns_with_drift})

    def write_csv(self, path):
        self.return_comparison_df.to_csv(path)

class IndexCalculator(ReturnSeriesCalculator):
    """
    IndexCalculator is a class for calculating and writing cumulative and indexed returns data based on the parent ReturnSeriesCalculator class.

    Attributes:
    benchmark_weights (pd.DataFrame): Dataframe of benchmark weights with datetime index.
    saa_weights (pd.DataFrame): Dataframe of strategic asset allocation weights with datetime index.
    manager_weights (dict): Dictionary with keys of return categories and values of dataframes of fund allocations for that asset type.
    index_returns (pd.DataFrame): Dataframe of index returns with datetime index.
    fund_returns (pd.DataFrame): Dataframe of fund returns with datetime index.
    benchmark_returns (pd.DataFrame): Dataframe of calculated benchmark returns with datetime index.
    saa_returns (pd.DataFrame): Dataframe of calculated SAA returns with datetime index.
    manager_returns (dict): Dictionary with keys of return categories and values of dataframes of calculated fund returns for that asset type.
    cumulative_return_df (pd.DataFrame): Dataframe of cumulative returns with datetime index.
    indexed_df (pd.DataFrame): Dataframe of indexed returns with datetime index.

    Methods:
    get_benchmark_returns(): Calculates and sets benchmark_returns attribute.
    get_saa_returns(): Calculates and sets saa_returns attribute.
    get_manager_returns(): Calculates and sets manager_returns attribute.
    get_manager_returns_with_drift(): Calculates and returns manager returns as if only rebalanced on target dates.
    get_all_returns(): Calls all return calculation methods to populate all return attributes.
    create_return_comparison(): Creates a comparison df of all calculated returns.
    get_cumulative_return(): Calculates and sets cumulative_return_df attribute.
    get_indexed_df(date): Calculates and sets indexed_df attribute based on the given date.
    write_csv(path): Writes indexed_df to a CSV file at the given file path.
    return_gen(return_df, allocation_df): Generator function for calculating returns based on given return and allocation data.
    """

    def get_cumulative_return(self):
        self.cumulative_return_df = (
            self.return_comparison_df + 1)[::-1].cumprod(axis=0)[::-1] - 1

    def get_indexed_df(self, date="2020-03-06"):
        """This method, get_indexed_df, creates an indexed version of the
        cumulative_return_df attribute.
        The indexed version is created by dividing each row of the
        cumulative_return_df by the value at a specified date and multiplying
        the result by 100. The specified date should be a string in the format 
        "YYYY-MM-DD" and should be in the index of cumulative_return_df.
        The resulting indexed dataframe is stored in the indexed_df attribute.

        Parameters
        ----------
        date : str, optional
            date on which to index, by default "2020-03-06"

        """
        assert (pd.to_datetime(date) in
                self.cumulative_return_df.index), "date should be str in format YYYY-MM-DD and be in index"
        date = pd.to_datetime(date)

        one_plus_cumulative_return = self.cumulative_return_df + 1
        index_row = one_plus_cumulative_return.loc[date]

        self.indexed_df = one_plus_cumulative_return.div(index_row, axis=1) * 100

    def write_csv(self, path):
        self.indexed_df.to_csv(path)


class CumulativeOutperformanceCalculator(ReturnSeriesCalculator):
    """
    CumulativeOutperformanceCalculator is a class for calculating and writing cumulative outperformance data based on the parent ReturnSeriesCalculator class.

    Attributes:
    benchmark_weights (pd.DataFrame): Dataframe of benchmark weights with datetime index.
    saa_weights (pd.DataFrame): Dataframe of strategic asset allocation weights with datetime index.
    manager_weights (dict): Dictionary with keys of return categories and values of dataframes of fund allocations for that asset type.
    index_returns (pd.DataFrame): Dataframe of index returns with datetime index.
    fund_returns (pd.DataFrame): Dataframe of fund returns with datetime index.
    benchmark_returns (pd.DataFrame): Dataframe of calculated benchmark returns with datetime index.
    saa_returns (pd.DataFrame): Dataframe of calculated SAA returns with datetime index.
    manager_returns (dict): Dictionary with keys of return categories and values of dataframes of calculated fund returns for that asset type.
    cumulative_return_df (pd.DataFrame): Dataframe of cumulative returns with datetime index.
    cumulative_outperformance_df (pd.DataFrame): Dataframe of cumulative outperformance with datetime index.

    Methods:
    get_benchmark_returns(): Calculates and sets benchmark_returns attribute.
    get_saa_returns(): Calculates and sets saa_returns attribute.
    get_manager_returns(): Calculates and sets manager_returns attribute.
    get_manager_returns_with_drift(): Calculates and returns manager returns as if only rebalanced on target dates.
    get_all_returns(): Calls all return calculation methods to populate all return attributes.
    create_return_comparison(): Creates a comparison df of all calculated returns.
    get_cumulative_return(): Calculates and sets cumulative_return_df attribute.
    get_cumulative_outperformance_df(): Calculates and sets cumulative_outperformance_df attribute.
    write_csv(path): Writes cumulative_outperformance_df to a CSV file at the given file path.
    return_gen(return_df, allocation_df): Generator function for calculating returns based on given return and allocation data.
    """
    def get_cumulative_return(self):
        self.cumulative_return_df = (
            self.return_comparison_df + 1)[::-1].cumprod(axis=0)[::-1]

    def get_cumulative_outperformance_df(self):
        """
        Calculate the cumulative outperformance of different return series.

        This method calculates the difference between three return series in the `cumulative_return_df` dataframe:
        1. "SAA Returns" versus "Benchmark Returns"
        2. "Manager Returns" versus "SAA Returns"
        3. "Manager Returns With Drift" versus "Manager Returns"

        The resulting outperformance values are stored in the `cumulative_outperformance_df` dataframe.
        """
        self.cumulative_outperformance_df = pd.DataFrame(
            {"SAA vs Benchmark": self.cumulative_return_df["SAA Returns"] - self.cumulative_return_df["Benchmark Returns"],
             "Manager vs SAA": self.cumulative_return_df["Manager Returns"] - self.cumulative_return_df["SAA Returns"],
             "Manager Drift vs Daily Rebalance": self.cumulative_return_df["Manager Returns With Drift"] - self.cumulative_return_df["Manager Returns"]})

    def write_csv(self, path):
        self.cumulative_outperformance_df.to_csv(path)
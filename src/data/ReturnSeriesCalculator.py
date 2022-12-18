import pandas as pd
import numpy as np

class ReturnSeriesCalculator:

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

        self.benchmark_returns = pd.concat(

            [df for df in self.return_gen(
                self.index_returns.loc[:, self.benchmark_weights.columns],
                self.benchmark_weights)]

        ).sum(axis=1)

        assert np.isclose(
            (self.index_returns.iloc[0, 0:5] * self.benchmark_weights.iloc[0]).sum(),
            self.benchmark_returns[0])

    def get_saa_returns(self):

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
        np.testing.assert_array_equal(fund_returns_df.columns, fund_weights.columns)

        for date, fund_returns in fund_returns_df.iterrows():

            fund_allocation = fund_weights.iloc[

                len(fund_weights.index) -
                fund_weights.index[::-1].searchsorted(
                    date, side='left')]

            assert date > fund_allocation.name

            yield pd.DataFrame(fund_returns).T * fund_allocation

    def get_manager_returns(self):

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

        fund_saa_allocation = pd.concat(temp_fund_allocation_list, axis=1)

        # weights should sum to 1 across all assets now instead of within asset
        np.testing.assert_array_almost_equal(
            fund_saa_allocation.sum(axis=1),
            np.ones(len(fund_saa_allocation)))

        self.manager_returns = pd.concat(

            [df for df in
             self.fund_return_gen(fund_returns_df=self.fund_returns,
                                  fund_weights=fund_saa_allocation)],
            axis=0

        ).sum(axis=1)

    def create_return_comparison_df(self):

        self.return_comparison_df = pd.DataFrame(
            {"Benchmark Returns": self.benchmark_returns,
             "SAA Returns": self.saa_returns,
             "Manager Returns": self.manager_returns})

    def write_csv(self, path):
        self.return_comparison_df.to_csv(path)
import pandas as pd
import numpy as np

import load_and_impute
from ReturnSeriesCalculator import ReturnSeriesCalculator

def main():
    benchmark_weights, saa_weights, manager_weights, returns = load_and_impute.get_data()

    ReturnCalc = ReturnSeriesCalculator(benchmark_weights, saa_weights, manager_weights, returns)
    ReturnCalc.get_saa_returns()
    ReturnCalc.get_benchmark_returns()
    ReturnCalc.get_manager_returns()
    ReturnCalc.get_manager_returns_with_drift()

    ReturnCalc.create_return_comparison_df()
    ReturnCalc.write_csv("../../data/processed/returns.csv")


if __name__ == '__main__':

    main()
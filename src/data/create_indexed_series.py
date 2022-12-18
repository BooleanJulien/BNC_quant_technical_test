import pandas as pd
import numpy as np
import sys

import load_and_impute
from Calculators import IndexCalculator

def main(date="2020-03-06"):
    benchmark_weights, saa_weights, manager_weights, returns = load_and_impute.get_data()

    IndexCalc = IndexCalculator(benchmark_weights, saa_weights, manager_weights, returns)
    IndexCalc.get_all_returns()

    IndexCalc.get_cumulative_return()
    IndexCalc.get_indexed_df(date)
    IndexCalc.write_csv("../../data/processed/indexed_performance.csv")


if __name__ == '__main__':
    assert len(sys.argv) in [1, 2], "One str date argument is allowed"
    # print(sys.argv)
    # print(len(sys.argv))

    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1])
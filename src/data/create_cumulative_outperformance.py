import pandas as pd
import numpy as np
import sys

import load_and_impute
from Calculators import CumulativeOutperformanceCalculator

def main():
    benchmark_weights, saa_weights, manager_weights, returns = load_and_impute.get_data()

    CumulativeCalc = CumulativeOutperformanceCalculator(benchmark_weights, saa_weights, manager_weights, returns)
    CumulativeCalc.get_all_returns()

    CumulativeCalc.get_cumulative_return()
    CumulativeCalc.get_cumulative_outperformance_df()
    CumulativeCalc.write_csv("../../data/processed/cumulative_outperformance.csv")


if __name__ == '__main__':

    main()

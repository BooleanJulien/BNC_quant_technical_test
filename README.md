![](data/external/nb_logo_2.png)

# Portfolio Attribution Analysis
==============================

## Overview

This project is designed to conduct a portfolio attribution analysis for a given set of securities. The analysis will consist of three tasks:

1. Generating a portfolio return series
2. Generating a portfolio index series
3. Generating a cumulative outperformance series

## Getting Started

### Prerequisites

In order to run this project, you will need the following software and libraries:

- Python 3
- NumPy
- Pandas

### Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/BooleanJulien/BNC_quant_technical_test.git
```
2. Navigate to the project directory:

```bash
cd portfolio_attribution
```
3. Install the required libraries:

```bash
pip install -r requirements.txt
```

### Usage

Before running the analysis, make sure to add your copy of `Technical Test - Portfolio Attribution.xlsm` to the `data/raw` directory.

To run the analysis, navigate to the `src/data` directory and execute the following command:

```bash
python run_all.py
```

The results of the analysis will be saved to the `data/processed` directory as three CSV files:

1. `returns.csv`: Contains the portfolio return series.
2. `indexed_performance.csv`: Contains the portfolio index series.
3. `cumulative_outperformance.csv`: Contains the cumulative outperformance series.

## Additional Resources

You may find the notebooks in the [notebooks](notebooks) directory useful for exploratory data analysis and understanding the inner workings of the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank the reviewers for their time and consideration in evaluating this project. Your feedback is greatly appreciated.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

import subprocess
import sys

def main(index_date="2020-03-06"):

    print("Running load_and_impute.py...") 
    subprocess.run(["python", "load_and_impute.py"])

    print("Running create_return_series.py...")
    subprocess.run(["python", "create_return_series.py", index_date])

    print("Running create_indexed_series.py...")
    subprocess.run(["python", "create_indexed_series.py"])

    print("Running create_cumulative_outperformance.py...")
    subprocess.run(["python", "create_cumulative_outperformance.py"])

    print("Done. Check data/processed folder for results csvs.")

if __name__ == '__main__':
    assert len(sys.argv) in [1, 2], "One str date argument is allowed"
    # print(sys.argv)
    # print(len(sys.argv))

    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1])
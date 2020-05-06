import argparse
import os

import pandas as pd

model_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(model_dir)
path = os.path.join(project_dir, 'fire_data', 'PlotsEnviroBiomass.xlsx')


def main():
    df = pd.read_excel(path, sheet_name="SEQI_YOSE_2014_Table")
    print(df.index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("")
    main()

import argparse
import os

import pandas as pd

model_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(model_dir)
data_dir = os.path.join(project_dir, 'fire_data')
path = os.path.join(data_dir, 'soil_moisture', 'SoilMoisture_AllDates_through_Jun_2018.xlsx')


def main():
    df = pd.read_excel(path)
    print(df.index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()
    parser.add_argument("")

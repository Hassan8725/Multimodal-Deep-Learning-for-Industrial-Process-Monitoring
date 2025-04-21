import pandas as pd
import numpy as np
import os
import random
from multimodal.helper_functions import  read_config, reproducible_comp, detect_peaks


# dataloader class to combine sensor data
class DataLoader:

    # create meta_file, read excel file and scan accelerometer data
    def __init__(self, meta_file: str, acc_data_folders: [str], sheet_name_meta: str, sheet_name_curves:str):
        self._read_excel(meta_file,sheet_name_meta,sheet_name_curves)
        self._scan_acc_data_files(acc_data_folders)

    # read excel file "StampingDataset_metafile" and write to meta_file
    def _read_excel(self, meta_file: str, sheet_name_meta: str, sheet_name_curves:str):
        # read sheet "Plan" with meta data of experiments (acceleration ID to curveID sheet)
        self.meta = pd.read_excel(meta_file, sheet_name=sheet_name_meta)
        # read sheet "Curves_fn" with fn data from integrated sensor
        df_curves = pd.read_excel(meta_file, sheet_name=sheet_name_curves)
        # find all columns with fn values
        value_columns = [item for item in df_curves.columns if str(item).startswith("Value")]
        # find curve ids of fn data, ".unique()" because every curve id has two rows (f, n)
        curve_ids = df_curves["fnId"].unique()
        # make sure to have double the amount of curves to curve ids (every curve id has two rows of data)
        assert len(curve_ids) * 2 == len(df_curves)
        # create a new dict for single fn curves
        self.fn_curves = {}

        # find every curve id in curve ids
        for curve_id in curve_ids:
            # write rows in df_curves with matching curve id
            view = df_curves.loc[df_curves["fnId"] == curve_id, value_columns]
            # transpose matrix (due to raw data)
            self.fn_curves[curve_id] = view.values.T

    # scan accelerometer data files
    def _scan_acc_data_files(self, folders: [str]):
        # create new dicts for csv files and config_files
        self.csv_files = {}
        self.config_files = {}
        # find all folders (not necessary anymore, because now all files in one folder)
        for folder in folders:
            # find all .csv files and put in csv_files dict
            csv_files = [file for file in os.listdir(folder) if file.endswith("csv")]
            # get name (accId) of file and write in acc_ids
            acc_ids = [int(item.split("_")[1][:-4]) for item in csv_files]
            # find matching acc ids from config files and csv files
            for file, acc_id in zip(csv_files, acc_ids):
                self.csv_files[acc_id] = os.path.join(folder, file)
                self.config_files[acc_id] = os.path.join(folder, file[:-4] + "_config.txt")

    # check own length
    def __len__(self):
        return len(self.meta)

    # set current position to zero
    def __iter__(self):
        self._i = 0
        return self

    # check index
    def index(self):
        return self.meta.iloc[self._i - 1]["accId"]
    
    # read config files of accelerometer data, return rate in Hz and scale in g
    def acc_config(self, file):
        # open files
        with open(file, "r") as f:
            # read lines in file
            lines = f.readlines()
        # split data by ","
        _, _, rate, scale = lines[0].split(",")
        # get rid of " g" and " Hz" and get values
        rate = float(rate.split(":")[-1][:-2])
        scale = float(scale.split(":")[-1][:-2])
        # return rate and scale values
        return rate, scale


    
    

    # stop if at end of list
    def __next__(self):
        if self._i == len(self):
            raise StopIteration()
        res = self[self._i]
        self._i += 1
        return res

    # write meta, acc and fn data to dataframes
    def __getitem__(self, i: int):
        # make sure i is not greater than list length
        assert i < len(self)
        # Read new meta data
        row = self.meta.loc[i, :]

      
        # check if there is any acc data
        acc_id = int(row["accId"])
        # read acc data for each acc_id in new meta data
        if acc_id not in self.csv_files.keys():
            # if no data is present set to false
            acc_present = False
            # no data to config
            config = None
            # no data to acc_data
            acc_data = None

        else:
            # write acc data from csv files with matching acc id
            acc_data = pd.read_csv(self.csv_files[acc_id])
            # drop first two columns (due to raw data) and write into same list
            acc_data.drop(acc_data.columns[[0, 1]], axis=1, inplace=True)
            # Read config for acc id and add meta data to config
            config = tuple([*self.acc_config(self.config_files[acc_id]),
                            *row[["accId","fnId","die_size", "position", "category", "lubrication", "slug_pos"]]])
            # if data is present set to true
            acc_present = True

        # check if there is any fn data
        fn_id = row["fnId"]
        # read data fn data for each fn_id in new meta data
        try:
            # write fn data with matching fn id
            fn_data = self.fn_curves[fn_id]
            # if data is present set to true
            fn_present = True
        # if there is no data
        except KeyError:
            # get some other fn_data to get correct shape
            some_item = self.fn_curves[self.fn_curves.keys().__iter__().__next__()]
            # fill array with zeros of some item with fn data
            fn_data = np.zeros_like(some_item)
            # if no data is present set to false
            fn_present = False
        # return meta data, acc data, fn data, variable if acc data present, variable if fn data present
        return (acc_present, config, acc_data, acc_id), (fn_present, fn_data)

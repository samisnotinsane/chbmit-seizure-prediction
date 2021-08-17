import os
import re
import random
import pprint
import mne
from mne.io.edf.edf import RawEDF
import numpy as np
from datetime import datetime

class DataParser:
    def __init__(self, rootdir, patient_id, *args, **kwargs):
        self.root = rootdir
        self.patient_id = patient_id

    def _parse_summary(self, fpath) -> list:
        file_metadata = []
        with open(fpath) as f:
            content_str = f.read()
            regex = re.compile(r'^\Z|\*+') # match empty string or literal asterisks
            filtered = [x for x in content_str.split('\n') if not regex.search(x)]
            regex = re.compile('Channel \d+') # match channel numbers
            # channels = [x.split(':')[-1].strip() for x in filtered if regex.search(x)]
            regex = re.compile('Data Sampling Rate:')
            # fs = int([x.split(':')[-1].strip() for x in filtered if regex.search(x)][0].split(' ')[0])
            regex = re.compile('^(?!Channel|Data).') # match file names
            file_metas = [x for x in filtered if regex.findall(x)]
            file_meta = {}
            for x in file_metas:
                k, v = x.partition(':')[::2]

                if k == 'Seizure Start Time':
                    file_meta['Seizure Start Time'] = v
                if k == 'Seizure End Time':
                    file_meta['Seizure End Time'] = v
                    tup_meta = {'File Name': file_meta['File Name'], 
                                    'File Start Time': file_meta['File Start Time'], 
                                    'File End Time': file_meta['File End Time'],
                                    'Number of Seizures in File': file_meta['Number of Seizures in File'],
                                    'Seizure Start Time': file_meta['Seizure Start Time'],
                                    'Seizure End Time': file_meta['Seizure End Time']
                                }
                    file_metadata.append(tup_meta)

                if k == 'File Name':
                    file_meta['File Name'] = v.strip()
                if k == 'File Start Time':
                    file_meta['File Start Time'] = v.strip()
                if k == 'File End Time':
                    file_meta['File End Time'] = v.strip()
                if k == 'Number of Seizures in File':
                    if int(v) == 0:
                        if 'Seizure End Time' in file_meta:
                            del file_meta['Seizure End Time']
                        if 'Seizure Start Time' in file_meta:
                            del file_meta['Seizure Start Time']
                        file_meta['Number of Seizures in File'] = 0
                        tup_meta = {'File Name': file_meta['File Name'], 
                                    'File Start Time': file_meta['File Start Time'], 
                                    'File End Time': file_meta['File End Time'],
                                    'Number of Seizures in File': file_meta['Number of Seizures in File']
                                }
                        file_metadata.append(tup_meta)
                    if int(v) > 0:
                        file_meta['Number of Seizures in File'] = int(v.strip())

        return file_metadata

    def _patient_summary(self) -> list:
        regex = re.compile('^chb\d{2}-summary.txt$')
        summary_fname = [x for x in os.listdir(self.root + self.patient_id) if regex.search(x)]
        summary_fpath = self.root + self.patient_id + '/' + summary_fname[0]
        patient_summary = self._parse_summary(summary_fpath)
        return patient_summary

    def print_summary(self):
        pp = pprint.PrettyPrinter()
        summary = self._patient_summary()
        pp.pprint(summary)

    def contains_seizures(self) -> dict:
        """
        Creates a dict with filename and seizure presence status.
        """
        seizure_files = {}
        summary = self._patient_summary()
        for info in summary:
            if info['Number of Seizures in File'] > 0:
                seizure_files[info['File Name']] = True
            else:
                seizure_files[info['File Name']] = False
        return seizure_files

    def seizure_filenames(self) -> list:
        seizure_filenames = []
        filenames = self.contains_seizures()
        for name, is_seizure in filenames.items():
            if is_seizure:
                seizure_filenames.append(name)
        return seizure_filenames

    def no_seizure_filenames(self) -> list:
        no_seizure_filenames = []
        filenames = self.contains_seizures()
        for name, is_seizure in filenames.items():
            if not is_seizure:
                no_seizure_filenames.append(name)
        return no_seizure_filenames

    def _get_raw(self, filepath) -> RawEDF:
        raw = mne.io.read_raw_edf(input_fname=filepath, preload=False, verbose='Error')
        return raw

    def seizure_start_time(self, filename) -> int:
        """
        Returns seizure start time in seconds.
        """
        seizure_start_time = -1
        summary = self._patient_summary()
        for info in summary:
            if info['File Name'] == filename:
                start_time_str = info['Seizure Start Time'].split(' ')[1]
                seizure_start_time = int(start_time_str)
        return seizure_start_time

    def seizure_end_time(self, filename) -> int:
        """
        Returns seizure end time in seconds.
        """
        seizure_end_time = -1
        summary = self._patient_summary()
        for info in summary:
            if info['File Name'] == filename:
                end_time_str = info['Seizure End Time'].split(' ')[1]
                seizure_end_time = int(end_time_str)
        return seizure_end_time

    def mins_in_secs(self, mins:int) -> int:
        return mins*60

    def check_preictal_interval_exists(self, filename:str, interval_time:int) -> bool:
        """
            Returns True if an interval of supplied length in seconds exists before the 
            seizure start time. If the seizure occurs before `interval_time` seconds have passed 
            then this function returns False.
        """
        seizure_start_time = self.seizure_start_time(filename)
        if seizure_start_time == -1:
            return False
        diff = seizure_start_time - interval_time
        if diff < 0:
            return False
        return True
    
    def preictal_interval_times(self, filename:str, interval_time:int) -> int:
        if not self.check_preictal_interval_exists(filename, interval_time):
            raise ValueError('Preictal interval does not exist! Try again with shorter interval_time.')
        preictal_end_time = self.seizure_start_time(filename)
        preictal_start_time = preictal_end_time - interval_time
        return preictal_start_time, preictal_end_time

    def _time_difference(self, time_start:str, time_end:str) -> float:
        "Deprecated. Use `file_start_end_times` instead."
        s1 = time_start
        s2 = time_end
        FMT = '%H:%M:%S'
        tdelta = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
        return tdelta.total_seconds()

    def file_start_end_times(self, filename:str) -> int:
        """
            Returns the start and end times of file in Unix epoch time format.
            Unix epoch time is defined as the number of seconds passed since 01.01.1900
        """
        start_time_str, end_time_str = '', ''
        summary = self._patient_summary()
        for info in summary:
            if info['File Name'] == filename:
                start_time_str = info['File Start Time']
                end_time_str = info['File End Time']
                break
        start_date_time = datetime.datetime.strptime(start_time_str, "%H:%M:%S")
        start_epoch_time = start_date_time - datetime.datetime(1900, 1, 1) # Unix epoch

        end_date_time = datetime.datetime.strptime(end_time_str, "%H:%M:%S")
        end_epoch_time = end_date_time - datetime.datetime(1900, 1, 1)
        return start_epoch_time, end_epoch_time

    def file_duration(self, filename:str) -> int:
        start_time, end_time = self.file_start_end_times(filename)
        return end_time - start_time

    def ex_range(self, lower_bound:int, exclude_lower_bound:int, exclude_upper_bound:int, upper_bound:int) -> int:
        excluded_range = list(range(lower_bound, exclude_lower_bound)) + list(range(exclude_upper_bound, upper_bound))
        return excluded_range

    def is_preictal(self, filename) -> bool:
        res = [x for x in self.seizure_filenames() if (x in filename)]
        return bool(res)

    def random_interictal_interval(self, filename:str, interval_time:int) -> int:
        """
        Return a contiguous interval of interictal times outside of preictal and interictal intervals.
        """
        if self.is_preictal(filename):
            file_start, file_end = self.file_start_end_times(filename)
            excluded_start = self.preictal_interval_times()[0]
            excluded_end = self.seizure_end_time
            file_range = self.ex_range(file_start, excluded_start, excluded_end, file_end)
            start_idx = random.randrange(len(file_range))
            # ...
    
    # TODO: write a method to randomly choose an interictal file and select an interval 
    # given by start and end times.

        





if __name__ == '__main__':
    rootdir = '/Volumes/My Passport/AI_Research/data/physionet.org/files/chbmit/1.0.0/'
    parser = DataParser(rootdir, 'chb01')
    # summary = parser.print_summary()
    seizure_files = parser.seizure_filenames()
    print('Seizure file count:', len(seizure_files))
    no_seizure_files = parser.no_seizure_filenames()
    print('No seizure file count:', len(no_seizure_files))
    start_time = parser.seizure_start_time('chb01_03.edf')
    print('chb01_03.edf seizure start time:', start_time)
    end_time = parser.seizure_end_time('chb01_03.edf')
    print('chb01_03.edf seizure end time:', end_time)
    print('60 mins in secs:', parser.mins_in_secs(60))
    print('15 mins in secs:', parser.mins_in_secs(15))
    print('15 mins interval exists:', parser.check_preictal_interval_exists('chb01_03.edf', parser.mins_in_secs(15)))
    print('15 mins interval exists:', parser.check_preictal_interval_exists('chb01_21.edf', parser.mins_in_secs(15)))
    print('chb01_03.edf preictal start time:', parser.preictal_interval_times('chb01_03.edf', parser.mins_in_secs(15))[0])
    print('chb01_03.edf preictal start time:', parser.preictal_interval_times('chb01_03.edf', parser.mins_in_secs(15))[1])

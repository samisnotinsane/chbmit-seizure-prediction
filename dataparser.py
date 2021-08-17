import os
import re
import pprint
import mne
from mne.io.edf.edf import RawEDF
import numpy as np

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
                seizure_start_time = info['Seizure Start Time']
        return seizure_start_time

    def seizure_end_time(self, filename) -> int:
        """
        Returns seizure end time in seconds.
        """
        seizure_end_time = -1
        summary = self._patient_summary()
        for info in summary:
            if info['File Name'] == filename:
                seizure_end_time = info['Seizure End Time']
        return seizure_end_time

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


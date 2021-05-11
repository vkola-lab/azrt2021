"""
data_from_csv.py
audio dataset reading from CSV output file;
"""
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class AudioDatasetFromCsv(Dataset):
    """dVoice dataset."""

    def __init__(self, csv_in, **kwargs):
        """
        init
        """
        self.csv_in = csv_in
        self.df_dat = pd.read_csv(csv_in)
        self.do_segment_audio = kwargs.get('do_segment_audio', False)
        self.segment_audio = kwargs.get('segment_audio')
        self.segment_audio_kw = kwargs.get('segment_audio_kw', {})

    def __len__(self):
        """
        length
        """
        return len(self.df_dat)

    def __getitem__(self, idx):
        """
        get item
        """
        if self.do_segment_audio:
            fea, start_end = self.segment_audio(self.df_dat.loc[idx,'audio_fn'],
                **self.segment_audio_kw)
            start, end = start_end
        else:
            start = self.df_dat.loc[idx, 'start'] if 'start' in self.df_dat.loc[idx] else None
            end = self.df_dat.loc[idx, 'end'] if 'end' in self.df_dat.loc[idx] else None
            fea = np.load(self.df_dat.loc[idx, 'audio_fn'])
        try:
            if (start is not None and end is not None) and\
                (not np.isnan(start) and not np.isnan(end)):
                start = int(start)
                end = int(end)
                fea = fea[start:end]
        except TypeError as error:
            print(start)
            print(end)
            print(type(start))
            print(type(end))
            raise error
        return fea, self.df_dat.loc[idx, 'label'], self.df_dat.loc[idx, 'patient_id'], start, end

    @property
    def labels(self):
        """
        convert label column to np array;
        """
        return self.df_dat.label.to_numpy()

    @property
    def df_sampling_weights(self):
        """
        convert label to numpy() and add 1 to each;
        """
        return self.df_dat.label.to_numpy() + 1

    @property
    def audio_fns(self):
        """
        convert audio filename columns to np array;
        """
        return self.df_dat['audio_fn'].to_numpy()

    @property
    def transcript_fns(self):
        """
        convert transcript filename columns to np array;
        """
        return self.df_dat['transcript_fn'].to_numpy()

def segment_collate_fn(batch):
    """
    collect audio path, label, patient ID, start, end
    """
    aud = [itm[0] for itm in batch]
    lbl = np.stack([itm[1] for itm in batch])
    pid = np.stack([itm[2] for itm in batch])
    start = np.stack(itm[3] for itm in batch)
    end = np.stack([itm[4] for itm in batch])
    return aud, lbl, pid, start, end

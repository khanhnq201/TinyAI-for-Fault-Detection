import numpy as np
from scipy.io import loadmat
import random
import torch
from torch.utils.data import DataLoader, Dataset
from imblearn.over_sampling import SMOTE
from pathlib import Path
import re
from scipy.signal import hilbert
from scipy.fft import fft, ifft


def analyze_data(Y):
    """Analyze the number and percent of each class in a dataset. 

    Args:
        labels (np.ndarray): The array of labels of the dataset.
    """

    if len(Y) == 0:
        print('[WARNING] We re fucked up cuz no data')
        return
    
    unique_classes, counts = np.unique(Y, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        percentage = (count/len(Y)) *100
        print(f'Class {cls}: {count} samples ({percentage:.2f}%)')

def get_label_from_path(file_path: Path) -> int:
    """Analyze the number and percent of each class in a dataset. 

    Args:
        file_path (path): The path of given file.

    Returns:
        label (int): 4 classes originally, but 3 classes after selecting.
            - 0 for 'Normal'
            - 1 for 'Inner'
            - 2 for 'Outter'
    """
    path_parts = file_path.parts
    if 'Normal' in path_parts:
        print('    [INFO] Label: Normal')
        return 0
    
    #--For full label only--
    # elif 'B' in path_parts:
    #     return 1
    # elif 'IR' in path_parts:
    #     return 2
    # elif 'OR' in path_parts:
    #     return 3
    #------------------------

    elif 'IR' in path_parts:
        print('    [INFO] Label: IR')
        return 1
    elif 'OR' in path_parts:
        print('    [INFO] Label: OR')
        return 2
    return -1

def import_cwru_data(
    file_keys: list,
    sample_length: int,
    overlapping_ratio: float,
    base_path: str = 'CWRU-dataset-main'
):
    """
    Import data from file name list 
    
    Args:
        file_keys (np.arr[str]): file name list in format 'number of file'+'sensor location'.
        sample_length (int): length of each sample importing (2048, 4096 recommended).
        overlapping_ratio (float 0 -> 1): (0.25 for training set recommended).
        base_path (path): the path to data folder in the workspace.

    Returns
    ------
        X (np.arr): data 
        Y (np.arr): labels (wo onehot coding)
    """

    all_samples = []
    all_labels = []
    base_path_obj = Path(base_path)

    if overlapping_ratio > 1 or overlapping_ratio < 0: overlapping_ratio = 0
    step = int(sample_length * (1 - overlapping_ratio))
    if step < 1:
        step = 1

    for key in file_keys:        
        match = re.match(r'(\d+)(\w+)', key)      
        file_num_str, data_key_suffix = match.groups()
        print(f">>> Importing: Record = '{file_num_str}', Key = '{data_key_suffix}'")

        glob_pattern = f'{file_num_str}_*.mat'
        found_files = list(base_path_obj.rglob(glob_pattern))

        if not found_files:
            print(f"    [WARNING] No file for '{file_num_str}'")
            continue
        
        file_path = found_files[0]
        label = get_label_from_path(file_path)

        mat_data = loadmat(file_path)
        
        # Each key in mat file stand for each sensor 
        mat_key_zfill = f'X{file_num_str.zfill(3)}_{data_key_suffix}_time'
        mat_key_normal = f'X{file_num_str}_{data_key_suffix}_time'
        
        if mat_key_zfill in mat_data:
            mat_key = mat_key_zfill
        elif mat_key_normal in mat_data:
            mat_key = mat_key_normal

        time_series = mat_data[mat_key].flatten()
        print(f"    [INFO] Size: {len(time_series)// sample_length}")

        #Overlapping
        num_samples_in_file = 0
        file_samples = []
        for i in range(0, len(time_series) - sample_length + 1, step):
            sample = time_series[i : i + sample_length]
            file_samples.append(sample)
            num_samples_in_file += 1
        
        if num_samples_in_file > 0:
            all_samples.extend(file_samples)
            all_labels.extend([label] * num_samples_in_file)

    X = np.array(all_samples)
    Y = np.array(all_labels)
    
    return X, Y

def data_import(overlapping_ratio, base_path, sample_length, preprocessing): 
    """
    Import train-val-test set (fixed file)
    
    Args:
        sample_length (int): length of each sample importing (2048, 4096 recommended).
        overlapping_ratio (float): for training set only (0.25 recommended).
        base_path (path): the path to data folder in the workspace.
        preprocessing (bool): turn on if you wanna do envelope analysis

    Returns
    ------
        X (np.arr): data 
        Y (np.arr): labels (wo onehot coding)
    """
    if preprocessing:
        sample_length = sample_length *2

    train_files = [
        # --- Normal Data (Label 0) ---
        '97DE', '97FE',   # Normal @ 0HP
        '98DE', '98FE',  # Normal @ 1HP

        # --- Inner Race (IR) Faults (Label 2) ---
        '209DE', '209FE',  # DE, IR, 0.021"
        '210DE',          # DE, IR, 0.021"
        '278DE', '278FE',  # FE, IR, 0.007"
        '280DE', '280BA',  # FE, IR, 0.007"
        '271DE', '271FE', '271BA', # FE, IR, 0.021"
        '276FE', '276BA',  # FE, IR, 0.014"
        '277FE', '277BA',  # FE, IR, 0.014"
        

        # --- Outer Race (OR) Faults (Label 3) ---
        '130DE', '131DE',  # DE, OR (Centred), 0.007"
        '144DE', '144BA',  # DE, OR (Orthogonal), 0.007"
        '145DE', '145FE', '145BA', # DE, OR (Orthogonal), 0.007"
        '156DE', '156FE',  # DE, OR (Opposite), 0.007"
        '310DE', '310FE',  # FE, OR (Orthogonal), 0.007"
        #'309DE',          # FE, OR (Orthogonal), 0.014"
        '311DE', '311FE',  # FE, OR (Orthogonal), 0.014"
        '313DE', '313FE',  # FE, OR (Centred), 0.007"  
    ]

    val_files = [
        # --- Normal Data (Label 0) ---
        '99DE', '99FE',   # Normal @ 2HP

        # --- Inner Race (IR) Faults (Label 2) ---
        '211DE',          # DE, IR, 0.021"
        '279DE',          # FE, IR, 0.007"
        '274FE',          # FE, IR, 0.014"
        '272DE', '272FE', '272BA', # FE, IR, 0.021"
        
        # --- Outer Race (OR) Faults (Label 3) ---
        '132DE',          # DE, OR (Centred), 0.014"
        '146DE', '146FE', '146BA', # DE, OR (Orthogonal), 0.014"
        '159DE',          # DE, OR (Opposite), 0.014"
        '312DE', '312FE',  # FE, OR (Orthogonal), 0.021"
        '315DE',          # FE, OR (Centred), 0.021"
    ]

    test_files = [
        # --- Normal Data (Label 0) ---
        '100DE', '100FE', # Normal @ 3HP

        # --- Inner Race (IR) Faults (Label 2) ---
        '212DE',          # DE, IR, 0.021"
        #'281DE',          # FE, IR, 0.007"
        '275FE',          # FE, IR, 0.014"
        '273DE', '273FE', '273BA', # FE, IR, 0.021"

        # --- Outer Race (OR) Faults (Label 3) ---
        '133DE',          # DE, OR (Centred), 0.021"
        '147DE', '147FE', '147BA', # DE, OR (Orthogonal), 0.021"
        '160DE',          # DE, OR (Opposite), 0.021"
        '317DE', '317FE', #'317BA', # FE, OR (Orthogonal), 0.021"
    ]

    X_train, Y_train = import_cwru_data(train_files, sample_length, overlapping_ratio, base_path)
    print('='*20, 'Data for training', '='*20)
    analyze_data(Y_train)
    print('='*50)
    X_val, Y_val = import_cwru_data(val_files, sample_length, 0, base_path)
    print('='*20, 'Data for validating', '='*20)
    analyze_data(Y_val) 
    print('='*50)
    X_test, Y_test = import_cwru_data(test_files, sample_length, 0, base_path)
    print('='*20, 'Data for testing', '='*20)
    analyze_data(Y_test)

    # if preprocessing:
    _, X_train_final = batch_envelope_analysis(X_train, 12000)
    _, X_val_final = batch_envelope_analysis(X_val, 12000)
    _, X_test_final = batch_envelope_analysis(X_test, 12000)

    return X_train_final, Y_train, X_val_final, Y_val, X_test_final, Y_test

def batch_cepstrum_prewhitening(signals):
    """
    Cepstrum Prewhitening.
    
    Args:
        signals (np.arr): signal at time domain

    Returns:
        signals (np.arr): signal at time domain after whitening
    """
    signals_fft = fft(signals, axis=-1)

    #Remove noise on frequency spectrum
    magnitude = np.abs(signals_fft)
    epsilon = 1e-12
    magnitude[magnitude < epsilon] = epsilon
    whitened_fft = signals_fft / magnitude

    whitened_signals_complex = ifft(whitened_fft, axis=-1)
    return np.real(whitened_signals_complex)

def batch_envelope_analysis(signals, fs):
    """
    Envelope analysis for time domain signal 
    
    Args:
        signals (np.arr): 2D signal at time domain, shape [num_samples, sample_length].
        fs (int): sampling frequency (12k).

    Returns
    ------
        freq_axis (np.arr): axis for plotting
        ses_spectra (np.arr): signals at frequency domain, half of sample length
    """
    num_samples, sample_length = signals.shape

    #whitened_signals = batch_cepstrum_prewhitening(signals)

    analytic_signals = hilbert(signals, axis=-1)
    envelopes = np.abs(analytic_signals)
    envelopes_mean_removed = envelopes - envelopes.mean(axis=1, keepdims=True)
    ses_ffts = fft(envelopes_mean_removed**2, axis=-1)

    N_fft = sample_length
    ses_spectra = 2.0/N_fft * np.abs(ses_ffts[:, 0:N_fft//2])
    freq_axis = np.linspace(0.0, 0.5 * fs, N_fft//2)

    return freq_axis, ses_spectra

def create_imbalanced_data(X_train, Y_train, ratio, num_classes):
    """
    Make data imbalanced (for only training data), reduce the sample number of faulty classes.
    
    Args:
        X_train (np.arr): data
        Y_train (np.arr): label
        ratio (int): the ratio of sample number of normal class and faulty classes.
        num_classes (int): 3 or 4

    Returns
    ------
        X (np.arr): data
        Y (np.arr): label
    """
    _, counts = np.unique(Y_train, return_counts= True)
    faulty_samples = counts[0]//ratio

    X_train_imbalanced = []
    Y_train_imbalanced = []

    for cls in range(num_classes):
        # Retain all the normal samples
        if cls == 0: 
            X_train_class = X_train[np.where(Y_train==cls)[0]]
            X_train_imbalanced.append(X_train_class)
            
            labels = np.full(np.shape(X_train_class)[0], cls)
            Y_train_imbalanced.append(labels)
        
        # Select randomly samples from faulty class
        else:
            cls_indices = np.where(Y_train==cls)[0]
            if len(cls_indices) > faulty_samples:
                selected_cls_indices = np.random.choice (cls_indices, size=faulty_samples, replace=False)
                X_train_class = X_train[selected_cls_indices]
            else:
                X_train_class = X_train[np.where(Y_train==cls)[0]]
            X_train_imbalanced.append(X_train_class)

            labels = np.full(np.shape(X_train_class)[0], cls)
            Y_train_imbalanced.append(labels)

    final_X_train_imbalanced = np.concatenate(X_train_imbalanced, axis=0)
    final_Y_train_imbalanced = np.concatenate(Y_train_imbalanced, axis=0)

    print('='*20, 'Training data after imbalancing', '='*20)
    analyze_data(final_Y_train_imbalanced)

    return final_X_train_imbalanced, final_Y_train_imbalanced

def normalize_data(X_train, X_val, X_test): 
    """ Normalize train, val, test by the information from training set
    """
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)

    X_train_normalized = (X_train - train_mean) / train_std
    X_val_normalized = (X_val - train_mean) / train_std
    X_test_normalized = (X_test - train_mean) / train_std

    return X_train_normalized, X_val_normalized, X_test_normalized

def apply_smote(X_train, Y_train, random_state, sample_length):
    """
    Data augmentation by SMOTE. Make minority classes have equal sample number with majority class.
    """
    smote = SMOTE(random_state=random_state)

    X_train_reshaped = X_train.reshape(-1, sample_length)
    X_train_smoted, Y_train_smoted = smote.fit_resample(X_train_reshaped, Y_train)

    print('='*20, 'SMOTING DATA', '='*20)
    analyze_data(Y_train_smoted)

    return X_train_smoted, Y_train_smoted

class BearingDataset(Dataset):
    def __init__(self, X_data, Y_data, is_train=True): 
        self.data = torch.from_numpy(X_data).float()
        self.labels = torch.from_numpy(Y_data).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def create_dataloaders(X_train, Y_train, 
                       X_val, Y_val, 
                       X_test, Y_test, 
                       sample_length, CNN1D_input, batch_size, random_state):
    """ Load data
    """
    if CNN1D_input:
        X_train_reshaped = np.reshape(X_train, (-1, 1, sample_length))
        X_val_reshaped = np.reshape(X_val, (-1, 1, sample_length))
        X_test_reshaped = np.reshape(X_test, (-1, 1, sample_length))
    else: 
        size = int(sample_length**(1/2))
        X_train_reshaped = np.reshape(X_train, (-1, 1, size, size))
        X_val_reshaped = np.reshape(X_val, (-1, 1, size, size))
        X_test_reshaped = np.reshape(X_test, (-1, 1, size, size))


    train_dataset = BearingDataset(X_train_reshaped, Y_train)
    train_loader = DataLoader(train_dataset, batch_size= batch_size,shuffle=True, num_workers=0)

    val_dataset = BearingDataset(X_val_reshaped, Y_val)
    val_loader = DataLoader(val_dataset, batch_size= batch_size,shuffle=False, num_workers=0)

    test_dataset = BearingDataset(X_test_reshaped, Y_test)
    test_loader = DataLoader(test_dataset, batch_size= batch_size,shuffle=False, num_workers=0)

    print("\n" + "="*50)
    print("DATA DISTRIBUTION SUMMARY")
    print("="*50)
    print(f"Training samples:   {len(train_loader.dataset):>6}")
    print(f"Validation samples: {len(val_loader.dataset):>6}")
    print(f"Test samples:       {len(test_loader.dataset):>6}")
    print(f"Total samples:      {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset):>6}")

    return train_loader, val_loader, test_loader

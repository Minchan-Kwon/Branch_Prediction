import pandas as pd
import numpy as np

class BranchHistory(pd.DataFrame):
    '''
    Class for a branch history dataframe
    Inherits pandas.DataFrame
    '''

    _metadata = ['_csv_path']

    @property
    def _constructor(self):
        return BranchHistory

    def __init__(self, *args, **kwargs):
        self._csv_path = kwargs.pop('csv_path', None)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_csv(cls, csv_path):
        '''
        Creates and returns a BranchHistory object from a given csv file
        '''
        df = pd.read_csv(csv_path)

        if 'branch_pc' not in df.columns or 'taken' not in df.columns:
            raise ValueError("CSV must contain 'branch_pc' and 'taken' columns")

        #Convert the hex strings into decimal integers
        df['branch_pc'] = df['branch_pc'].apply(lambda x: int(x, 16) if isinstance(x, str) else x)
        
        #Create Instance of BranchHistory
        branch_history = cls(df, csv_path=csv_path)
        print("Branch History Loaded")
        return branch_history

    def print_stats(self):
        total_branches = len(self)
        unique_pcs = self['branch_pc'].nunique()
        total_taken = self['taken'].sum()
        total_not_taken = total_branches - total_taken
        taken_rate = self['taken'].mean()

        print("\nGlobal Branch History Statistics")
        print(f"  Total branches executed: {total_branches:,}")
        print(f"  Unique branch PCs: {unique_pcs:,}")
        print(f"  Total taken branches: {total_taken:,}")
        print(f"  Total not-taken branches: {total_not_taken:,}")
        print(f"  Overall taken rate: {taken_rate:.2%}")

    def extract_branch_history(self, branch_pc, history_length, cutoff, max_history = 50000):
        '''
        Extracts previous branch histories of length 'history_length' for a given branch PC
        Args:
            branch_pc: The branch PC to extract training history for
            history_length: The length of the branch history to extract
            cutoff: The number of LSBs to keep from each PC
            max_history: Max number of training datapoints.
        Returns:
            tuple(histories, targets, vocab_size)
            histories: np.ndarray of shape (# of samples, history_length)
            targets: np.ndarray of shape (# of samples, )
            vocab_size: 2^(cutoff + 1)
        '''

        branch_pc_isthere = self['branch_pc'] == branch_pc  #Returns a boolean series. True if the branch_pc exists in that position
        pc_indices = self[branch_pc_isthere].index.tolist() #pc_indices is a list of indices that point to the position of the branch_pc
        print(f"Found {len(pc_indices):,} branch histories for PC 0x{branch_pc:08x} from the global history")

        if len(pc_indices) == 0:
            print(f"No branch history found for the PC 0x{branch_pc:08x}")
            return None, None, None

        #Create mask for LSBs
        mask = 2 ** cutoff - 1 
        pc_cutoff_array = self['branch_pc'].values & mask #Binary bitwise and
        taken_array = self['taken'].values.astype(np.int8)

        #Shift left the LSBs, augment with the branch direction
        encoded_array = (pc_cutoff_array << 1) | taken_array

        histories = []
        targets = []

        for idx in pc_indices:
            #Skip PCs with insufficient history
            if idx < history_length:
                continue
            #Break if number of histories exceed max_history
            if len(histories) > max_history:
                break

            histories.append(encoded_array[idx-history_length:idx].astype(np.int16))
            targets.append(taken_array[idx])

        #Label has to be a float for BCE loss
        histories = np.array(histories, dtype=np.int16)
        targets = np.array(targets, dtype=np.float32)
        vocab_size = 2 ** (cutoff + 1)

        print(f"\nSuccessfully extracted {len(histories):,} training samples")
        print(f"  History length: {history_length}")
        print(f"  PC cutoff bits: {cutoff}")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Histories shape: {histories.shape}")
        print(f"  Targets shape: {targets.shape}")

        return histories, targets, vocab_size

    def print_branch_info(self, branch_pc):
        branch_data = self[self['branch_pc'] == branch_pc]

        if len(branch_data) == 0:
            print(f"No branch history found for PC 0x{branch_pc:08x}")
        
        total = len(branch_data)
        taken = branch_data['taken'].sum()
        taken_rate = taken / total
        not_taken = total - taken

        print(f"Branch PC: 0x{branch_pc:08x}")
        print(f"  Total occurrences: {total:,}")
        print(f"  Taken: {taken:,} ({taken_rate:.2%})")
        print(f"  Not taken: {not_taken:,} ({1-taken_rate:.2%})")

    def print_history_sample(self, branch_pc, history_length, cutoff, index):
        '''
        Prints a sample of the training data

        Args:
            branch_pc: The branch PC to extract training history for
            history_length: The length of the past branch sequence to extract
            cutoff: The number of LSBs to keep from each PC
            index: The index of the sample to display
        '''
        branch_data, _, _ = self.extract_branch_history(branch_pc, history_length, cutoff)

        if len(branch_data) < index + 1:
            print("Index out of range, choose a lower value")
            return
        
        print(f"Sample {index} for PC 0x{branch_pc:08x}")
        print(branch_data[index])
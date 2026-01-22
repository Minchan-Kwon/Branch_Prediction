import pandas as pd
from pathlib import Path 
from collections import defaultdict

class BaselineModel():
    '''
    Baseline model class for extracting H2P branches
    Maintains separate baseline models for unique PCs
    '''
    class SaturatingCounter():
        '''
        States:
            0: Strongly not-taken
            1: Weakly not-taken
            2: Weakly taken
            3: Strongly taken
        State Transition:
            +1 if branch is actually taken
            -1 if branch is not taken
        '''
        def __init__(self, initial_state = 3):  
            self.current_state = initial_state

        def predict(self):
            #Prediction is based on current state
            return 1 if self.current_state > 1 else 0

        def update(self, actual_direction):
            #Update the next state based on actual branch direction
            if actual_direction == 1:
                self.current_state = min(self.current_state + 1, 3)
            else:
                self.current_state = max(self.current_state - 1, 0)
    
    def __init__(self, initial_state = 3):
        #Creates a new instance of the baseline model for each unique PC using the lambda function
        self.initial_state = initial_state
        self.predictors = defaultdict(lambda: self.SaturatingCounter(initial_state))
        self.branch_stats = {}

    def predict(self, data, output_name = 'baseline_prediction.csv'):
        '''
        Args:
            data: A pandas.DataFrame object with columns 'branch_pc', 'taken'
            output_name: The file name of the baseline prediction summary
        '''
        #Reset stats
        self.reset()

        for branch_pc, direction in data.itertuples(index = False):
            #Loads a saturating counter for the given branch_pc, creates one if it doesn't exist
            baseline_model = self.predictors[branch_pc]

            #Make prediction
            prediction = baseline_model.predict()

            if branch_pc not in self.branch_stats:
                #Initialize branch stats for unseen PC
                self.branch_stats[branch_pc] = {
                    'correct': 0,
                    'total': 0,
                    'taken': 0,
                    'not-taken': 0 
                }
            
            #Update stats
            self.branch_stats[branch_pc]['total'] += 1
            self.branch_stats[branch_pc]['taken' if direction == 1 else 'not-taken'] += 1
            if prediction == direction:
                self.branch_stats['correct'] += 1
            
            #Update state
            baseline_model.update(direction)

        #Calculate accuracy and overall stats
        overall_total = 0
        overall_correct = 0
        overall_taken = 0

        for branch_pc in self.branch_stats:
            self.branch_stats[branch_pc]['accuracy'] = self.branch_stats[branch_pc]['correct'] / self.branch_stats[branch_pc]['total']
            self.branch_stats[branch_pc]['taken_rate'] = self.branch_stats[branch_pc]['taken'] / self.branch_stats[branch_pc]['total']
            overall_total += self.branch_stats[branch_pc]['total']
            overall_taken += self.branch_stats[branch_pc]['taken']
            overall_correct += self.branch_stats[branch_pc]['correct']
        
        overall_taken_rate = overall_taken / overall_total
        overall_accuracy = overall_correct / overall_total

        #Print statistics
        print("Baseline Prediction Statistics\n")
        print("Total Predictions: ",overall_total)
        print("Overall Prediction Accuracy: ",overall_accuracy)
        print("Overall Taken Rate: ",overall_taken_rate)

        #Create DataFrame for results
        result_df = None
        if len(self.branch_stats) > 0:
            result_df = pd.DataFrame([
                {
                    'branch_pc': f"0x{pc:8x}",
                    'total_occurrences': stats['total'],
                    'correct_predictions': stats['correct'],
                    'incorrect_predictions': stats['total'] - stats['correct'],
                    'accuracy': stats['accuracy'],
                    'actual_taken': stats['taken'],
                    'actual_not_taken': stats['total'] - stats['taken'],
                    'actual_taken_rate': stats['taken_rate']
                }
                for pc, stats in self.branch_stats.items()])
            
        #Sort by accuracy
        results_df = results_df.sort_values('accuracy')

        #Save as CSV
        csv_dir = Path("../../run/baseline")
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / output_name
        result_df.to_csv(csv_path, index = False)
        print(f"Results saved to {csv_path}")

    #Prints stats for a given PC
    def print_branch_stats(self, branch_pc):
        if self.branch_stats[branch_pc] is None:
            print(f"No PC {branch_pc} Found!")
            return
        print(f"{self.branch_stats[branch_pc]}")

    #Reset counters and stats
    def reset(self):
        self.predictors.clear()
        self.branch_stats.clear()
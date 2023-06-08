import preprocessing.pipelines as pipelines
import pandas as pd
import utilities.dataset_utils as dataset_utils

check = False # Check = True to verify installation, =False for full training.
ncores = 2

list_datasets = [
    'train'
    ]

list_dataset_names = [
        'Train'
    ]

list_dataset_locations = ['datasets/PPBS/train_test.txt']

use_evolutionary = False # True to use evolutionary information (requires hhblits and a sequence database), False otherwise.

pipeline = pipelines.ScanNetPipeline(
        with_atom=True,
        aa_features='pwm' if use_evolutionary else 'sequence',
    )

dataset_table = pd.read_csv('datasets/PPBS/table.csv',sep=',')

list_inputs = []
list_outputs = []
list_weights = []

for dataset, dataset_name, dataset_location in zip(list_datasets, list_dataset_names, list_dataset_locations):
    (list_origins,  # List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
     list_sequences,  # List of corresponding sequences.
     list_resids,  # List of corresponding residue identifiers.
     list_labels) = dataset_utils.read_labels(dataset_location)

#print("list_origins : ",list_origins)
#print("list_sequences :",list_sequences)
#print("list_resids :",list_resids)
#print("list_labels :",list_labels)

inputs,outputs,failed_samples = pipeline.build_processed_dataset(
            'PPBS_%s'%(dataset+'_check' if check else dataset),
            list_origins=list_origins, # Mandatory
            list_resids=list_resids, # Optional
            list_labels=list_labels, # Optional
            biounit=True, # Whether to use biological assembly files or the regular pdb files (asymmetric units). True for PPBS data set, False for BCE data set.
            save = True, # Whether to save the results in pickle file format. Files are stored in the pipeline_folder defined in paths.py
            fresh = False, # If fresh = False, attemps to load pickle files first.
            permissive=True, # Will not stop if some examples fail. Errors notably occur when a biological assembly file is updated.
            ncores = ncores # Number of parallel processes.
        )

print(inputs)
print(outputs)
print(failed_samples)







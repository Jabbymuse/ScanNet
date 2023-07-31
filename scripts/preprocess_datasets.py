import sys,os
sys.path.append(os.getcwd())
def set_num_threads(num_threads=2):
    os.environ["MKL_NUM_THREADS"] = "%s"%num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = "%s"%num_threads
    os.environ["OMP_NUM_THREADS"] = "%s"%num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = "%s"%num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = "%s"%num_threads
    os.environ["NUMBA_NUM_THREADS"] = "%s"%num_threads
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
set_num_threads(num_threads=4)

import preprocessing.pipelines_modifie as pipelines
import utilities.dataset_utils as dataset_utils
import utilities.wrappers as wrappers
import utilities.io_utils as io_utils
import pandas as pd
import numpy as np
import utilities.paths as paths

check = False  # Check = True to verify installation, =False for full training.
all_use_evolutionary = [False]#,True]#[False]
all_motion_vectors = [False,1,2,3,4,5,6,7,8,9,10]  # False # is not working for 1 or 2 which is strange ...
ncores = 32#8
model_acronyms = {'interface':'PPI','epitope': 'PAI', 'idp': 'PIDPI'}
dataset_folders = {'interface':'PPBS','epitope': 'BCE', 'idp': 'PIDPBS'}
full_names = {'interface':'Protein-protein binding sites','epitope': 'B-cell epitopes', 'idp': 'intrinsically disordered protein binding sites'}
biounits = {'interface':True, 'epitope': False, 'idp': False}

list_datasets = {
    'interface': [
    'train',
    'validation_70',
    'validation_homology',
    'validation_topology',
    'validation_none',
    'test_70',
    'test_homology',
    'test_topology',
    'test_none',
],
    'epitope': [f'fold{k}' for k in range(1,6)],
    'idp': [f'fold{k}' for k in range(5)],
}

list_dataset_names = {
    'interface': [
    'Train',
    'Validation (70\%)',
    'Validation (Homology)',
    'Validation (Topology)',
    'Validation (None)',
    'Test (70\%)',
    'Test (Homology)',
    'Test (Topology)',
    'Test (None)'
],
    'epitope': [f'Fold {k}' for k in range(1,6)],
    'idp': [f'Fold {k}' for k in range(1,6)],
}

def fetch_dataset(
        mode,
        dataset,
        motion_vectors,
        use_evolutionary,
        check=False
):
    super_pipeline_name = 'pipeline_ScanNet_aa-%s_atom-%s_frames-%s_Beff-%s' % (
        'both' if max(all_use_evolutionary) else 'sequence',
        'valency',
        'triplet_sidechain',
        500,
    )
    super_pipeline_name += '-motion-%s' % max(all_motion_vectors)
    dataset_short_name = dataset_folders[mode] + '_%s' % dataset + ('_check' if check else '')
    location_processed_dataset = pipelines.pipeline_folder + dataset_short_name + '_%s.data' % super_pipeline_name
    env  = io_utils.load_pickle(location_processed_dataset)
    inputs,outputs,failed_samples = env['inputs'],env['outputs'],env['failed_samples']
    if motion_vectors:
        inputs[2] = wrappers.wrap_list([x[..., :motion_vectors,:] for x in inputs[2]])
    else:
        inputs = inputs[:2] + inputs[3:]
    if use_evolutionary:
        inputs[2] = wrappers.wrap_list([x[..., 20:] for x in inputs[2]])
    else:
        inputs[2] = wrappers.wrap_list([x[..., :20] for x in inputs[2]])
    return inputs,outputs,failed_samples


if __name__ == '__main__':
    super_pipeline = pipelines.ScanNetPipeline(
        with_atom=True,
        aa_features='both' if max(all_use_evolutionary) else 'sequence',
        motion_vectors=all_motion_vectors[-1]
    )
    for mode in dataset_folders.keys():
        for dataset, dataset_name in zip(list_datasets[mode], list_dataset_names[mode]):
            print(mode,dataset_name)
            dataset_location = os.path.join('datasets', dataset_folders[mode], 'labels_%s.txt' % dataset)
            dataset_short_name = dataset_folders[mode] + '_%s' % dataset + ('_check' if check else '')
            # Parse label files
            (list_origins,  # List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
             list_sequences,  # List of corresponding sequences.
             list_resids,  # List of corresponding residue identifiers.
             list_labels) = dataset_utils.read_labels(dataset_location)  # List of residue-wise labels

            if check:
                list_origins = list_origins[:10]
                list_sequences = list_sequences[:10]
                list_resids = list_resids[:10]
                list_labels = list_labels[:10]

            '''
            Build processed dataset. For each protein chain, build_processed_chain does the following:
            1. Download the pdb file (biounit=True => Download assembly file, biounit=False => Download asymmetric unit file).
            2. Parse the pdb file.
            3. Construct atom and residue point clouds, determine triplets of indices for each atomic/residue frame.
            4. If evolutionary information is used, build an MSA using HH-blits and calculates a Position Weight Matrix (PWM).
            5. If labels are provided, aligns them onto the residues found in the pdb file.
            '''
            inputs, outputs, failed_samples = super_pipeline.build_processed_dataset(
                dataset_short_name,
                list_origins=list_origins,  # Mandatory
                list_resids=list_resids,  # Optional
                list_labels=list_labels,  # Optional
                biounit=biounits[mode],
                # Whether to use biological assembly files or the regular pdb files (asymmetric units). True for PPBS data set, False for BCE data set.
                save=True,
                # Whether to save the results in pickle file format. Files are stored in the pipeline_folder defined in paths.py
                fresh=False,  # If fresh = False, attemps to load pickle files first.
                permissive=True,
                # Will not stop if some examples fail. Errors notably occur when a biological assembly file is updated.
                ncores=ncores  # Number of parallel processes.
            )

        print('Checking fetch function')
        for use_evolutionary in all_use_evolutionary:
            for motion_vectors in all_motion_vectors:
                inputs,outputs,failed_samples = fetch_dataset(
                    mode,
                    dataset,
                    motion_vectors,
                    use_evolutionary,
                    check=check)




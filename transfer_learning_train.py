import preprocessing.pipelines_modifie as pipelines
import utilities.dataset_utils as dataset_utils
import utilities.wrappers as wrappers
import network.scannet_modif as scannet
import pandas as pd
import numpy as np
import utilities.paths as paths
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard
from keras.optimizers import Adam
from keras.backend import clear_session
import os,sys,pickle
from train import make_PR_curves

if __name__ == '__main__':
    '''
    Script to train and evaluate ScanNet on the B-cell epitope data set.
    Model is trained via transfer learning, using the PPBS model as starting point.
    Five-fold cross-validation is used, i.e. five models are trained.
    '''
    check = False # Check = True to verify installation, =False for full training.
    train = True # True to retrain, False to evaluate the model shown in paper.
    transfer = True # If False, retrain from scratch.
    freeze = False # If True, evaluate the binding site network without fine tuning.
    use_evolutionary = False # True to use evolutionary information (requires hhblits and a sequence database), False otherwise.
    save_predictions = 'heldout' # ['all','heldout',False] Whether to save the held-out predictions for additional analysis.
    tensorboard = False
    epochs_max = 2 if check else 100
    ncores = 4
    mode = 'epitope'
    assert mode in ['epitope','idp']
    model_acronyms = {'epitope': 'PAI', 'idp': 'PIDPI'}
    dataset_folders = {'epitope':'BCE', 'idp':'PIDPBS'}
    full_names = {'epitope':'B-cell epitopes', 'idp':'intrinsically disordered protein binding sites'}
    Lmax_aas = {'epitope':2120, 'idp': 1352}
    biounits = {'epitope':False,'idp':False}
    Lmax_aa = 256 if check else Lmax_aas[mode]
    ''' 
    Lmax_aa is the maximum length of the protein sequences.
    Sequences longer than Lmax_aa are truncated, sequences shorter are grouped and processed using the protein serialization trick (see Materials and Methods of paper).
    If memory allows it, use the largest protein length found in the dataset.
    In the paper, we used Lmax_aa = 2120, which was the largest antigen in the dataset (two chains of a SARS-CoV-2 spike protein)
    '''

    if train: # Retrain model.
        model_name = 'ScanNet_%s_retrained' % model_acronyms[mode]
        root_model_name = 'ScanNet_PPI' # The initial model.
        if len(sys.argv)>1: # python transfer_learning_train.py 1/2/...
            model_name += '_%s'%sys.argv[1] # Retrain multiple times for error bars.
            root_model_name += '_retrained_%s'%sys.argv[1] # Retrain multiple times for error bars.
        if not use_evolutionary:
            model_name += '_noMSA'
            root_model_name += '_noMSA'
        if not transfer:
            model_name += '_scratch'
        if freeze:
            model_name += '_freeze'
        if check:
            model_name += '_check'

    else: # Evaluate paper model.
        model_name = 'ScanNet_%s' % model_acronyms[mode]
        if not use_evolutionary:
            model_name += '_noMSA'


    model_names = [model_name + '_%s'%k for k in range(5)]


    list_datasets = [
    'fold1',
    'fold2',
    'fold3',
    'fold4',
    'fold5',
    ]

    list_dataset_names = [
        'Fold 1',
        'Fold 2',
        'Fold 3',
        'Fold 4',
        'Fold 5'
    ]


    #%% Gather and preprocess each dataset.

    pipeline = pipelines.ScanNetPipeline(
        with_atom=True,
        aa_features='pwm' if use_evolutionary else 'sequence',
    )


    list_dataset_locations = ['datasets/%s/labels_%s.txt'% (dataset_folders[mode],dataset) for dataset in list_datasets]
    dataset_table = pd.read_csv('datasets/%s/table.csv' % dataset_folders[mode],sep=',')

    list_inputs = []
    list_outputs = []
    list_weights = []

    if save_predictions:
        list_origins_ = []
        list_sequences_ = []
        list_resids_ = []
        list_labels_ = []

    for dataset,dataset_name,dataset_location in zip(list_datasets,list_dataset_names,list_dataset_locations):
        # Parse label files
        (list_origins,# List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
        list_sequences,# List of corresponding sequences.
        list_resids,#List of corresponding residue identifiers.
        list_labels)  = dataset_utils.read_labels(dataset_location) # List of residue-wise labels

        if check:
            list_origins = list_origins[:10]
            list_sequences = list_sequences[:10]
            list_resids = list_resids[:10]
            list_labels = list_labels[:10]

        if save_predictions:
            list_origins_ += list(list_origins)
            list_sequences_ += list(list_sequences)
            list_resids_ += list(list_resids)
            list_labels_ += list(list_labels)

        if dataset_folders[mode] == 'UBS':
            list_labels = wrappers.wrap_list([(labels>=2).astype(int) for labels in list_labels])


        '''
        Build processed dataset. For each protein chain, build_processed_chain does the following:
        1. Download the pdb file (biounit=True => Download assembly file, biounit=False => Download asymmetric unit file).
        2. Parse the pdb file.
        3. Construct atom and residue point clouds, determine triplets of indices for each atomic/residue frame.
        4. If evolutionary information is used, build an MSA using HH-blits and calculates a Position Weight Matrix (PWM).
        5. If labels are provided, aligns them onto the residues found in the pdb file.
        '''
        inputs,outputs,failed_samples = pipeline.build_processed_dataset(
            '%s_%s'%(dataset_folders[mode],dataset+'_check' if check else dataset),
            list_origins=list_origins, # Mandatory
            list_resids=list_resids, # Optional
            list_labels=list_labels, # Optional
            biounit=biounits[mode], # Whether to use biological assembly files or the regular pdb files (asymmetric units). True for PPBS data set, False for BCE data set.
            save = True, # Whether to save the results in pickle file format. Files are stored in the pipeline_folder defined in paths.py
            fresh = False, # If fresh = False, attemps to load pickle files first.
            ncores = ncores,
            permissive=False
        )

        weights = np.array(dataset_table['Sample weight'][ dataset_table['Set'] == dataset_name ] )
        if check:
            weights = weights[:10]
        weights = np.array([weights[b] for b in range(len(weights)) if not b in failed_samples])

        list_inputs.append(inputs)
        list_outputs.append(outputs)
        list_weights.append( weights )

    '''
    Input format:
    [
    input[0]: amino acid-wise triplets of indices for constructing frames ([Naa,3], integer).
    input[1]: amino acid-wise attributes: either one-hot-encoded sequence or position-weight matrix ([Naa,20/21],).
    input[2]: amino acid sequence index ([Naa,1], integer).
    input[3]: amino acid point cloud for constructing frames ([2*Naa+1,3] Calpha,sidechains).
    input[4]: atom-wise triplets of indices for constructing frames ([Natom,3], integer).
    input[5]: amino acid-wise attributes: atom group type ([Natom,1] integer)
    input[6]: atom sequence index ([Natom,1], integer).
    input[7]: atom point cloud for constructing frames ([Natom+Nvirtual atoms,3]).
    ]

    Output format:
    output: amino acid-wise binary label, one-hot-encoded as [Naa,2].

    Weight format: List of chain-wise weights (optional).
    '''

    all_cross_predictions = []
    if save_predictions == 'all':
        all_predictions = []

    for k in range(5): # 5-fold training/evaluation.
        if tensorboard:
            clear_session()
        not_k = [i for i in range(5) if i!=k]
        train_inputs = [np.concatenate([list_inputs[i][j] for i in not_k]) for j in range( len(list_inputs[0]) ) ]
        train_outputs = np.concatenate([list_outputs[i] for i in not_k])
        train_weights = np.concatenate([list_weights[i] for i in not_k])

        test_inputs = list_inputs[k]
        test_outputs = list_outputs[k]
        test_weights = list_weights[k]

        if train:
            #%% Load initial model.
            model = wrappers.load_model(paths.model_folder + root_model_name, Lmax=Lmax_aa, load_weights=transfer) # If transfer, load weights from root network; otherwise, only load architecture and loss.
            if not freeze:
                optimizer = Adam(lr=1e-4, beta_2=0.99, epsilon=1e-4)
                model.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
                    'categorical_crossentropy', 'categorical_accuracy']),  # Recompile model with an optimizer with lower learning rate.
                extra_params = {'batch_size':1,'epochs':epochs_max}

                # %% Train!
                extra_params['validation_data'] = (
                    test_inputs, test_outputs, test_weights)
                extra_params['callbacks'] = [
                    EarlyStopping(monitor='val_categorical_crossentropy', min_delta=0.001, patience=4,
                                  verbose=1, mode='min', restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_categorical_crossentropy', factor=0.5,
                                      patience=2, verbose=1, mode='min', min_delta=0.001, cooldown=1)
                ]
                if tensorboard:
                    logdir = os.path.join(paths.library_folder ,'training_logs/', model_name, f'fold{k+1}')
                    extra_params['callbacks'].append(
                        TensorBoard(
                            log_dir=logdir,
                            histogram_freq=1,
                            batch_size=1,
                            write_graph=True,
                            write_images=False,
                            embeddings_freq=0,
                            embeddings_layer_names=None,
                            embeddings_metadata=None,
                            embeddings_data=None,
                            update_freq='epoch',
                                ) )
                print('Starting training for fold %s...' % k)
                history = model.fit(train_inputs, train_outputs,sample_weight=train_weights, **extra_params)
                print('Training completed for fold %s! Saving model' % k)
                model.save(paths.model_folder + model_names[k])
        else:
            # No training. Load trained model.
            model = wrappers.load_model(paths.model_folder + model_names[k], Lmax=Lmax_aa)

        # %% Predict for test set and evaluate performance.
        if save_predictions == 'all':
            print('Performing predictions on the entire set for fold %s...' % (k + 1))
            predictions = np.concatenate([model.predict(
                inputs_,
                return_all=False,
                # Only return the binding site probability p. return_all=True gives [1-p,p] for each residue.
                batch_size=1) for inputs_ in list_inputs])

        print('Performing predictions on the test set for fold %s...'%(k+1))
        test_predictions = model.predict(
                test_inputs,
                return_all=False,
                # Only return the binding site probability p. return_all=True gives [1-p,p] for each residue.
                batch_size=1)
        all_cross_predictions.append(test_predictions)
        if save_predictions == 'all':
            all_predictions.append(predictions)


    all_cross_predictions = np.concatenate(all_cross_predictions)
    if save_predictions == 'all':
        all_predictions = wrappers.wrap_list([np.stack([all_predictions[k_][i] for k_ in range(5)],axis=1) for i in range(len(all_cross_predictions))])
    all_labels = np.concatenate([wrappers.wrap_list([
            np.argmax(label, axis=1)[:Lmax_aa]  # Back to 0-1 labels from one-hot encoding.
            for label in list_outputs[i]]) for i in range(5)] )
    all_weights = np.concatenate(list_weights)

    if save_predictions:
        list_source_dataset = []
        for k,list_weight in enumerate(list_weights):
            list_source_dataset += [list_dataset_names[k]] * len(list_weights)
        env = {
            'list_origins': list_origins_,
            'list_sequences': list_sequences_,
            'list_resids': list_resids_,
            'list_labels' : list_labels_,
            'list_predictions': all_cross_predictions,
            'list_weights': all_weights,
            'list_source_dataset': list_source_dataset,
        }
        if save_predictions == 'all':
            env['list_all_predictions'] = all_predictions
        os.makedirs(paths.predictions_folder,exist_ok=True)
        pickle.dump(env, open(os.path.join(paths.predictions_folder, 'predictions_%s_%s.pkl' % (mode, model_name)),'wb' ) )

    if not os.path.isdir(paths.library_folder + 'plots/'):
        os.mkdir(paths.library_folder + 'plots/')

    fig, ax,_ = make_PR_curves(
        [all_labels],
        [all_cross_predictions],
        [all_weights],
        ['Cross-validation'],
        title='%s prediction: %s' % (full_names[mode],model_name),
        figsize=(10, 10),
        margin=0.05, grid=0.1
        , fs=25)

    fig.savefig(paths.library_folder + 'plots/PR_curve_%s_%s.png' % (mode,model_name), dpi=300)









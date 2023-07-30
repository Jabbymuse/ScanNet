import preprocess_datasets
import utilities.wrappers as wrappers
import network.scannet_modif as scannet
import pandas as pd
import numpy as np
import utilities.io_utils as io_utils
import utilities.paths as paths
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import sys
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.backend import clear_session
import argparse
from train import make_PR_curves

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='interface',type=str)
    parser.add_argument('--repeat',default=0,type=int)
    parser.add_argument('--motion_vectors',default=0,type=int)
    parser.add_argument('--use_evolutionary',action='store_true')
    parser.add_argument('--tensorboard',action='store_true')
    parser.add_argument('--motion_architecture',default=0,type=int)
    parser.add_argument('--check', action='store_true')
    return parser.parse_args()



if __name__ == '__main__':
    '''
    Script to train and evaluate ScanNet on the Protein-protein binding site data set.
    Model is trained from scratch.
    '''
    args = parse_args()
    Lmax_aas = {'interface':1024,'epitope':2120, 'idp': 1352}
    Lmax_aa = 256 if args.check else Lmax_aas[args.mode]
    epochs_max = 100
    save_predictions = True

    architectures = [
        {'nembedding_motion':6, 'nlayers_motion':2,'use_bn':False},
        {'nembedding_motion': 16, 'nlayers_motion': 2, 'use_bn': True},
        {'nembedding_motion': 32, 'nlayers_motion': 2, 'use_bn': True},
    ]


    model_name = 'ScanNet_%s_grid' % preprocess_datasets.model_acronyms[args.mode]
    if args.repeat>0:
        model_name += '_%s'% args.repeat # Retrain multiple times for error bars.
    if args.motion_vectors:
        model_name += '_motion_%s' % args.motion_vectors
        model_name += '_arch_%s' % args.motion_architecture
    if not args.use_evolutionary:
        model_name += '_noMSA'
    if args.check:
        model_name += '_check'
    if args.mode != 'interface':
        base_model_name = model_name.replace(preprocess_datasets.model_acronyms[args.mode],preprocess_datasets.model_acronyms['interface'])


    dataset_table = pd.read_csv('datasets/%s/table.csv' % preprocess_datasets.dataset_folders[args.mode],sep=',')

    list_inputs = []
    list_outputs = []
    list_weights = []

    for dataset,dataset_name in zip(preprocess_datasets.list_datasets[args.mode],preprocess_datasets.list_dataset_names[args.mode]):
        inputs, outputs, failed_samples = preprocess_datasets.fetch_dataset(
                args.mode,
                dataset,
                args.motion_vectors,
                args.use_evolutionary,
                check=args.check
        )
        weights = np.array(dataset_table['Sample weight'][ dataset_table['Set'] == dataset_name ] )
        if args.check:
            weights = weights[:10]
        weights = np.array([weights[b] for b in range(len(weights)) if not b in failed_samples])
        list_inputs.append(inputs)
        list_outputs.append(outputs)
        list_weights.append(weights)

    if args.mode == 'interface':
        train_inputs = list_inputs[0]
        train_outputs = list_outputs[0]
        train_weights = list_weights[0]
        validation_inputs = [np.concatenate([list_inputs[i][j] for i in [1,2,3,4] ] ) for j in range( len(list_inputs[0]) ) ]
        validation_outputs = np.concatenate([list_outputs[i] for i in [1,2,3,4]])
        validation_weights = np.concatenate([list_weights[i] for i in [1,2,3,4]])

        model, extra_params = scannet.initialize_ScanNet(
            train_inputs,
            train_outputs,
            with_atom=True, # Whether to use atomic coordinates or not.
            Lmax_aa=Lmax_aa, # Maximum protein length used for training
            K_aa=16, # Size of neighborhood for amino acid Neighborhood Embedding Module (NEM)
            K_atom=16, # Size of neighborhood for atom Neighborhood Embedding Module (NEM)
            K_graph=32, # Size of neighborhood for Neighborhood Attention Module (NAM)
            Dmax_aa=11., # Cut-off distance for the amino acid NEM. Only used when initializing the aa gaussian kernels.
            Dmax_atom=4., # Cut-off distance for the atom NEM. Only used when initializing the gaussian kernels.
            Dmax_graph=13., # Cut-off distance for the amino acid NAM. Only used when initializing the gaussian kernels.
            N_aa=32, # Number of gaussian kernels for amino acid NEM
            N_atom=32, # Number of gaussian kernels for atom NEM
            N_graph=32, # Number of gaussian kernels for amino acid NAM
            nfeatures_aa=21 if args.use_evolutionary else 20, # Number of amino acid-wise input attributes.
            nfeatures_atom=12, # Number of atom-wise input attributes (categorical variable).
            nembedding_atom=12, # Dimension of atom attribute embedding. If = nfeatures_atom, use non-trainable one-hot-encoding. # possible changes
            nembedding_aa=32, # Dimension of amino acid attribute embedding.
            nembedding_graph=1, # Number of values per edge for the NAM.
            dense_pooling=64, # Number of channels for atom -> amino acid pooling operation.
            nattentionheads_pooling=64,  # Number of attention heads for atom -> amino acid pooling operation.
            nfilters_atom=128, # Number of atomic spatio-chemical filters
            nfilters_aa=128, # Number of amino acid spatio-chemical filters
            nfilters_graph=2, # Number of outputs for NAM.
            nattentionheads_graph=1, # Number of attention heads for NAM.
            filter_MLP=[32], # Dimensionality reduction (trainable dense layer) applied after amino acid NEM and before NAM.
            covariance_type_atom='full', # Full or diagonal covariance matrix for atom NEM module
            covariance_type_aa='full', # Full or diagonal covariance matrix for amino acid NEM module
            covariance_type_graph='full', # Full or diagonal covariance matrix for graph NEM module
            activation='relu', # Activation function
            coordinates_atom=['euclidian'], # Local coordinate system used for the atom NEM
            coordinates_aa=['euclidian'], # Local coordinate system used for the amino acid NEM
            frame_aa='triplet_sidechain', # Choice of amino acid frames (backbone-only also supported).
            coordinates_graph=['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'], # Local coordinate system used for the amino acid NAM
            index_distance_max_graph=8, # Maximum sequence distance used.
            l12_atom=2e-3, # Sparse regularization for atom NEM.
            l12_aa=2e-3, # Sparse regularization for amino acid NEM.
            l12_pool=2e-3, # Sparse regularization for atom to amino acid pooling.
            optimizer='adam', # Optimizer.
            batch_size=1, # Batch size.
            epochs=epochs_max,  # Maximum number of epochs
            initial_values_folder = paths.initial_values_folder,
            save_initial_values= False if args.check else True, # Cache the initial Gaussian kernels for next training.
            n_init=2, # Parameter for initializing the Gaussian kernels. Number of initializations for fitting the GMM model with sklearn. 10 were used for the paper.
            motion_vectors=args.motion_vectors,
            nembedding_motion = architectures[args.motion_architecture]['nembedding_motion'],
            nlayers_motion = architectures[args.motion_architecture]['nlayers_motion'],
            bn_motion = architectures[args.motion_architecture]['use_bn'],
        )

        #%% Train!
        extra_params['validation_data'] = (
            validation_inputs, validation_outputs, validation_weights)
        extra_params['callbacks'] = [
            EarlyStopping(monitor='val_categorical_crossentropy', min_delta=0.001, patience=5,
                          verbose=1, mode='min', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_categorical_crossentropy', factor=0.5,
                              patience=2, verbose=1, mode='min', min_delta=0.001, cooldown=2),
        ]
        if args.tensorboard:
            # Tensorboard visualization
            extra_params['callbacks'].append( TensorBoard(f"training_logs/{args.mode}/{model_name}/",histogram_freq=1) )
        # Fitting
        history = model.fit(train_inputs, train_outputs,sample_weight=train_weights, **extra_params)
        print('Training completed! Saving model')
        model.save(paths.model_folder +model_name)

        print('Performing predictions on the test set...')
        test_predictions = [
            model.predict(
                list_inputs[i],
                return_all=False, # Only return the binding site probability p. return_all=True gives [1-p,p] for each residue.
                batch_size=1
            )
            for i in [5,6,7,8]
        ]
        test_labels = [ wrappers.wrap_list([
                    np.argmax(label, axis=1)[:Lmax_aa] # Back to 0-1 labels from one-hot encoding and truncate to Lmax.
                    for label in list_outputs[i]]) for i in [5,6,7,8] ]


        test_weights = [list_weights[i] for i in [5,6,7,8] ]
        print('Evaluating predictions on the test set...')

    else:
        test_predictions = []
        test_labels = [ wrappers.wrap_list([
                    np.argmax(label, axis=1)[:Lmax_aa] # Back to 0-1 labels from one-hot encoding and truncate to Lmax.
                    for label in list_outputs[i]]) for i in range(5) ]
        test_weights = list_weights

        for k in range(5):  # 5-fold training/evaluation.
            model_name_ = model_name + '_%s'%k
            if args.tensorboard:
                clear_session()
            not_k = [i for i in range(5) if i != k]
            train_inputs = [np.concatenate([list_inputs[i][j] for i in not_k]) for j in range(len(list_inputs[0]))]
            train_outputs = np.concatenate([list_outputs[i] for i in not_k])
            train_weights = np.concatenate([list_weights[i] for i in not_k])
            val_inputs = list_inputs[k]
            val_outputs = list_outputs[k]
            val_weights = list_weights[k]
            model = wrappers.load_model(paths.model_folder + base_model_name, Lmax=Lmax_aa,load_weights=True)  # If transfer, load weights from root network; otherwise, only load architecture and loss.
            optimizer = Adam(lr=1e-4, beta_2=0.99, epsilon=1e-4)
            model.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
                'categorical_crossentropy',
                'categorical_accuracy']),  # Recompile model with an optimizer with lower learning rate.
            extra_params = {'batch_size': 1, 'epochs': epochs_max}
            # %% Train!
            extra_params['validation_data'] = (
                val_inputs, val_outputs, val_weights)
            extra_params['callbacks'] = [
                EarlyStopping(monitor='val_categorical_crossentropy', min_delta=0.001, patience=4,
                              verbose=1, mode='min', restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_categorical_crossentropy', factor=0.5,
                                  patience=2, verbose=1, mode='min', min_delta=0.001, cooldown=1)
            ]
            if args.tensorboard:
                # Tensorboard visualization
                extra_params['callbacks'].append( TensorBoard(f"training_logs/{args.mode}/{model_name}/",histogram_freq=1) )
            print('Starting training for fold %s...' % k)
            history = model.fit(train_inputs, train_outputs,sample_weight=train_weights, **extra_params)
            print('Training completed for fold %s! Saving model' % k)
            model.save(paths.model_folder + model_name_)
            test_predictions.append( model.predict(
                    val_inputs,
                    return_all=False,
                    # Only return the binding site probability p. return_all=True gives [1-p,p] for each residue.
                    batch_size=1) )



    if not os.path.isdir(paths.library_folder + 'plots/'):
        os.mkdir(paths.library_folder + 'plots/')
    fig,ax,AUCPRs = make_PR_curves(
            test_labels,
            test_predictions,
            test_weights,
            preprocess_datasets.list_datasets[args.mode][-5:],
            title = 'Protein-protein binding site prediction: %s'%model_name,
            figsize=(10, 10),
            margin=0.05,grid=0.1
            ,fs=16)

    _,_,AUCPR_all = make_PR_curves(
            [np.concatenate(test_labels)],
            [np.concatenate(test_predictions)],
            [np.concatenate(test_weights)],
            ['All'],
            title = 'Protein-protein binding site prediction: %s'%model_name,
            figsize=(10, 10),
            margin=0.05,grid=0.1
            ,fs=16)
    dict_AUCPRs = dict(zip(preprocess_datasets.list_datasets[args.mode][-5:], AUCPRs))
    dict_AUCPRs['All'] = AUCPR_all[0]

    fig.savefig(paths.library_folder + 'plots/PR_curve_%s_%s.png'%(preprocess_datasets.model_acronyms[args.mode],model_name),dpi=300)

    results_file = paths.library_folder + 'plots/grid_search_%s.csv' % preprocess_datasets.model_acronyms[args.mode]
    header = 'model_name,mode,check,use_evolutionary,motion_vectors,motion_architecture,repeat,'
    header += ','.join(dict_AUCPRs.keys())
    header += '\n'
    line = f'{model_name},{args.mode},{args.check},{args.use_evolutionary},{args.motion_vectors},{args.motion_architecture},{args.repeat},'
    line += ','.join(['%.3f'%x for x in dict_AUCPRs.values()])
    line += '\n'
    if not os.path.exists(results_file):
        with open(results_file,'w') as f:
            f.write(header)
    with open(results_file,'a') as f:
        f.write(line)

    if save_predictions:
        if not os.path.isdir(paths.library_folder + 'all_predictions/'):
            os.mkdir(paths.library_folder + 'all_predictions/')

        env = {
            'model_name': model_name,
            'mode': args.mode,
            'repeat':args.repeat,
            'motion_vectors':args.motion_vectors,
            'use_evolutionary': args.use_evolutionary,
            'motion_architecture': args.motion_architecture,
            'test_labels':test_labels,
            'test_predictions':test_predictions,
            'test_weights':test_weights,
            'dict_AUCPRs':dict_AUCPRs
        }
        io_utils.save_pickle(env, paths.library_folder + 'all_predictions/predictions_%s_%s.pkl'%(preprocess_datasets.model_acronyms[args.mode],model_name) )
























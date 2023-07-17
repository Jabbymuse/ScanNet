import os,sys
sys.path.append(os.getcwd())
from preprocessing import PDBio
from preprocessing import pipelines_modifie as pipelines
import numpy as np
import network.scannet_modif as scannet
from utilities import wrappers

example_id = '1a3x_A'
motion_vectors = 10
use_evolutionary = False

file_location,chain_ids = PDBio.getPDB(example_id)
_, chain = PDBio.load_chains(file=file_location,chain_ids=chain_ids)

pipeline = pipelines.ScanNetPipeline(
    with_aa=True,
    with_atom=True,
    aa_features= 'pwm' if use_evolutionary else 'sequence',
    motion_vectors= motion_vectors
    )

inputs,_ = pipeline.process_example(chain_obj=chain)

Lmax_aa = len(inputs[0])
# model = wrappers.load_model('models/ScanNet_PPI_noMSA',Lmax=Laa)

model, extra_params = scannet.initialize_ScanNet(
    None,
    None,
    motion_vectors = motion_vectors,
    with_atom=True,  # Whether to use atomic coordinates or not.
    Lmax_aa=Lmax_aa,  # Maximum protein length used for training
    K_aa=16,  # Size of neighborhood for amino acid Neighborhood Embedding Module (NEM)
    K_atom=16,  # Size of neighborhood for atom Neighborhood Embedding Module (NEM)
    K_graph=32,  # Size of neighborhood for Neighborhood Attention Module (NAM)
    Dmax_aa=11.,  # Cut-off distance for the amino acid NEM. Only used when initializing the aa gaussian kernels.
    Dmax_atom=4.,  # Cut-off distance for the atom NEM. Only used when initializing the gaussian kernels.
    Dmax_graph=13.,  # Cut-off distance for the amino acid NAM. Only used when initializing the gaussian kernels.
    N_aa=32,  # Number of gaussian kernels for amino acid NEM
    N_atom=32,  # Number of gaussian kernels for atom NEM
    N_graph=32,  # Number of gaussian kernels for amino acid NAM
    nfeatures_aa=21 if use_evolutionary else 20,  # Number of amino acid-wise input attributes.
    nfeatures_atom=12,  # Number of atom-wise input attributes (categorical variable).
    nembedding_atom=12,
    # Dimension of atom attribute embedding. If = nfeatures_atom, use non-trainable one-hot-encoding. # possible changes
    nembedding_aa=32,  # Dimension of amino acid attribute embedding.
    nembedding_graph=1,  # Number of values per edge for the NAM.
    dense_pooling=64,  # Number of channels for atom -> amino acid pooling operation.
    nattentionheads_pooling=64,  # Number of attention heads for atom -> amino acid pooling operation.
    nfilters_atom=128,  # Number of atomic spatio-chemical filters
    nfilters_aa=128,  # Number of amino acid spatio-chemical filters
    nfilters_graph=2,  # Number of outputs for NAM.
    nattentionheads_graph=1,  # Number of attention heads for NAM.
    filter_MLP=[32],  # Dimensionality reduction (trainable dense layer) applied after amino acid NEM and before NAM.
    covariance_type_atom='full',  # Full or diagonal covariance matrix for atom NEM module
    covariance_type_aa='full',  # Full or diagonal covariance matrix for amino acid NEM module
    covariance_type_graph='full',  # Full or diagonal covariance matrix for graph NEM module
    activation='relu',  # Activation function
    coordinates_atom=['euclidian'],  # Local coordinate system used for the atom NEM
    coordinates_aa=['euclidian'],  # Local coordinate system used for the amino acid NEM
    frame_aa='triplet_sidechain',  # Choice of amino acid frames (backbone-only also supported).
    coordinates_graph=['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'],
    # Local coordinate system used for the amino acid NAM
    index_distance_max_graph=8,  # Maximum sequence distance used.
    l12_atom=2e-3,  # Sparse regularization for atom NEM.
    l12_aa=2e-3,  # Sparse regularization for amino acid NEM.
    l12_pool=2e-3,  # Sparse regularization for atom to amino acid pooling.
    optimizer='adam',  # Optimizer.
    batch_size=1,  # Batch size.
    epochs=100,  # Maximum number of epochs
    n_init=2,
    # Parameter for initializing the Gaussian kernels. Number of initializations for fitting the GMM model with sklearn. 10 were used for the paper.
)

batch_inputs = [input[np.newaxis] for input in inputs] # Add batch dimension
outputs = model.predict(batch_inputs)

## Manually run on keras model, without wrapper.
keras_model = model.model # Keras model (no wrapper).
Lmaxs = model.Lmax # Length of each of the 8 input.

if motion_vectors:
    padding_values = [-1,0.,-1,0.,0.,-1, 0, -1, 0.] # Padding value for each of the 8 input.
    ninputs = 9
else:
    padding_values = [-1,0.,-1,0.,-1, 0, -1, 0.] # Padding value for each of the 8 input.
    ninputs = 8

padded_batch_inputs = [pipelines.padd_matrix(inputs[k], Lmax = Lmaxs[k],padding_value = padding_values[k])[np.newaxis] for k in range(ninputs)]
padded_output = keras_model.predict(padded_batch_inputs)



import os,sys
sys.path.append(os.getcwd())
from preprocessing import pipelines, PDBio
import numpy as np
from utilities import wrappers

example_id = '1a3x_A'
file_location,chain_ids = PDBio.getPDB(example_id)
_, chain = PDBio.load_chains(file=file_location,chain_ids=chain_ids)

pipeline = pipelines.ScanNetPipeline(
    with_aa=True,
    with_atom=True,
    aa_features='sequence',
    atom_features='type',
    aa_frames='triplet_sidechain',
    )

inputs,_ = pipeline.process_example(chain_obj=chain)
Laa = len(inputs[0])
model = wrappers.load_model('models/ScanNet_PPI_noMSA',Lmax=Laa)
batch_inputs = [input[np.newaxis] for input in inputs] # Add batch dimension
outputs = model.predict(batch_inputs)

## Manually run on keras model, without wrapper.

keras_model = model.model # Keras model (no wrapper).
Lmaxs = model.Lmax # Length of each of the 8 input.
padding_values = [-1,0.,-1,0.,-1, 0, -1, 0.] # Padding value for each of the 8 input.
padded_batch_inputs = [pipelines.padd_matrix(inputs[k], Lmax = Lmaxs[k],padding_value = padding_values[k])[np.newaxis] for k in range(8)]
padded_output = keras_model.predict(padded_batch_inputs)



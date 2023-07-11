import PDBio
from PDB_processing import NOLB_to_motion_vectors,create_reduce_pdb

# tests for pdb file reduce
input_pdb_file = "../PDB/pdb1bcc.bioent"
selected_chain_ids = [(0,'B'),(0,'C'),(1,'B'),(1,'F')]
chain_objs = PDBio.load_chains(chain_ids= selected_chain_ids, file=input_pdb_file)[1]
print(chain_objs)
create_reduce_pdb(chain_objs)

# tests NOLB
aa_indices_lengths = [406,379,406,100]
pdb_name = "1bcc"
m = 10
list_EV = NOLB_to_motion_vectors(aa_indices_lengths,pdb_name,m)
print(list_EV[0][-1])
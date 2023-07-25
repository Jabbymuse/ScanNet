import sys,os
sys.path.append(os.getcwd())
from preprocessing import PDBio,PDB_processing

# tests for pdb file reduce
input_pdb_file = "PDB/pdb1bcc.bioent"
selected_chain_ids = [(0,'B'),(0,'C'),(1,'B'),(1,'F')]
chain_objs = PDBio.load_chains(chain_ids= selected_chain_ids, file=input_pdb_file)[1]
list_motion_vectors = PDB_processing.apply_NOLB(chain_objs,m=10)
print(list_motion_vectors)
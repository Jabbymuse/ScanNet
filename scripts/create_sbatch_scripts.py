import numpy as np
python_binary = '/home/iscb/wolfson/jeromet/anaconda/envs/py_scannet2/bin/python'
all_modes = ['interface']#,'epitope','idp']
all_use_evolutionary = [False]#,True]
all_motion_vectors = [0,2,4,6,8,10]#list(range(11))
all_motion_architecture = [0,1,2]
all_repeats = [0]

all_commands = []
for motion_vectors in all_motion_vectors:
    for motion_architecture in all_motion_architecture:
        for use_evolutionary in all_use_evolutionary:
            for mode in all_modes:
                for repeat in all_repeats:
                    command = python_binary + f' scripts/grid_search.py --mode {mode} --motion_vectors {motion_vectors} --motion_architecture {motion_architecture} --repeat {repeat}'
                    if use_evolutionary:
                        command += ' --use_evolutionary'
                    # print(command)
                    all_commands.append(command)

nconcurrent_jobs = 6
ntrainings_per_job = int(np.ceil(len(all_commands) / nconcurrent_jobs))

header = '''#!/bin/sh
#SBATCH --job-name=gpu
#SBATCH --partition=killable # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --gpus=1 # GPU cores per process
#SBATCH --cpus-per-task=8 # CPU cores per process
#SBATCH --constraint="tesla_v100|quadro_rtx_8000|geforce_rtx_3090|a100|a5000|a6000"'''

i = 0
for job in range(nconcurrent_jobs):
    sbatch_script = header + '\n'
    for training in range(ntrainings_per_job):
        sbatch_script += all_commands[i] + '\n'
        i+=1
    with open('submit_job_%s.sh'%job,'w') as f:
        f.write(sbatch_script)





 #!/bin/sh -l
ssh mishrash@submit-b.hpc.engr.oregonstate.edu << EOF
    module load slurm
    srun -A eecs -p dgx2 --gres=gpu:01 --pty bash
    source activate
    cd /scratch/shri/Projects/Hand-CNN
    python detect.py
EOF
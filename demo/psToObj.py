import os
from tqdm import tqdm
import torch

from demo.rt_options import RTOptions
from bodymocap.models import SMPL
import mocap_utils.geometry_utils as gu
from mocap_utils.obj_extractor import writeObjFile

# psToObj is separate module that runs independantly from the main module - demo_rt.
# This module converts txt file - that contains 24 3X3 rotation matrix of SMPL joints + 10 shape parameters - into
# .obj file that can be used in other applications, such as Blender
jointCount = 24
matSize = 3
shapeCount = 10
# Variables mentioned above are set to meet the SMPL parameters that demo_rt uses. JointCount refers to pose,
# matSize refers to size of rotmat, (3X3) and shapeCount refers to 10 shape params for SMPL.

# Setting up directory to get txt files that contain pose and shape.\
# Currently, txt files that are saved by toggling the obj button gets saved inside PSOutput folder.
homeDir = os.path.abspath(os.curdir)
txtDir = os.path.join(homeDir, 'PSOutput')
# Since the name of files always have same patterns and the program does not have any function to erase the whole
# folder, it is HIGHLY RECOMMENDED TO EMPTY PSOutput and Output folder before demo_rt,
# and it is also HIGHLY RECOMMENDED TO BACK UP THE FILES that are inside output or PSOutput if you want to keep
# the .obj files that were created during demo_rt, after demo_rt is finished.

# getting RTOptions for smpl Model path.
args = RTOptions().parse()

# getting smpl and device ready for use.
smplModelPath = args.smpl_dir + '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
smpl = SMPL(smplModelPath, batch_size=1, create_transl=False).to(device)

# pred_rotmat and betas are basically temporary variables to get informations from txt files.
pred_rotmat = torch.empty(size=[1, jointCount, matSize, matSize], dtype=torch.float32, device=device)
betas = torch.empty(size=[1, shapeCount], dtype=torch.float32, device=device)

# For all files inside directory /PSOutput   =>>>>> tqdm is there for progress bar. Refreshes bar every 1sec.
for txtfile in tqdm(os.listdir(txtDir), mininterval=1.0):
    # Open each files as f
    with open(os.path.join(txtDir, txtfile)) as f:
        # Read lines
        lines = f.readlines()
        # Pose = (1, 24, 3, 3)
        for joint in range(jointCount):
            rotValue = lines[joint].split()
            for i in range(matSize):
                for j in range(matSize):
                    pred_rotmat[0, joint, i, j] = float(rotValue[(i*matSize + j)])
        # Shape = (1, 10)
        for k in range(shapeCount):
            betas[0, k] = float(lines[(jointCount + k)])

        # Calculate rotation matrices for every joints and change them to fit the input of smpl.
        pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
        pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
        # Use smpl model to get vertices
        smpl_output = smpl(
            betas=betas,
            body_pose=pred_aa[:, 3:],
            global_orient=pred_aa[:, :3],
            pose2rot=True)
        pred_vertices = smpl_output.vertices[0].cpu().numpy()
        # Since txt files name is in format of "res???.txt", we are going to get substring for objnum
        objnum = int(txtfile[3:-4])
        # Save obj files in the name of "res???.obj"
        writeObjFile(homeDir, pred_vertices, smpl.faces, objnum)
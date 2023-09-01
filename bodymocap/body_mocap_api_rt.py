import os
import torch
import numpy as np
import threading

from bodymocap.models import hmr, SMPL, SMPLX
from bodymocap import constants
from bodymocap.utils.imutils import process_image_bbox
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
import mocap_utils.geometry_utils as gu
from mocap_utils.obj_extractor import writeObjFile, writePoseNShapeFile

from demo.inputManager import *

# BodyMocap is class responsible for pose estimation via frank mocap.
class BodyMocap(object):
    # __init__ takes checkpoint, smpl_directory, device and use_smplx
    # use_smplx is to specify if you are going to use smpl or smplX, which contains hand and face.
    def __init__(self, regressor_checkpoint, smpl_dir, device=torch.device('cuda'), use_smplx=False):

        # device is given to make every torch & tensors to be on same cuda.
        #self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        # objnum : keep track of current output file(both obj and poseshape txt file) name.
        self.objnum = 0
        # objSaveDir : get absolute path for project folder for saving output files.
        self.objSaveDir = os.path.abspath(os.curdir)

        # Load parametric model (SMPLX or SMPL)
        if use_smplx:
            smplModelPath = smpl_dir + '/SMPLX_NEUTRAL.pkl'
            self.smpl = SMPLX(smpl_dir,
                    batch_size=1,
                    num_betas = 10,
                    use_pca = False,
                    create_transl=False).to(self.device)
            self.use_smplx = True
        else:
            smplModelPath = smpl_dir + '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
            self.smpl = SMPL(smplModelPath, batch_size=1, create_transl=False).to(self.device)
            self.use_smplx = False
            
        # Load pre-trained neural network
        SMPL_MEAN_PARAMS = './extra_data/body_module/data_from_spin/smpl_mean_params.npz'
        self.model_regressor = hmr(SMPL_MEAN_PARAMS).to(self.device)
        checkpoint = torch.load(regressor_checkpoint)
        self.model_regressor.load_state_dict(checkpoint['model'], strict=False)
        self.model_regressor.eval()

    def regress(self, img_original, body_bbox_list, betas):
        """
            args: 
                img_original: original raw image (BGR order by using cv2.imread)
                body_bbox: bounding box around the target: (minX, minY, width, height)
            outputs:
                pred_vertices_img:
                pred_joints_vis_img:
                pred_rotmat
                pred_betas
                pred_camera
                bbox: [bbr[0], bbr[1],bbr[0]+bbr[2], bbr[1]+bbr[3]])
                bboxTopLeft:  bbox top left (redundant)
                boxScale_o2n: bbox scaling factor (redundant) 
        """
        # pred_output_list will be used to store everything that needs to be returned.
        pred_output_list = list()

        # For every bbox detected, we are going to perform regression of model
        for body_bbox in body_bbox_list:
            img, norm_img, boxScale_o2n, bboxTopLeft, bbox = process_image_bbox(
                img_original, body_bbox, input_res=constants.IMG_RES)
            bboxTopLeft = np.array(bboxTopLeft)
            # ~~processing bbox

            # bboxTopLeft = bbox['bboxXYWH'][:2]
            if img is None:
                pred_output_list.append(None)
                continue

            with torch.no_grad():
                # model forward
                # Since we are going to use global beta, that is calculated (estimated) by different model,
                # We don't need to waste memory to store pred_betas, that is not going to be used in the project
                pred_rotmat, _, pred_camera = self.model_regressor(norm_img.to(self.device))
                #pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))

                # Convert rot_mat to aa since hands are always in aa
                # pred_aa = rotmat3x3_to_angle_axis(pred_rotmat)
                # FYI, aa stands for angle axis.
                pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
                pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
                # Use smpl model to calculate various parameters, such as joints, shape, vertices ...
                smpl_output = self.smpl(
                    betas=betas,
                    body_pose=pred_aa[:,3:],
                    global_orient=pred_aa[:,:3], 
                    pose2rot=True)
                pred_vertices = smpl_output.vertices
                pred_joints_3d = smpl_output.joints

                # get cam and vertices ready for processing output
                pred_vertices = pred_vertices[0].cpu().numpy()

                pred_camera = pred_camera.cpu().numpy().ravel()
                camScale = pred_camera[0]
                camTrans = pred_camera[1:]

                pred_output = dict()
                # Convert mesh to original image space (X,Y are aligned to image)
                # 1. SMPL -> 2D bbox
                # 2. 2D bbox -> original 2D image
                pred_vertices_bbox = convert_smpl_to_bbox(pred_vertices, camScale, camTrans)
                pred_vertices_img = convert_bbox_to_oriIm(
                    pred_vertices_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])

                # Convert joint to original image space (X,Y are aligned to image)
                pred_joints_3d = pred_joints_3d[0].cpu().numpy() # (1,49,3)
                pred_joints_vis = pred_joints_3d[:,:3]  # (49,3)
                pred_joints_vis_bbox = convert_smpl_to_bbox(pred_joints_vis, camScale, camTrans) 
                pred_joints_vis_img = convert_bbox_to_oriIm(
                    pred_joints_vis_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0]) 

                # Output
                pred_output['img_cropped'] = img[:, :, ::-1]
                pred_output['pred_vertices_smpl'] = smpl_output.vertices[0].cpu().numpy() # SMPL vertex in original smpl space
                pred_output['pred_vertices_img'] = pred_vertices_img # SMPL vertex in image space
                pred_output['pred_joints_img'] = pred_joints_vis_img # SMPL joints in image space

                pred_aa_tensor = gu.rotation_matrix_to_angle_axis(pred_rotmat.detach().cpu()[0])
                pred_output['pred_body_pose'] = pred_aa_tensor.cpu().numpy().reshape(1, 72) # (1, 72)

                pred_output['pred_rotmat'] = pred_rotmat.detach().cpu().numpy() # (1, 24, 3, 3)
                pred_output['pred_betas'] = betas.detach().cpu().numpy() # (1, 10)

                pred_output['pred_camera'] = pred_camera
                pred_output['bbox_top_left'] = bboxTopLeft
                pred_output['bbox_scale_ratio'] = boxScale_o2n
                pred_output['faces'] = self.smpl.faces

                # Write out obj file
                if inputKey["obj"]:
                    self.objnum += 1
                    # if you want to save obj directly while doing realtime estimation, you can use the commented line.
                    """
                    threading.Thread(target=writeObjFile,
                                     args=(self.objSaveDir, pred_output['pred_vertices_smpl'], self.smpl.faces, self.objnum)).start()
                    """
                    # if you want to know more about extracting obj file, see mocap_utils.obj_extractor.py
                    threading.Thread(target=writePoseNShapeFile,
                                     args=(self.objSaveDir, pred_output['pred_rotmat'], pred_output['pred_betas'], self.objnum)).start()

                # extra processing if you are using smplx, but we are using smpl now.
                if self.use_smplx:
                    #img_center = np.array((img_original.shape[1], img_original.shape[0]) ) * 0.5
                    # right hand
                    pred_joints = smpl_output.right_hand_joints[0].cpu().numpy()     
                    pred_joints_bbox = convert_smpl_to_bbox(pred_joints, camScale, camTrans)
                    pred_joints_img = convert_bbox_to_oriIm(
                        pred_joints_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])
                    pred_output['right_hand_joints_img_coord'] = pred_joints_img
                    # left hand 
                    pred_joints = smpl_output.left_hand_joints[0].cpu().numpy()
                    pred_joints_bbox = convert_smpl_to_bbox(pred_joints, camScale, camTrans)
                    pred_joints_img = convert_bbox_to_oriIm(
                        pred_joints_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])
                    pred_output['left_hand_joints_img_coord'] = pred_joints_img
                
                pred_output_list.append(pred_output)

        return pred_output_list

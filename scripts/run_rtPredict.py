import torch
import torchvision
import cv2

from models.humaniflow_model import HumaniflowModel
from models.smpl import SMPL
from models.pose2D_hrnet import PoseHighResolutionNet
from models.canny_edge_detector import CannyEdgeDetector
from configs.humaniflow_config import get_humaniflow_cfg_defaults
from configs.pose2D_hrnet_config import get_pose2D_hrnet_cfg_defaults
from configs import paths
from predict.predict_rtHumaniflow import predict_humaniflow
#-------------------------------------------
import time

"""--------------------------------------------------------------------"""

# Humaniflow class acts kinda like delegator.
# It initializes models that humaniflow model needs and invokes the estimation model.
# This is created, so that main code that calls calc_shape doesn't need to have all models and checkpoints that are
# used in humaniflow.
class Humaniflow:
    humaniflow_model = None
    humaniflow_cfg = None
    hrnet_model = None
    pose2D_hrnet_cfg = None
    edge_detect_model = None
    object_detect_model = None
    smpl_model = None
    joints2Dvisib_threshold = None
    num_pred_samples = None
    webcam = None
    def __init__(self, device):
        # ------------------------- Model Loading -------------------------
        # Configs
        self.pose2D_hrnet_cfg = get_pose2D_hrnet_cfg_defaults()
        self.humaniflow_cfg = get_humaniflow_cfg_defaults()

        # Bounding box / Object detection model
        self.object_detect_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)

        if self.object_detect_model is not None:
            self.object_detect_model.eval()

        # HRNet model for 2D joint detection
        self.hrnet_model = PoseHighResolutionNet(self.pose2D_hrnet_cfg).to(device)
        hrnet_checkpoint = torch.load('./model_files/pose_hrnet_w48_384x288.pth', map_location=device)
        self.hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)
        #print('\nLoaded HRNet weights from', './model_files/pose_hrnet_w48_384x288.pth')

        self.hrnet_model.eval()

        # Edge detector with canny edge detector
        self.edge_detect_model = CannyEdgeDetector(non_max_suppression=self.humaniflow_cfg.DATA.EDGE_NMS,
                                              gaussian_filter_std=self.humaniflow_cfg.DATA.EDGE_GAUSSIAN_STD,
                                              gaussian_filter_size=self.humaniflow_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                              threshold=self.humaniflow_cfg.DATA.EDGE_THRESHOLD).to(device)

        # SMPL model
        # print('\nUsing {} SMPL model with {} shape parameters.'.format('neutral', str(self.humaniflow_cfg.MODEL.NUM_SMPL_BETAS)))
        self.smpl_model = SMPL(paths.SMPL,
                          batch_size=1,
                          gender='neutral',
                          num_betas=self.humaniflow_cfg.MODEL.NUM_SMPL_BETAS).to(device)

        # HuManiFlow - 3D shape and pose distribution predictor
        self.humaniflow_model = HumaniflowModel(device=device,
                                           model_cfg=self.humaniflow_cfg.MODEL,
                                           smpl_parents=self.smpl_model.parents.tolist()).to(device)
        checkpoint = torch.load('./model_files/humaniflow_weights.tar', map_location=device)
        self.humaniflow_model.load_state_dict(checkpoint['best_model_state_dict'], strict=True)
        self.humaniflow_model.pose_so3flow_transform_modules.eval()
        #print('\nLoaded HuManiFlow weights from', './model_files/humaniflow_weights.tar')

        self.humaniflow_model.eval()

        #default 값
        self.joints2Dvisib_threshold = 0.75
        self.num_pred_samples = 50

    def calcShape(self, frame, result_queue, timer_queue, device):
        '''
        get estimated shape data and enqueue it into result queue
        params:
            frame:          bgr converted image, type of cv2 image
            result_queue:   queue that passed from outer scope
            device:         device used for shape estimation
        '''
        start = time.time()
        shape = predict_humaniflow(humaniflow_model=self.humaniflow_model,
                                   humaniflow_cfg=self.humaniflow_cfg,
                                   smpl_model=self.smpl_model,
                                   hrnet_model=self.hrnet_model,
                                   hrnet_cfg=self.pose2D_hrnet_cfg,
                                   edge_detect_model=self.edge_detect_model,
                                   device=device,
                                   camImage=frame,
                                   num_pred_samples=self.num_pred_samples,
                                   joints2Dvisib_threshold=self.joints2Dvisib_threshold
                                   )
        end = time.time()
        # print(f"{end - start:.5f} sec >>>>>>>>>>>>> huflow")

        timer_queue.put(end - start)
        result_queue.put(shape)

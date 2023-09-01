import torch
import cv2

from predict.predict_hrnet import predict_hrnet

from utils.image_utils import batch_crop_pytorch_affine
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch

# predict_humaniflow is responsible for whole process of getting shape parameter using Humaniflow.
def predict_humaniflow(humaniflow_model,
                       humaniflow_cfg,
                       smpl_model,
                       hrnet_model,
                       hrnet_cfg,
                       edge_detect_model,
                       device,
                       camImage,
                       object_detect_model=None,
                       num_pred_samples=50,
                       joints2Dvisib_threshold=0.75):
    with torch.no_grad():
        # ------------------------- INPUT LOADING AND PROXY REPRESENTATION GENERATION -------------------------
        # Converting images - in this project, frame that we got from webcam - for processing into model.
        alphaImage = cv2.cvtColor(camImage, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(alphaImage.transpose(2, 0, 1)).float().to(device) / 255.0
        # Predict Person Bounding Box + 2D Joints
        hrnet_output = predict_hrnet(hrnet_model=hrnet_model,
                                     hrnet_config=hrnet_cfg,
                                     object_detect_model=object_detect_model,
                                     image=image,
                                     object_detect_threshold=humaniflow_cfg.DATA.BBOX_THRESHOLD,
                                     bbox_scale_factor=humaniflow_cfg.DATA.BBOX_SCALE_FACTOR)

        # Transform predicted 2D joints and image from HRNet input size to input proxy representation size
        hrnet_input_centre = torch.tensor([[hrnet_output['cropped_image'].shape[1],
                                            hrnet_output['cropped_image'].shape[2]]],
                                          dtype=torch.float32,
                                          device=device) * 0.5
        hrnet_input_height = torch.tensor([hrnet_output['cropped_image'].shape[1]],
                                          dtype=torch.float32,
                                          device=device)
        cropped_for_proxy = batch_crop_pytorch_affine(
            input_wh=(hrnet_cfg.MODEL.IMAGE_SIZE[0], hrnet_cfg.MODEL.IMAGE_SIZE[1]),
            output_wh=(humaniflow_cfg.DATA.PROXY_REP_SIZE, humaniflow_cfg.DATA.PROXY_REP_SIZE),
            num_to_crop=1,
            device=device,
            joints2D=hrnet_output['joints2D'][None, :, :],
            rgb=hrnet_output['cropped_image'][None, :, :, :],
            bbox_centres=hrnet_input_centre,
            bbox_heights=hrnet_input_height,
            bbox_widths=hrnet_input_height,
            orig_scale_factor=1.0)

        # Create proxy representation with 1) Edge detection and 2) 2D joints heatmaps generation
        edge_detector_output = edge_detect_model(cropped_for_proxy['rgb'])
        proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if humaniflow_cfg.DATA.EDGE_NMS else \
        edge_detector_output['thresholded_grad_magnitude']
        proxy_rep_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(joints2D=cropped_for_proxy['joints2D'],
                                                                         img_wh=humaniflow_cfg.DATA.PROXY_REP_SIZE,
                                                                         std=humaniflow_cfg.DATA.HEATMAP_GAUSSIAN_STD)

        hrnet_joints2Dvisib = hrnet_output['joints2Dconfs'] > joints2Dvisib_threshold
        hrnet_joints2Dvisib[
            [0, 1, 2, 3, 4, 5, 6]] = True  # Only removing joints [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] if occluded
        proxy_rep_heatmaps = proxy_rep_heatmaps * hrnet_joints2Dvisib[None, :, None, None]
        proxy_rep_input = torch.cat([proxy_rep_img, proxy_rep_heatmaps], dim=1).float()  # (1, 18, img_wh, img_wh)

        # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
        # See models.humaniflow_model.py for details.
        pred = humaniflow_model(proxy_rep_input,
                                num_samples=num_pred_samples,
                                use_shape_mode_for_samples=False,
                                return_input_feats=True)
        return pred['shape_mode']

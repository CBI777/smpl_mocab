# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import cv2
import threading
import queue

# ===============================================================
from demo.rt_options import RTOptions
from bodymocap.body_mocap_api_rt import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
from mocap_utils.timer import Timer
from renderer.viewer2D_rt import ImShow
# ================================================================
from scripts.run_rtPredict import Humaniflow
from utils.gpu_utilize import *
from demo.inputManager import *

def main():
    # See rt_options for detailed arguments.
    # rt_options.py is modified to support the user to run the program with:
    # 1. No output file 2. No argument 3. Webcam input and Output with mesh rendered on top
    # It is also possbile to give arguments that user want when running the program, but for now,
    # it is okay to run the program with only "python -m demo.demo_rt"
    args = RTOptions().parse()
    # Cuda Initialization
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize the class for humaniflow model, which is used for independant shape estimation.
    # See scripts.run_rtPredict.py for details
    humaniflow_Class = Humaniflow(device)

    # Set bbox detector : Bounding Box Detection Model set up for pose estimation
    # The bbox estimator that they are using is "Lightweighted Openpose".
    # They are only using return value of bbox, but the original model returns 2d human pose + bbox at the same time.
    body_bbox_detector = BodyPoseEstimator()

    # Set mocap regressor : Frank Mocap's main model for pose estimation
    # different initialization between SMPLX and SMPL. For this case, we are using SMPL
    # See bodymocap.body_mocap_api_rt.py for details.
    use_smplx = args.use_smplx
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # Frank mocap does not consider much about shape, so it is almost pointless to use regressed shape param from
    # frankmocap. So, we are setting up separate shape parameters that will be shared among the program.
    # We are just initailzing them here.
    shape = torch.zeros([1, 10], dtype=torch.float32, device=device)

    # =======================================================================================#
    # ============================= Estimation Loop Begins here =============================#
    # =======================================================================================#

    # First, we set input type and data. In case of Frank Mocap, it can manipulate many different outputs.
    # (ex. Video, Image, Webcam...)
    # However, what we want right now is only webcam input. So, we are leading the program to use specific input only.
    # If needed, please see demo_bodymocap.py file to get reference for different types of inputs.
    input_data = cv2.VideoCapture(0)

    # Related to [Start Frame]. It is set to 0 by default
    cur_frame = args.start_frame
    video_frame = 0
    # 'timer' is basically a timer that is provided by frank mocap.
    # It shows fps and process speed. I don't see any reasons for not using this.
    timer = Timer()

    # to store estimated shape data from other threads
    result_queue = queue.Queue()
    thread_timer_queue = queue.Queue()

    while True:
        print("--------------------------------------")

        timer.tic()
        # read from webcam. The fisrt return value is res, which is not used in this program.
        _, img_original_bgr = input_data.read()

        # frame related stuff for timer
        if video_frame < cur_frame:
            video_frame += 1
            continue
        if img_original_bgr is not None:
            video_frame += 1

        cur_frame += 1

        # Termiation condition check. For Frank mocap, the user can set start and end frame.
        # We used default value, so the program runs from 0 ~ inf = infinite loop.
        # If you were to change arg values, you might want to modify this if statement
        if img_original_bgr is None or cur_frame > args.end_frame:
            break

        timer.time_stamp(bPrint=True, title='read image from webcam')

        # There's option to give bbox values that are, for example, stored in computer,
        # but for now, we are checking bbox at realtime from the input.
        _, body_bbox_list = body_bbox_detector.detect_body_pose(img_original_bgr)

        if len(body_bbox_list) < 1:
            print(f"No body detected!")
            continue

        # Sort the bbox using bbox size
        # (to make the order as consistent as possible without tracking)
        # There's options for relationship between bbox size/count and rendering/ignoring.
        # In this part, we are manipulating bbox for 'ignoring bboxes that are not the biggest.
        bbox_size = [(x[2] * x[3]) for x in body_bbox_list]
        idx_big2small = np.argsort(bbox_size)[::-1]
        body_bbox_list = [body_bbox_list[i] for i in idx_big2small]
        if args.single_person and len(body_bbox_list) > 0:
            body_bbox_list = [body_bbox_list[0], ]

        timer.time_stamp(bPrint=True, title='bounding box detection')

        # Body Pose Regression
        # See body_mocap.body_mocap_api_rt for details
        pred_output_list = body_mocap.regress(img_original_bgr, body_bbox_list, shape)
        assert len(body_bbox_list) == len(pred_output_list)

        timer.time_stamp(bPrint=True, title='pose estimation')

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        # See mocap_utils.demo_utils for details
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        timer.time_stamp(bPrint=True, title='get vertices from estimated pose and shape')

        # visualization
        # see renderer.screen_free_visualizer for details
        res_img = visualizer.visualize(
            img_original_bgr,
            pred_mesh_list=pred_mesh_list,
            body_bbox_list=body_bbox_list)

        # show result in the screen
        # if args.no_display which is stated inside rt_options is set as True,
        # the program won't display second screen that shows bbox and rendered image at the same time.
        # If args.no_display is False, 'Image' window will take the key.
        # else, glRenderer Screen will take the key.

        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            # Imshow is function that shows result
            # viewer2D_rt.py is created and used to make the program perform functions that we want.
            # For more reference, please see renderer/viewer2D_rt.py
            res = ImShow(res_img)
            if res == 'Exit':
                # CBI :
                # if ESC is pressed, we break and close the window + exit the program.
                break
            elif res == 'Shape':
                # if Spacebar is pressed, we recalculate the shape from the input.
                # shape = humaniflow_Class.calcShape(img_original_bgr)
                threading.Thread(target=humaniflow_Class.calcShape,
                                 args=(img_original_bgr, result_queue, thread_timer_queue, device)).start()
                # print(shape)

        # Functions take place according to glRenderer screen inputs
        # For more details, please see demo.inputmanager
        else:
            if inputKey["esc"]:
                #esc
                inputKey["esc"] = False
                break
            elif inputKey["shape"]:
                #space bar
                inputKey["shape"] = False
                threading.Thread(target=humaniflow_Class.calcShape,
                                 args=(img_original_bgr, result_queue, thread_timer_queue, device)).start()

        timer.toc(bPrint=True, title="Time")

        if not result_queue.empty():
            # get all estimated shapes(tensor) and calculate average of those
            que_len = result_queue.qsize()
            temp_shape = result_queue.get()
            temp_time = thread_timer_queue.get()
            for i in range(result_queue.qsize() - 1):
                temp_shape += result_queue.get()
                temp_time += thread_timer_queue.get()
            shape = temp_shape / que_len
            temp_time = temp_time / que_len
            print('shape estimation took {:0.5f} seconds (average of {} execution)'.format(temp_time, que_len))
            # the mean of shape data would be applied to smpl mesh at next frame

        # get allocated GPU memory information
        # below code could be removed after all things done
        # allocated_mem, cached_mem = get_gpu_memory_usage()
        # print(f"Allocated GPU Memory: {allocated_mem:.2f} GB")
        # print(f"Cached GPU Memory: {cached_mem:.2f} GB")

    # If somehow the loop is broken through (like pressing esc), we shut the window and release input data to finish
    # the program.
    if input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    exit(main())

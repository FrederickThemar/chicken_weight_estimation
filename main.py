import os
import cv2
import time
import torch

import open3d as o3d
import numpy as np

from tqdm import tqdm
from yolo import yolo
from pcd  import pcd
from kpconv import kpconv

# CLASS: creates other objects to manage the video-to-estimate pipeline
class Main():
    def __init__(self, path=None, depth=None, mode=None, save=False):
        if path == None:
            print("ERROR: filepath empty.")
            exit(1)
        if mode == None:
            print("ERROR: mode empty.")
            exit(1)
        if not isinstance(mode, int):
            print("ERROR: mode must be 0, 1, or 2.")
            exit(1)
        if mode > 2 or mode < 0:
            print("ERROR: mode must be 0, 1, or 2.")
            exit(1)
        
        self.save = save
        self.mode = mode
        self.init_path = path
        self.depth_path = depth

        # Check that filepaths exist, return error otherwise
        self.checkPath(self.init_path)

        # Initialize YOLO object
        self.yolo = yolo()
        self.pcd = pcd()
        self.kpconv = kpconv()

    # Begins the pipeline based on the mode provided to the object
    def begin(self):
        if self.mode == 0:
            self.process_frame()
        elif self.mode == 1:
            self.process_video()
        elif self.mode == 2:
            self.process_live()
        else:
            print("ERROR: Incorrect mode input.")
            print(f'Mode: {self.mode}')
            exit(1)

    # HELPER: Checks if input filepath exists.
    def checkPath(self, inputPath):
        if not os.path.exists(inputPath):
            print("ERROR: Path does not exist.")
            print(f'Path: {inputPath}')
            exit(1)

    # Estimates weight of an individual frame
    def process_frame(self):
        # Store color frame
        framepath = self.init_path
        color = cv2.imread(framepath)

        # Make sure depth file provided
        if self.depth_path == None:
            print("ERROR: For this mode, a depth frame must be provided.")
            exit(1)
        self.checkPath(self.depth_path)
        depthpath = self.depth_path
        # depth = cv2.imread(self.depth_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = o3d.io.read_image(self.depth_path)
        # rgb_im = o3d.io.read_image(self.color_path)

        # Mask frame.
        success, mask, _ = self.yolo.mask_frame(color, self.save)
        if not success:
            print("ERROR: No chicken detected in frame.")
            exit(1)
        
        pcd = self.pcd.pcd_frame(color, depth, mask, self.save)
        if pcd == None:
            print("\nERROR: The mask for this frame falls outside the acceptable boundaries for the model. Try a different frame.")
            exit(1)
        output = self.kpconv.estimate_frame(pcd)

        print(f'\nOUTPUT:\t\t\n{output[0][0]}')
        print(f'YOLO time:   {self.yolo.times}')
        print(f'PCD time:    {self.pcd.times}')
        print(f'KPConv time: {self.kpconv.times}')

    # Estimates weight of all the frames in a video
    def process_video(self):
        # Maybe use process_frame for this? 
        #   - Can have pframe return estimate and append to list
        #   - Save the frames separately, then have a list storing the paths to each. Give to pframe() 
        #       - Maybe each item in array can be another JSON, with paths to mask, depth, and color?
        #   - TQDM can run based on number of files in frame folder
        
        # Log start time of function
        start = time.time()

        videopath = self.init_path

        # # Keyboard commands used for pausing/stopping program
        # glfw_key_escape = 256
        # glfw_key_space = 32

        # Open the video
        reader = o3d.io.AzureKinectMKVReader()
        reader.open(videopath)

        count = 0
        outputs = []
        while not reader.is_eof():
            rgbd = reader.next_frame()
            if rgbd is None:
                continue

            # Store the color and depth frames
            # color = rgbd.color
            depth = rgbd.depth
            
            # This might not be needed!
            DEBUG_depthCV2 = np.asarray(depth)
            DEBUG_depthCV2 = cv2.cvtColor(DEBUG_depthCV2, cv2.COLOR_RGB2BGR)
            color = np.asarray(rgbd.color)
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # print(DEBUG_depthCV2.dtype)
            # cv2.imwrite('img.png',DEBUG_depthCV2)
            # o3d.io.write_image('img2.png', depth)
            # cv2.waitKey(0)

            success, mask, overlay = self.yolo.mask_frame(color, self.save)
            if not success:
                # Write frame count
                color = cv2.rectangle(color, (0, 0), (250, 50), (200,0,0), -1)
                color = cv2.putText(
                    color,
                    f'Frame: {count+1}',
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', color)
                cv2.waitKey(10)
                
                count+=1
                continue

            pcd = self.pcd.pcd_frame(color, depth, mask, self.save)
            if pcd == None:
                # Write frame count
                color = cv2.rectangle(color, (0, 0), (250, 50), (200,0,0), -1)
                color = cv2.putText(
                    color,
                    f'Frame: {count+1}',
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', color)
                cv2.waitKey(10)
                
                count+=1
                continue

            output = self.kpconv.estimate_frame(pcd)
            outputs.append(output)
            
            # Write output on visualization frame
            overlay = cv2.rectangle(overlay, (0, 0), (250, 50), (200,0,0), -1)
            overlay = cv2.putText(
                overlay,
                f'Frame: {count+1}',
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            overlay = cv2.rectangle(overlay, (0, 50), (250, 100), (0,200,0), -1)
            overlay = cv2.putText(
                overlay,
                'Output: %.2f' % round(output[0][0],2),
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            # Display the visualization frame
            cv2.imshow('frame', overlay)
            cv2.waitKey(10)

            count+=1
            # break
        print(f'COUNT: {count}')
        print(f'LEN:   {len(outputs)}')
        print(f'AVG:   {np.mean(outputs)}')

        # Close the video
        reader.close()

        # Close cv2 window
        cv2.destroyAllWindows()

        # Print process times
        end = time.time()
        duration = (end - start)
        print(f'PROCESS TIME: {duration} seconds, {duration/60} minutes')
        print(f'YOLO avg time:   {np.mean(self.yolo.times)}')
        # print(f'YOLO time len:   {len(self.yolo.times)}')
        print(f'PCD avg time:    {np.mean(self.pcd.times)}')
        # print(f'PCD time len:    {len(self.pcd.times)}')
        print(f'KPConv avg time: {np.mean(self.kpconv.times)}')
        # print(f'KPConv time len: {len(self.kpconv.times)}')
        # print(self.yolo.times)
        print(f'\nDataset avg time: {np.mean(self.kpconv.dataset_times)}')
        print(f'Sampler avg time: {np.mean(self.kpconv.sampler_times)}')
        print(f'Collate avg time: {np.mean(self.kpconv.collate_times)}')
        print(f'Loader avg time:  {np.mean(self.kpconv.loader_times)}')
        print(f'Loop avg time:    {np.mean(self.kpconv.loop_times)}')
        print(f'Output avg time:  {np.mean(self.kpconv.output_times)}')

    def process_live(self):
        log("ERROR: Not yet implemented. Process live video.")

# DEBUG: Prints debug messages. Remove later.
def log(inputStr):
    print(inputStr)


# Main function
if __name__ == '__main__':
    print("Begin.")

    # Grab user input to determine mode and filepath
    # Unacceptable mask:
    # TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240326/chicken3/color/000100.jpg'
    # TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240326/chicken3/depth/000100.png'
    # Acceptable mask
    # TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240409/chicken20/color/000098.jpg'
    # TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240409/chicken20/depth/000098.png'

    TEMP_video = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Spring2024/20240409/chicken_16.mkv'
    # TEMP_video = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Spring2024/20240306/chicken_16.mkv'

    # Initialize Main object to handle the pipeline
    # main = Main(path=TEMP_path, depth=TEMP_depth, mode=0)
    main = Main(path=TEMP_video, mode=1)

    # Begin the pipeline
    main.begin()
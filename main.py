import os
import torch

import open3d as o3d

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
        framepath = self.init_path

        # Make sure depth file provided
        if self.depth_path == None:
            print("ERROR: For this mode, a depth frame must be provided.")
            exit(1)
        self.checkPath(self.depth_path)
        depthpath = self.depth_path

        # Mask frame.
        success, mask = self.yolo.mask_frame(framepath, self.save)
        if not success:
            print("ERROR: No chicken detected in frame.")
            exit(1)
        
        pcdPath = self.pcd.pcd_frame(framepath, depthpath, mask, self.save)
        output = self.kpconv.estimate_frame(pcdPath)

        print(f'OUTPUT:\t\t\n{output[0][0]}')

    # Estimates weight of all the frames in a video
    def process_video(self):
        # Maybe use process_frame for this? 
        #   - Can have pframe return estimate and append to list
        #   - Save the frames separately, then have a list storing the paths to each. Give to pframe() 
        #       - Maybe each item in array can be another JSON, with paths to mask, depth, and color?
        #   - TQDM can run based on number of files in frame folder

        videopath = self.init_path

        # # Keyboard commands used for pausing/stopping program
        # glfw_key_escape = 256
        # glfw_key_space = 32

        # Open the video
        reader = o3d.io.AzureKinectMKVReader()
        reader.open(videopath)

        count = 0
        while not reader.is_eof():
            rgbd = reader.next_frame()
            if rgbd is None:
                continue

            # Store the color and depth frames
            color = rgbd.color
            depth = rgbd.depth

            count+=1
        print(f'COUNT: {count}')

        # Close the video
        reader.close()

        print(videopath)

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

    TEMP_video = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Spring2024/20240409/chicken_13.mkv'

    # Initialize Main object to handle the pipeline
    # main = Main(path=TEMP_path, depth=TEMP_depth, mode=0)
    main = Main(path=TEMP_video, mode=1)

    # Begin the pipeline
    main.begin()
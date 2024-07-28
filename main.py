import os
import torch

from tqdm import tqdm
from yolo import yolo
from pcd  import pcd

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
        if mode > 2:
            print("ERROR: mode must be 0, 1, or 2.")
            exit(1)
        
        self.save = save
        self.mode = mode
        self.init_path = path
        self.depth_path = depth

        # Check that filepaths exist, return error otherwise
        self.checkPath(self.init_path)

        # Info for processing methods
        self.depthOutput = "./outputDepth/"
        self.pcdOutput = "./outputPCD/"
        self.checkPath(self.depthOutput)
        self.checkPath(self.pcdOutput)

        # Initialize YOLO object
        self.yolo = yolo()
        self.pcd = pcd()

        # Call class method based on mode
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
        success = self.yolo.mask_frame(framepath, self.save)
        if not success:
            print("ERROR: No chicken detected in frame.")
            exit(1)
        
        self.pcd.pcd_frame(framepath, depthpath, self.yolo.maskPaths[0], self.save)

    # Estimates weight of all the frames in a video
    def process_video(self):
        # Maybe use process_frame for this? 
        #   - Can have pframe return estimate and append to list
        #   - Save the frames separately, then have a list storing the paths to each. Give to pframe() 
        #       - Maybe each item in array can be another JSON, with paths to mask, depth, and color?
        #   - TQDM can run based on number of files in frame folder
        log("ERROR: Not yet implemented. Process video")

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
    TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240409/chicken20/color/000098.jpg'
    TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240409/chicken20/depth/000098.png'

    # Initialize Main object to handle the pipeline
    main = Main(path=TEMP_path, depth=TEMP_depth, mode=0)

    # Begin the pipeline
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
from azure import ViewerWithCallback

# CLASS: creates other objects to manage the video-to-estimate pipeline
class Main():
    def __init__(self, path=None, depth=None, mode=None, save=False):
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

        # If not using live-video mode, need to check that path to video or rgb frame.
        if self.mode != 2:
            if self.init_path == None:
                print("ERROR: filepath empty.")
                exit(1)

            # Check that filepaths exist, return error otherwise
            self.checkPath(self.init_path)

        # Initialize YOLO object
        self.yolo = yolo()
        self.pcd = pcd()
        self.kpconv = kpconv()

        # Used when drawing overlay
        self.alpha = 0.3
        self.beta = 1 - self.alpha

        # Used to store a the estimates and average for each ID given by YOLO
        self.table = {}

        # Used when calculating moving average
        self.ema_alpha = 0.025

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
        # Store rgb frame
        framepath = self.init_path
        rgb = cv2.imread(framepath)

        # Make sure depth file provided
        if self.depth_path == None:
            print("ERROR: For this mode, a depth frame must be provided.")
            exit(1)
        self.checkPath(self.depth_path)
        depthpath = self.depth_path
        depth = o3d.io.read_image(self.depth_path)

        # Mask frame.
        success, masks, ids, boxes = self.yolo.mask_frame(rgb, self.save)
        if not success:
            print("ERROR: No chicken detected in frame.")
            exit(1)
        
        pcds, accep_masks, pcd_idxs = self.pcd.pcd_frame(rgb, depth, masks, self.save)
        if pcds == []:
            print("\nERROR: The mask for this frame falls outside the acceptable boundaries for the model. Try a different frame.")
            exit(1)
        outputs, accep_idxs = self.kpconv.estimate_frame(pcds, pcd_idxs)

        for i in range(len(accep_idxs)):
            # print(f'ID: {i}')
            output = outputs[0][i][0]
            print(f'ID {ids[i]}: {output} kg')
        print()
        print(f'YOLO time:   {self.yolo.times}')
        print(f'PCD time:    {self.pcd.times}')
        print(f'KPConv time: {self.kpconv.times}')

        # DEBUG: Draw bounding box onto overlay and display it
        # defaultBox = ((350, 245),(1400,1000)) # Bounding box coords
        # rgb = cv2.rectangle(rgb, defaultBox[0], defaultBox[1], (255,0,0), 2)

        # for i in range(len(ids)):
        #     box = boxes[i]
        #     top_left = (int(box[0]), int(box[1])-35)
        #     bot_righ = (int(box[0])+115, int(box[1]))

        #     rgb = cv2.rectangle(rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,255,255), 2)
        #     rgb = cv2.rectangle(rgb, top_left, bot_righ, (255,255,255), -1)
        #     rgb = cv2.putText(
        #         rgb,                                        # Base img
        #         f'{ids[i]}',                                # Text
        #         (int(box[0])+5, int(box[1])-10),            # Org, ie bottom left
        #         cv2.FONT_HERSHEY_SIMPLEX,                   # Font
        #         0.85,                                       # Font Scale
        #         (0,0,0),                                    # Color
        #         1,                                          # Thickness
        #         cv2.LINE_AA
        #     )

        # # DEBUG: Show frame
        # cv2.imshow('debug', rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Runs a given RGBD frame through the pipeline.
    def handle_rgbd(self, rgbd, count):
        # Store the rgb and depth frames
        depth = rgbd.depth
        rgb = np.asarray(rgbd.color)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        success, masks, ids, boxes = self.yolo.mask_frame(rgb, self.save)
        
        if len(masks) != len(ids) or len(masks) != len(boxes):
            print("ERROR: LENGTHS DON'T MATCH")
            print(len(masks))
            print(len(ids))
            print(len(boxes))
            exit(1)

        if not success:
            # Write frame count
            rgb = cv2.rectangle(rgb, (0, 0), (250, 50), (200,0,0), -1)
            rgb = cv2.putText(
                rgb,
                f'Frame: {count+1}',
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            return None, rgb
            
        # Generate pointclouds from masks
        pcds, accep_masks, accep_idxs = self.pcd.pcd_frame(rgb, depth, masks, self.save)
        
        if pcds == []:
            rgb = cv2.rectangle(rgb, (0, 0), (250, 50), (200,0,0), -1)
            rgb = cv2.putText(
                rgb,
                f'Frame: {count+1}',
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # No mask detected, write "N/A"
            rgb = cv2.rectangle(rgb, (0, 50), (250, 100), (0,200,0), -1)
            rgb = cv2.putText(
                rgb,
                'Output: N/A',
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            return None, rgb

        pcd_idxs = []
        outputs = self.kpconv.estimate_frame(pcds)
        
        # Draw acceptable masks and data onto image
        overlay = rgb.copy()

        # Make areas outside bounding box darker
        isolated = np.zeros(overlay.shape[:3], np.uint8)
        isolated = cv2.rectangle(isolated, self.pcd.defaultBox[0], self.pcd.defaultBox[1], (255,255,255), -1)
        iso_loc = np.invert(isolated.astype(bool))
        overlay[iso_loc] = cv2.addWeighted(overlay, self.alpha, isolated, self.beta, 0.0)[iso_loc]

        # Update the table
        for i in range(len(accep_idxs)):
            # Grab weight estimate
            new_weight = outputs[0][i][0]

            # Initialize ID in table if not already there
            if ids[accep_idxs[i]] not in self.table:
                self.table[ids[accep_idxs[i]]] = {}
                self.table[ids[accep_idxs[i]]]["list"] = []
                self.table[ids[accep_idxs[i]]]["list"].append(new_weight)
                self.table[ids[accep_idxs[i]]]["curr_size"] = 1
                self.table[ids[accep_idxs[i]]]["curr_avg"] = new_weight
                continue
            
            # Add weight to corresponding ID array, update moving average
            self.table[ids[accep_idxs[i]]]["list"].append(new_weight)                       # Extract data for ID from table
            curr_avg = self.table[ids[accep_idxs[i]]]["curr_avg"]                           # Grab current average
            self.table[ids[accep_idxs[i]]]["curr_size"]+=1                                  # DEBUG: Increment size. Don't actually need this, only to quickly see when saving table.
            new_avg = (self.ema_alpha * new_weight) + ((1 - self.ema_alpha) * curr_avg)     # (alpha * new) + ((1-alpha) * curr)
            self.table[ids[accep_idxs[i]]]["curr_avg"] = new_avg                            # Update avg for given ID

        for k in range(len(accep_idxs)):
            # Store index
            i = accep_idxs[k]

            # Draw mask
            mask_location = masks[i].astype(bool)
            overlay[mask_location] = cv2.addWeighted(rgb, self.alpha, masks[i], self.beta, 0.0)[mask_location]

            # Draw box and write output inside 
            # left=0, top, right, bottom
            # Get coordinates
            box = boxes[i]
            top_left = (int(box[0]), int(box[1])-35)
            bot_righ = (int(box[0])+115, int(box[1]))
            # Get color
            color_idx = ids[i] % len(self.yolo.colors)
            color = self.yolo.colors[color_idx-1]
            # Draw the boxes and text
            overlay = cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            overlay = cv2.rectangle(overlay, top_left, bot_righ, color, -1)
            overlay = cv2.putText(
                overlay,                                # Base img
                "{:0.2f} kg".format(self.table[ids[i]]["curr_avg"]),  # Text
                (int(box[0])+5, int(box[1])-10),        # Org, ie bottom left
                cv2.FONT_HERSHEY_SIMPLEX,               # Font
                0.85,                                   # Font Scale
                (0,0,0),                          # Color
                1,                                      # Thickness
                cv2.LINE_AA
            )

        # Write info and weights to frame
        overlay = cv2.rectangle(overlay, (0, 0), (250, 50), (200,0,0), -1)
        overlay = cv2.putText(
            overlay,
            f'Frame: {count+1}',
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return outputs, overlay


    # Estimates weight of all the frames in a video
    def process_video(self):
        # Log start time of function
        start = time.time()

        videopath = self.init_path

        # Open the video
        reader = o3d.io.AzureKinectMKVReader()
        reader.open(videopath)

        count = 0
        outputs = []
        self.exit_early = False
        while not reader.is_eof() and not self.exit_early:
            rgbd = reader.next_frame()
            if rgbd is None:
                continue

            output, overlay = self.handle_rgbd(rgbd, count)

            # Display the visualization frame
            cv2.imshow('frame', overlay)
            key = cv2.waitKey(1)

            # Exit early if Esc key hit
            if key == 27:
                self.exit_early = True

            if output is None: 
                count+=1
                continue

            outputs.append(output)
            count+=1
            
        # Close the video
        reader.close()

        # Close cv2 window
        cv2.destroyAllWindows()

        # Print process times
        end = time.time()
        duration = (end - start)
        print(f'PROCESS TIME: {duration} seconds, {duration/60} minutes')
        print(f'YOLO avg time:   {np.mean(self.yolo.times)}')
        print(f'PCD avg time:    {np.mean(self.pcd.times)}')
        print(f'KPConv avg time: {np.mean(self.kpconv.times)}')
        print(f'\nLoop start time:  {np.mean(self.kpconv.loop_times)}')

        # DEBUG: print table
        #print(self.table)

    # Callback function. Exit loop in process_live when user hits Esc
    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def process_live(self):
        # Config for live Azure
        # config = o3d.io.AzureKinectSensorConfig()
        config = o3d.io.read_azure_kinect_sensor_config("./config.json")

        # Create the object for reading from the camera
        self.sensor = o3d.io.AzureKinectSensor(config)
        device = 0
        if not self.sensor.connect(device):
            raise RuntimeError("Failed to connect to sensor.")

        self.flag_exit = False
        outputs = []
        count = 0
        start = time.time()
        while not self.flag_exit:

            rgbd = self.sensor.capture_frame(True)
            if rgbd is None:
                continue

            output, overlay = self.handle_rgbd(rgbd, count)

            # Display the visualization frame
            cv2.imshow('frame', overlay)
            key = cv2.waitKey(1)

            # If key is Esc, exit loop.
            if key == 27:
                self.flag_exit = True
                # Don't exit early, make sure most recent

            # Skip appending output if there is none
            if output is None:
                count+=1
                continue

            outputs.append(output)

            # Want loop to end after 60 seconds
            end = time.time()
            duration = end - start
            count+=1
        
        # Close cv2 window
        cv2.destroyAllWindows()


# Main function
if __name__ == '__main__':
    print("Begin.")

    # Grab user input to determine mode and filepath
    # Unacceptable mask:
    # TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240326/chicken3/color/000100.jpg'
    # TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240326/chicken3/depth/000100.png'
    # Acceptable mask

    # Used for mode=0, process frame
    # TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240409/chicken20/color/000098.jpg'
    # TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240409/chicken20/depth/000098.png'
    # These ones have two chickens
    TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Summer2024/20240619/color/000254.jpg'
    TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Summer2024/20240619/depth/000254.png'
    # Used for testing when certain masks get removed. 303.jpg processes both, but not 304.
    # TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Summer2024/20240619/color/000304.jpg'
    # TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Summer2024/20240619/depth/000304.png'
    TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Summer2024/20240619/color/000529.jpg'
    TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Summer2024/20240619/depth/000529.png'

    # Used for mode=1, process video
    # TEMP_video = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Spring2024/20240409/chicken_16.mkv'
    # TEMP_video = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Spring2024/20240306/chicken_16.mkv'

    # Used with mode=1, has multiple chickens
    # TEMP_video = "/home/jzbumgar/datasets/Summer_videos/20240617_chicken03.mkv"
    TEMP_video = "/home/jzbumgar/datasets/Summer_videos/20240619_chicken03.mkv"
    # TEMP_video = "/home/jzbumgar/Downloads/chicken_07(1).mkv"

    # Initialize Main object to handle the pipeline
    # main = Main(path=TEMP_path, depth=TEMP_depth, mode=0)
    main = Main(path=TEMP_video, mode=1)
    # main = Main(mode=2)

    # Begin the pipeline
    main.begin()

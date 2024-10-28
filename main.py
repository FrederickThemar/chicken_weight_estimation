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

        # Used to store a list of the 
        self.table = {}

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
        # depth = cv2.imread(self.depth_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = o3d.io.read_image(self.depth_path)
        # rgb_im = o3d.io.read_image(self.color_path)

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

        # print(f'\nOUTPUT:\t\t\n{outputs}')
        for i in range(len(accep_idxs)):
            # print(f'ID: {i}')
            output = outputs[i][0][0]
            print(f'ID {ids[i]}: {output} kg')
        print()
        print(f'YOLO time:   {self.yolo.times}')
        print(f'PCD time:    {self.pcd.times}')
        print(f'KPConv time: {self.kpconv.times}')

    # Runs a given RGBD frame through the pipeline.
    def handle_rgbd(self, rgbd, count):
        # Store the rgb and depth frames
        # rgb = rgbd.rgb
        depth = rgbd.depth
        
        # This might not be needed!
        # DEBUG_depthCV2 = np.asarray(depth)
        # DEBUG_depthCV2 = cv2.cvtColor(DEBUG_depthCV2, cv2.COLOR_RGB2BGR)
        rgb = np.asarray(rgbd.color)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # print(DEBUG_depthCV2.dtype)
        # cv2.imwrite('img.png',DEBUG_depthCV2)
        # o3d.io.write_image('img2.png', depth)
        # cv2.waitKey(0)
        
        # DEBUG: REMOVE LATER
        # print(count+1)
        # success, masks, overlay, ids = self.yolo.mask_frame(rgb, self.save)
        success, masks, ids, boxes = self.yolo.mask_frame(rgb, self.save)
        
        if len(masks) != len(ids) or len(masks) != len(boxes):
            print("ERROR: LENGTHS DON'T MATCH")
            print(len(masks))
            print(len(ids))
            print(len(boxes))
            exit(1)
        # DEBUG:
        # if len(masks) != len(ids):
        #     print("ERROR: LENGTHS DON'T MATCH")
        #     print(count+1)
        #     print()
        if not success:
            # Write frame count
            rgb = cv2.rectangle(rgb, (0, 0), (250, 50), (200,0,0), -1)
            rgb = cv2.putText(
                rgb,
                f'Frame: {count+1}',
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.imshow('frame', rgb)
            # cv2.waitKey(10)
            
            return None, rgb
            # count+=1
            # continue
            
        # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(f'Len IDS: \t{len(ids)}')
        # print(f'Len con: \t')
        # print(count+1)
        pcds, accep_masks, pcd_idxs = self.pcd.pcd_frame(rgb, depth, masks, self.save)
        # print(ids)
        # print(pcd_idxs)
        # Write frame count
            # cv2.imshow('frame', rgb)
            # cv2.waitKey(10)
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
            # count+=1
            # continue

        # print("MADE IT TO KPCONV")
        # if len(pcds) > 1:
        #     print(count+1)
        #     print(len(pcds))
        #     print()
        # print()
        outputs, accep_idxs = self.kpconv.estimate_frame(pcds, pcd_idxs)
        # print(len(outputs[0]))
        # outputs.append(output)
        
        # Draw acceptable masks and data onto image
        overlay = rgb.copy()
        # print(accep_idxs)

        # Update the table
        for i in range(len(accep_idxs)):
            # print(f'ID: {i}')
            # Initialize ID in table if not already there
            if ids[accep_idxs[i]] not in self.table:
                self.table[ids[accep_idxs[i]]] = {}
                self.table[ids[accep_idxs[i]]]["list"] = []
                self.table[ids[accep_idxs[i]]]["curr_size"] = 0
                self.table[ids[accep_idxs[i]]]["curr_avg"] = 0
            
            # Add weight to corresponding ID array, update moving average
            new_weight = outputs[i][0][0]
            self.table[ids[accep_idxs[i]]]["list"].append(new_weight)           # Add new weight
            curr_size = self.table[ids[accep_idxs[i]]]["curr_size"]             # Extract data for ID from table
            curr_avg = self.table[ids[accep_idxs[i]]]["curr_avg"]
            self.table[ids[accep_idxs[i]]]["curr_size"]+=1                      # Increment number of weight estimates for given ID
            new_avg = ((curr_avg * curr_size) + new_weight ) / (curr_size + 1)  # Calc new avg
            self.table[ids[accep_idxs[i]]]["curr_avg"] = new_avg                # Update avg for given ID

            # self.table[ids[accep_idxs[i]]]["list"].append(outputs[i][0][0])

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
        # for output in outputs:
        #     # Write output on visualization frame
        #     overlay = cv2.rectangle(overlay, (0, 50), (250, 100), (0,200,0), -1)
        #     overlay = cv2.putText(
        #         overlay,
        #         'Output: %.2f' % round(output[0][0],2),
        #         (15, 85),
        #         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # if len(outputs) > 1:
        #     print(count+1)
        #     for i in range(len(accep_ids)):
        #         print(f'ID: \t{accep_ids[i]}')
        #         print(f'Wei:\t{outputs[i]}')
        #     print(f'IDs: \t{ids}')
        #     print()

        # Add info to table
        # print(len(accep_idxs))
        # print(len(ids))
        # print(len(outputs))
        # print()
        # # Display the visualization frame
        # cv2.imshow('frame', overlay)
        # cv2.waitKey(10)
        
        return outputs, overlay

        # count+=1
        # break


    # Estimates weight of all the frames in a video
    def process_video(self):
        # Maybe use process_frame for this? 
        #   - Can have pframe return estimate and append to list
        #   - Save the frames separately, then have a list storing the paths to each. Give to pframe() 
        #       - Maybe each item in array can be another JSON, with paths to mask, depth, and rgb?
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

            output, overlay = self.handle_rgbd(rgbd, count)

            if output is None:
                cv2.imshow('frame', overlay)
                cv2.waitKey(10)
            
                count+=1
                continue

            # Display the visualization frame
            cv2.imshow('frame', overlay)
            cv2.waitKey(10)

            outputs.append(output)
            count+=1
            # DEBUG: REMOVE LATER
            # if count > 60:
            #     break
            
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
        # print(f'\nCount avg time:   {np.mean(self.kpconv.count_times)}')
        # print(f'Batch avg time:   {np.mean(self.kpconv.batch_times)}')
        # print(f'Output avg time:  {np.mean(self.kpconv.output_times)}')
        print(f'\nLoop start time:  {np.mean(self.kpconv.loop_times)}')
        # print(f'For loop len:     {np.mean(self.kpconv.counts_len)}')
        # print(f'For loop count:   {np.mean(self.kpconv.counts_count)}')
        # print(f'Lens: count {len(self.kpconv.count_times)}, batch {len(self.kpconv.batch_times)}, outputs {len(self.kpconv.output_times)}, loop {len(self.kpconv.loop_times)}')

        # DEBUG: print table
        print(self.table)

    # Callback function. Exit loop in process_live when user hits Esc
    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def process_live(self):
        # Idea: Have it show avg error every 10 frames processed
        #   - Can show a message saying 'No chicken detected!' if nothing seen by YOLO yet.
        #   - Use a Bool to see if chicken detected since program began, and a second to see if prev message printed yet
        print("BEGIN LIVE PROCESS")
        # Config for live Azure
        config = o3d.io.AzureKinectSensorConfig()

        # # viewer = ViewerWithCallback(config, 0, True)
        # # viewer.run()

        # Create the object for reading from the camera
        self.sensor = o3d.io.AzureKinectSensor(config)
        device = 0
        if not self.sensor.connect(device):
            raise RuntimeError("Failed to connect to sensor.")

        self.flag_exit = False
        outputs = []
        start = time.time()
        while not self.flag_exit:

            rgbd = self.sensor.capture_frame()
            if rgbd is None:
                continue

            output, overlay = self.handle_rgbd(rgbd, count)

            if output is None:
                cv2.imshow('frame', overlay)
                cv2.waitKey(10)
            
                count+=1
                continue

            # Display the visualization frame
            cv2.imshow('frame', overlay)
            cv2.waitKey(10)

            outputs.append(output)

            # Want loop to end after 60 seconds
            end = time.time()
            duration = end - start
            if duration > 60:
                self.flag_exit = True
        
        # Close cv2 window
        cv2.destroyAllWindows()


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

    # Used for mode=0, process frame
    # TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240409/chicken20/color/000098.jpg'
    # TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Spring2024/20240409/chicken20/depth/000098.png'
    # These ones have two chickens
    TEMP_path = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Summer2024/20240619/color/000254.jpg'
    TEMP_depth = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Depth/Summer2024/20240619/depth/000254.png'

    # Used for mode=1, process video
    # TEMP_video = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Spring2024/20240409/chicken_16.mkv'
    # TEMP_video = '/mnt/khoavoho/datasets/chicken_weight_dataset/jzbumgar/Spring2024/20240306/chicken_16.mkv'

    # Used with mode=1, has multiple chickens
    # TEMP_video = "/home/jzbumgar/datasets/Summer_videos/20240617_chicken03.mkv"
    TEMP_video = "/home/jzbumgar/datasets/Summer_videos/20240619_chicken03.mkv"
    # TEMP_video = "/home/jzbumgar/Downloads/chicken_07(1).mkv"

    # Initialize Main object to handle the pipeline
    main = Main(path=TEMP_path, depth=TEMP_depth, mode=0)
    # main = Main(path=TEMP_video, mode=1)
    # main = Main(mode=2)

    # Begin the pipeline
    main.begin()
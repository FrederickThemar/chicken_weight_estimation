import os
import torch
import cv2
import time
import numpy as np

from tqdm import tqdm
from ultralytics import YOLO
# from main import log

class yolo():
    def __init__(self):
        print("Initializing YOLO model... ", end="", flush=True)

        # self.modelName = "20240615_med15" # Old model, can only segment one chicken.
        self.modelName = "20241013_mult2"

        # Initialize model
        self.model = YOLO(f'./YOLO/{self.modelName}.pt')

        # Switch model to use CUDA
        self.model.to('cuda')

        # Where most recent mask for project is saved
        self.savedMask = None

        # Used when drawing colored masks onto isolated frame
        self.colors = [(0,255,0),(0,0,255),(255,0,0),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(255,127,64),(127,255,64),(127,64,255)]

        # DEBUG: Used for tracking how long yolo takes to process an image
        self.times = []

        print("Done!")

    def mask_frame(self, rgb, save):

        # Log start time of function
        start = time.time()

        # results = self.model(color, conf=0.80, verbose=False)
        results = self.model.track(rgb, conf=0.80, verbose=False, persist=True, tracker="YOLO/bytetrack.yaml")

        success  = False
        isolated = None
        overlay  = None
        IDs      = []
        boxes    = []
        masks    = []
        for result in results:
            mask = result.masks
            if mask is None:
                continue
            else:
                success = True

            origPath = result.path
            origEnd = origPath.split('/')[-1].split('.')[0]

            if save:
                # NOTE: Change this later to save to different directory
                overlayPath = f'{self.maskOutput}/overlay/{origEnd}.jpg'
                result.save(filename=overlayPath)

            # Isolate masks on a blank image
            img = np.copy(result.orig_img) # Original color frame

            # Get each mask on image
            for ci, c in enumerate(result):
                isolated = np.zeros(img.shape[:3], np.uint8) # Blank image to draw masks onto

                # Check if the tracking failed. If so, discard mask.
                if c.boxes.id is None:
                    continue

                # Get the ID of the object in the result
                obj_id = int(c.boxes.id[0])
                IDs.append(obj_id)
                box = (c.boxes.xyxy).tolist()[0]
                boxes.append(box)

                # Create contour mask
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(isolated, [contour], -1, self.colors[(obj_id % len(self.colors))-1], cv2.FILLED)

                # Add mask to list
                masks.append(isolated)

            # Save isolated mask into temp folder
            if save:
                maskOutputDir = f'{self.maskOutput}/{origEnd}.png'
                cv2.imwrite(maskOutputDir, isolated)

        if success is not False:
            # Save process time
            end = time.time()
            duration = (end - start)
            self.times.append(duration)

        # return success, masks, overlay, IDs
        return success, masks, IDs, boxes

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

        # Check output path
        self.maskOutput = "./outputMask/"
        self.checkPath(self.maskOutput)

        # Initialize model
        self.model = YOLO(f'./YOLO/{self.modelName}.pt')

        # Switch model to use CUDA
        self.model.to('cuda')

        # Where most recent mask for project is saved
        self.savedMask = None

        # Used when drawing colored masks onto overlay frame
        self.alpha = 0.3
        self.beta = 1 - self.alpha
        self.colors = [(255,255,255)]

        # DEBUG: Used for tracking how long yolo takes to process an image
        self.times = []

        print("Done!")

    def mask_frame(self, rgb, save):
        # print("\nMasking frame... ")
        # self.frame_path = path

        # Log start time of function
        start = time.time()

        # results = self.model(color, conf=0.80, verbose=False)
        results = self.model.track(rgb, conf=0.80, verbose=False, persist=True, tracker="YOLO/bytetrack.yaml")

        success  = False
        isolated = None
        overlay  = None
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
            isolated = np.zeros(img.shape[:3], np.uint8) # Blank image to draw masks onto

            # Get each mask on image
            # labels = [] NOT USED
            for ci, c in enumerate(result):
                label = c.names[c.boxes.cls.tolist().pop()]
                # labels.append(label)

                # Create contour mask
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(isolated, [contour], -1, (0,0,255), cv2.FILLED)

            # Draw part mask covers onto blank image
            # mask3ch = cv2.cvtColor(isolated, cv2.COLOR_GRAY2BGR)
            # isolated = cv2.bitwise_and(mask3ch, img)

            # Save isolated mask into temp folder
            if save:
                maskOutputDir = f'{self.maskOutput}/{origEnd}.png'
                cv2.imwrite(maskOutputDir, isolated)

            self.savedMask = isolated
            # overlay = np.asarray(result.cpu().numpy())
            # overlay = None
            # overlay = result.plot()

            # Draw mask onto overlay frame
            overlay = img.copy()
            mask_location = isolated.astype(bool)
            # print(overlay)
            # print()
            overlay[mask_location] = cv2.addWeighted(img, self.alpha, isolated, self.beta, 0.0)[mask_location]

        if success is not False:
            # Save process time
            end = time.time()
            duration = (end - start)
            self.times.append(duration)

        return success, isolated, overlay

    def mask_video(self, path):
        self.video_path = path
        print(self.video_path)

    def mask_live(self):
        # Not sure how to do this one. Maybe the Main object should save some frames from video then call this?
        pass

    # HELPER: Checks if input filepath exists.
    def checkPath(self, inputPath):
        if not os.path.exists(inputPath):
            print("ERROR: Path does not exist.")
            print(f'Path: {inputPath}')
            exit(1)
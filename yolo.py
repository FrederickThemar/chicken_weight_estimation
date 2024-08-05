import os
import torch
import cv2
import numpy as np

from tqdm import tqdm
from ultralytics import YOLO
# from main import log

class yolo():
    def __init__(self):
        print("Initializing YOLO model... ", end="", flush=True)

        self.modelName = "20240615_med15"

        # Check output path
        self.maskOutput = "./outputMask/"
        self.checkPath(self.maskOutput)

        # Initialize model
        self.model = YOLO(f'./YOLO/{self.modelName}.pt')

        # Switch model to use CUDA
        self.model.to('cuda')

        # Where most recent mask for project is saved
        self.savedMask = None

        print("Done!")

    def mask_frame(self, path, save):
        # print("\nMasking frame... ")
        self.frame_path = path
        
        results = self.model(self.frame_path, conf=0.80, verbose=False)

        success = False
        isolated = None
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
            b_mask = np.zeros(img.shape[:2], np.uint8) # Blank image

            # Get each mask on image
            labels = []
            for ci, c in enumerate(result):
                label = c.names[c.boxes.cls.tolist().pop()]
                labels.append(label)

                # Create contour mask
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255,255,255), cv2.FILLED)

            # Draw part mask covers onto blank image
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)

            # Save isolated mask into temp folder
            if save:
                maskOutputDir = f'{self.maskOutput}/{origEnd}.png'
                cv2.imwrite(maskOutputDir, isolated)

            self.savedMask = isolated

        return success, isolated

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
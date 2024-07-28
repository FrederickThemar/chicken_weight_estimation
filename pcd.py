import os
import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm


# Initial box shown to user
# PRESET_BOX = []

class pcd():
    def __init__(self):
        print("Initializing pointcloud module... ", end="", flush=True)
        
        self.cam = o3d.camera.PinholeCameraIntrinsic(
            width=1280,
            height=720,
            fx=608.66650390625,
            fy=608.518310546875,
            cx=643.125,
            cy=365.3201904296875,
        )

        # Path PCDs get temporarily stored in
        self.outputPath = "./outputPCD/"
        self.checkPath(self.outputPath)

        # Store default confidence box
        self.defaultBox = [[350, 325],[1400,925]] 

        print("Done!")

    def pcd_frame(self):
        pass

    def pcd_video(self):
        pass

    def pcd_live(self):
        pass

    # HELPER: Checks if input filepath exists.
    def checkPath(self, inputPath):
        if not os.path.exists(inputPath):
            print("ERROR: Path does not exist.")
            print(f'Path: {inputPath}')
            exit(1)

# Showing the user a bounding box for a given image, and taking user input to make new box.
class BoundingBoxWidget(object):
    def __init__(self, imgPath, coords=[], lower=None):
        currDir = os.path.dirname(__file__)
        self.original_image = cv2.imread(imgPath)
        self.clone = self.original_image.copy()
        self.date = imgPath.split('/')[-4]
        self.id = imgPath.split('/')[-3]

        self.lower_bound = lower

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = coords

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print(f'Date: {self.date}, ID: {self.id}')
            print('top left: {}, bottom right: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            print('x,y,w,h : ({}, {}, {}, {})'.format(self.image_coordinates[0][0], self.image_coordinates[0][1], self.image_coordinates[1][0] - self.image_coordinates[0][0], self.image_coordinates[1][1] - self.image_coordinates[0][1]))
            print('\n')

            # Save lower bound into attribute
            self.lower_bound = max(self.image_coordinates[0][1], self.image_coordinates[1][1])
            
            # Draw rectangle 
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        if len(self.image_coordinates) == 2:
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
        return self.clone

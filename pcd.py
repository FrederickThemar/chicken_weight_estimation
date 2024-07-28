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

        self.d_lo, self.d_hi = 10, 1500

        # Path PCDs get temporarily stored in
        self.outputPath = "./outputPCD/"
        self.checkPath(self.outputPath)

        # Store default confidence box
        self.defaultBox = [[350, 325],[1400,925]] 

        # Where filepaths to output PCDs are saved
        self.pcdPaths = []

        print("Done!")

    def pcd_frame(self, colorPath, depthPath, maskPath, save):
        print("POINTCLOUD FRAME")

        self.color_path = colorPath
        self.depth_path = depthPath
        self.mask_path  = maskPath

        # Create output filepath
        origEnd = self.color_path.split('/')[-1].split('.')[0]
        pcdPath = f'{self.outputPath}/{origEnd}.ply'
        
        accep_mask = False
        

    def pcd_video(self):
        pass

    def pcd_live(self):
        pass

    # HELPER: Tests a mask image to see if it is within the confidence box
    def testMask(framepath, box):
        # Store the bounds of the box for ease of access
        yLower = box[0][1]
        yUpper = box[1][1]
        xLower = box[0][0]
        xUpper = box[1][0]
        
        origMask = cv2.imread(framepath, cv2.IMREAD_GRAYSCALE)

        # Get the contours for the mask
        origMask = cv2.imread(framepath, cv2.IMREAD_GRAYSCALE)
        mask = origMask.astype(bool)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxAcceptable = True
        pointCount = 0
        numBadContours = 0
        for contour in contours:
            for point in contour:
                pointCount+=1
                
                # Store coordinate parts separately
                point_x = point[0][0]
                point_y = point[0][1]
                
                # Point format: [x, y]. x=0 is far left, y=0 is top of the screen.
                # If point is outside the bounds, set bool to False and exit loop.
                if point_x <= xLower or point_x >= xUpper:
                    numBadContours+=1
                    break
                if point_y <= yLower or point_y >= yUpper:
                    numBadContours+=1
                    break

        # Check to see how many contours in the image are bad.
        if numBadContours == len(contours):
            boxAcceptable = False

        return boxAcceptable

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

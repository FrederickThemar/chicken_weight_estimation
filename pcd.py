import os
import cv2
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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

        # Store default confidence box
        # self.defaultBox = [[350, 325],[1400,925]]
        self.defaultBox = [[350, 245],[1400,1000]] # NEW BOX

        self.times = []

        print("Done!")

    def pcd_frame(self, color, depth, masks, save):
        # self.color_path = colorPath
        # self.depth_path = depthPath
        # self.color = color
        # self.depth = depth
        # depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        # print(depth.dtype)

        # cv2.imshow('frame', np.asarray(depth))
        # cv2.waitKey(0)


        # Log start time of function
        start = time.time()

        pcds = []
        accep_masks = []
        accep_idxs   = []

        # Check that any masks have been provided
        if len(masks) == 0:
            return pcds, accep_masks, accep_idxs

        # self.mask = masks[0]

        # Create output filepaths
        # origEnd = self.color_path.split('/')[-1].split('.')[0]
        # pcdPath = f'{self.outputPath}/pcd.ply'
        for i in range(len(masks)):
            curr_mask = masks[i]
            accep_mask = self.testMask(curr_mask, self.defaultBox)
            if not accep_mask:
                # print("\nERROR: The mask for this frame falls outside the acceptable boundaries for the model. Try a different frame.")
                # exit(1)
                # print("MASK NOT ACCEP")
                # print(f'MASK {i} FAILS')
                # print(len(masks))
                continue

            # rgb = plt.imread(self.color_path) / 255
            # rgb = self.color / 255
            # depth = cv2.imread(self.depth_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            # frame = cv2.imread(self.mask, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # Convert depth to openCV image to work with mask
            depth_copy = np.asarray(depth).copy()
            temp_depth = depth_copy

            frame = curr_mask
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(frame)
            DEBUG_temp = np.zeros_like(temp_depth)
            mask[frame > 0] = 255
            # mask[temp_depth < self.d_lo] = 0
            # mask[temp_depth > self.d_hi] = 0

            # Stop here if this happens
            # if mask.all() == DEBUG_temp.all():
            #     print("MASK EMPTY")
                # continue
            # if i == 1:
            #     cv2.imshow('frame', np.asarray(temp_depth))
            #     cv2.waitKey(0)
            temp_depth[mask == 0] = 0

            # If depth is empty, the PCD will be as well. Thus, skip the mask
            # if (temp_depth == DEBUG_temp).all():
            #     print(f'101 SKIP MASK {i}')
            #     continue

            # Create pointcloud using depthmask
            # rgb_im = o3d.io.read_image(self.color_path)
            rgb_im = o3d.geometry.Image(color)
            # depth_im = o3d.geometry.Image(depth)
            depth_im = o3d.geometry.Image(temp_depth)
            rgbd_im = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_im,
                self.cam
            )
            pcd.translate(-pcd.get_center())
            # if len(np.asarray(pcd.points).astype(np.float32)) == 0:
                # cv2.imshow('frame', np.asarray(rgb_im))
                # cv2.waitKey(0)
            # print("APPENDING PCD")

            # if len(np.asarray(pcd.points).astype(np.float32)) < 10:
            #     print(f'MASK {i} TOO SMALL')
            #     continue

            # Store PCD, mask, and index
            pcds.append(pcd)
            accep_masks.append(curr_mask)
            accep_idxs.append(i)
            # print(f'APPENDING MASK {i}')
            # print(len(np.asarray(pcd.points).astype(np.float32)))

        # o3d.io.write_point_cloud(pcdPath, pcd)

        # Save process time
        end = time.time()
        duration = (end - start)
        self.times.append(duration)

        return pcds, accep_masks, accep_idxs
        

    def pcd_video(self):
        pass

    def pcd_live(self):
        pass

    # HELPER: Tests a mask image to see if it is within the confidence box
    def testMask(self, yoloMask, box):
        # Store the bounds of the box for ease of access
        yLower = box[0][1]
        yUpper = box[1][1]
        xLower = box[0][0]
        xUpper = box[1][0]
        
        # origMask = cv2.imread(framepath, cv2.IMREAD_GRAYSCALE)

        # Get the contours for the mask
        origMask = cv2.cvtColor(yoloMask, cv2.COLOR_BGR2GRAY)
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

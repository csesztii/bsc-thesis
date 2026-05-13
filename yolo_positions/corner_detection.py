import math
import cv2
import random
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time

import VIS_sidewalk_src_cc_utils_cyb as goose_lib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from utils.utils import IntrinsicParameters

def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [int(x), int(y)]

class CornerDetector:
    def __init__(self, project_dir, right_video, left_video, top_video):
        self.align = 0
        self.project_dir = project_dir
        self.right_background = self.createBG(right_video)
        self.left_background = self.createBG(left_video)
        self.top_background = self.createBG(top_video)
        self.align = None

    def get_all_backgrounds(self):
        return self.right_background, self.left_background, self.top_background

    def createBG(self, video_path):
        capture = cv2.VideoCapture(video_path)
        frames_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        from_length = 1000 if frames_count > 1000 else frames_count
        video_name_with_extension = os.path.basename(video_path)
        video_name_without_extension = os.path.splitext(video_name_with_extension)[0]
        bg_dir_path = os.path.join(self.project_dir, 'backgrounds')
        
        os.makedirs(bg_dir_path, exist_ok=True)
        
        bg_image_path = os.path.join(bg_dir_path, video_name_without_extension + "_bg.png")
        if os.path.exists(bg_image_path):
            try:
                bgimg = cv2.imread(bg_image_path)
            except Exception as e:
                bg_image_path = os.path.join(self.project_dir, 'backgrounds' ,   video_name_without_extension + "_copy.png")
                bgimg = self.create_bg_image_fromvideo(capture, bg_image_path, fromLength=from_length, IMG_LIMIT=100, doreturn=1, dowrite=1)
        else:
            bgimg = self.create_bg_image_fromvideo(capture, bg_image_path, fromLength=from_length, IMG_LIMIT=100, doreturn=1, dowrite=1)
        
        return bgimg
    
    def mode(self, a, axis=0):
        scores = np.unique(np.ravel(a))       # get ALL unique values
        testshape = list(a.shape)
        testshape[axis] = 1
        oldmostfreq = np.zeros(testshape)
        oldcounts = np.zeros(testshape)

        for score in scores:
            template = (a == score)
            counts = np.expand_dims(np.sum(template, axis),axis)
            mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
            oldcounts = np.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent

        return mostfrequent, oldcounts

    def create_bg_image_fromvideo(self, capture, bg_image_path, fromLength=1000, IMG_LIMIT=100, doreturn=0, dowrite=1, save_frameids_path=None):
        IMG_LIMIT = IMG_LIMIT if IMG_LIMIT<=fromLength else fromLength
        print("[corner_detection.py] :: Calculating background image from video, use ", IMG_LIMIT, " frames from the first ", fromLength)
        start_time = time.time()

        useframes = random.sample(range(fromLength), IMG_LIMIT)
        print("Using these frames: ", useframes)
        print("Useframes length: ", len(useframes))
        if save_frameids_path is not None:
            np.array(useframes).tofile(save_frameids_path, sep = ',')

        bg_samples = []
        
        frameNr = 0

        print("Start while loop: ", time.time() - start_time)
        while True: 
            success, img = capture.read()
            if success and frameNr in useframes:
                bg_samples.append(img)
            frameNr=frameNr+1
            if len(bg_samples)==IMG_LIMIT:
                break
        
        print("Start stack: ", time.time() - start_time)
        bg_samples = np.stack(bg_samples, axis=0)
        print("Start mode: ", time.time() - start_time)
        test_bg_image, _counts = self.mode(bg_samples, axis=0)
        print("Start squeeze: ", time.time() - start_time)
        test_bg_image = np.squeeze(test_bg_image)
        
        if dowrite and bg_image_path is not None: 
            print('[corner_detection.py] :: Save bg file to: ', bg_image_path)
            cv2.imwrite(bg_image_path, test_bg_image)

        if doreturn==1: 
            return test_bg_image.astype(np.uint8)
        
    def find_green_area(self, img, visualize=False):
        #hue sat value
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # low and upper green
        lower_green = np.array([40, 40, 60]) #80
        upper_green = np.array([85, 255, 255]) #150
        
        #green mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_mask = np.zeros_like(mask)

        # largest contour
        largest_contour = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if visualize:
                cv2.drawContours(img, [largest_contour], -1, (0, 0, 255), 3)
                cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(img_rgb)
                ax[0].set_title("Detected Green Area")
                ax[0].axis("off")

                ax[1].imshow(largest_mask, cmap="gray")
                ax[1].set_title("Green Mask")
                ax[1].axis("off")

                plt.show()
            
        return largest_contour
    
    def split_and_find_largest_quadrilateral(self, contour, visualize=False, image=None):
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Create a blank mask of the bounding box
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw the contour on the mask
        cv2.drawContours(mask, [contour - [x, y]], -1, 255, thickness=cv2.FILLED)

        # Find all external contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        # Sort contours by area (largest first)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour with fewer points (quadrilateral)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        for i in range(10):
            if(len(approx) != 4):
                epsilon += 0.01 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            else:
                break

        # Ensure we only return 4 points
        if len(approx) == 4:
            approx_points = approx.reshape(4, 2) + [x, y]  # Adjust back to original image coordinates
            #approx_points = [tuple(map(lambda p: np.int32(p), point)) for point in approx_points]
            return approx_points
        else:
            return None
        
    def find_concave_point(self, contour, visualize=False, image=None):
        if contour is None or len(contour) < 3:
            return []

        hull = cv2.convexHull(contour, returnPoints=False)

        defects = cv2.convexityDefects(contour, hull) if len(hull) > 3 else None

        concave_points = []

        if defects is not None:
            for i in range(defects.shape[0]):
                start_idx, end_idx, farthest_idx, _ = defects[i, 0]
                farthest = tuple(contour[farthest_idx][0])  # Concave point
                concave_points.append(farthest)
                
        #find the center of the shape (centroid)
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])  # Center x
        cy = int(moments['m01'] / moments['m00'])  # Center y

        # Find the concave point closest to the center
        closest_point = None
        min_distance = float('inf')
        
        for point in concave_points:
            x, y = point
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)  # Euclidean distance to center

            if distance < min_distance:
                min_distance = distance
                closest_point = point

        # If there are multiple points at the same minimum distance, select the highest one
        # highest_point = closest_point
        # for point in concave_points:
        #     x, y = point
        #     if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) == min_distance and y < highest_point[1]:
        #         highest_point = point

        return [int(closest_point[0]), int(closest_point[1])]
        
    def categorize_points(self, points, y_threshold, x_threshold, image=None):
        """
        Categorize 5 points into 'Top', 'Left', and 'Bottom' based on their positions.
        Returns:
        - categorized_points: Dictionary with categorized points: 'Top', 'Bottom', and 'Left'.
        """
        
        categorized_points = {
            'Top': [],
            'Bottom': [],
            'Left': []
        }

        for pt in points:
            x, y = pt
            
            if x < x_threshold:
                categorized_points['Left'].append(pt)
            else:
                # Categorize as Top or Bottom based on y_threshold only if it's not in the Left category
                if y < y_threshold:
                    categorized_points['Top'].append(pt)
                else:
                    categorized_points['Bottom'].append(pt)
            

        # Sort points for better visualization
        categorized_points['Top'] = sorted(categorized_points['Top'], key=lambda p: p[0])  # Sort by x coordinate
        categorized_points['Left'] = sorted(categorized_points['Left'], key=lambda p: p[1])  # Sort by y coordinate

        return categorized_points
    
    def find_all_intersections(self, approx, min_dist):
        intersections = []
        num_points = len(approx)
        
        # Loop through all pairs of lines
        for i in range(num_points):
            for j in range(i + 1, num_points):
                # Define the lines as pairs of points
                x1, y1 = approx[i]
                x2, y2 = approx[(i + 1) % num_points]

                x3, y3 = approx[j]
                x4, y4 = approx[(j + 1) % num_points]

                # ha barmely vonal rovid, akk nem kell
                if np.linalg.norm(np.array([int(x1), int(y1)]) - np.array([int(x2), int(y2)])) < min_dist or \
                np.linalg.norm(np.array([int(x3), int(y3)]) - np.array([int(x4), int(y4)])) < min_dist:
                    continue
                
                line1 = ((int(x1), int(y1)), (int(x2), int(y2)))
                line2 = ((int(x3), int(y3)), (int(x4), int(y4)))
                
                # Compute the intersection of the two lines
                intersection = line_intersection(line1, line2)
                
                if intersection is None:
                    continue
                
                # Only add unique intersection points (avoid duplicates)
                if intersection not in intersections and intersection[0] > 0 and intersection[1] > 0:
                    intersections.append(intersection)
                
        
        return intersections
    
    def is_this_valid_triangle(self, points):
        max_length = 0
        longest_side = None

        for (type1, point1) in points.items():
            for (type2, point2) in points.items():
                x1, y1 = point1
                x2, y2 = point2
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                if length > max_length:
                    max_length = length
                    longest_side = (type1, type2)

        return set(longest_side) == set(['primary_front_corner', 'secondary_back_corner'])

    def get_top_points(self):
        """"
            Returns the four corner points in a dictionary and the transformation matrix (H)
        """
        params = IntrinsicParameters()
        height, width = self.top_background.shape[:2]
        input_size = params.input_size
        cast_size = params.cast_size
        undistorted_top, _ = goose_lib.undistort(self.top_background, params.IntrinsicMatrix, params.distortionCoeff, input_size=input_size, cast_size=cast_size, output_size=input_size, output_camera_matrix=True)
    
        cv2.imwrite(os.path.join(self.project_dir, 'backgrounds', 'top_undistorted.png'), undistorted_top)
        
        largest_contour = self.find_green_area(undistorted_top)
        rect = self.split_and_find_largest_quadrilateral(largest_contour)
        concave_point = self.find_concave_point(largest_contour)
        corner_points = [*rect.tolist(), concave_point]
        x, y, w, h = cv2.boundingRect(largest_contour)
        categorized_points = self.categorize_points(corner_points, y+h//2, x+w//4)
        
        if len(categorized_points['Left']) < 2 or len(categorized_points['Top']) < 1 or len(categorized_points['Bottom']) < 1:
            return None, None

        if len(categorized_points['Top']) > 1:
            target_point = line_intersection((categorized_points['Top'][0], categorized_points['Top'][1]), (categorized_points['Left'][0],categorized_points['Left'][1]))
            self.align = 0
            target_point = [int(target_point[0]), int(target_point[1])]
            top_left = target_point
            bottom_left = max(categorized_points['Left'], key=lambda point: point[1])
        elif len(categorized_points['Bottom']) > 1:
            target_point = line_intersection((categorized_points['Bottom'][0], categorized_points['Bottom'][1]), (categorized_points['Left'][0],categorized_points['Left'][1]))
            self.align = 1
            target_point = [int(target_point[0]), int(target_point[1])]
            top_left = min(categorized_points['Left'], key=lambda point: point[1])
            bottom_left = target_point
        
        top_right = max(categorized_points['Top'], key=lambda point: point[0])
        bottom_right = max(categorized_points['Bottom'], key=lambda point: point[0])

        points_img = np.array([[top_left, top_right, bottom_right, bottom_left]], dtype=np.float32)
        points_rectangle = points_rectangle = np.array([
            [0, 0],
            [275, 0],
            [275, 200],
            [0, 200] 
        ], dtype=np.float32)

        H, _ = cv2.findHomography(points_img, points_rectangle)
        return {"top_left": top_left, "top_right": top_right, "bottom_right": bottom_right, "bottom_left": bottom_left}, H
    
    def get_difficult_side_points(self):
        if self.align is not None:
            if self.align == 1:
                original_bg_image = self.left_background
                generated_background_img = cv2.flip(original_bg_image, 1) # 1 horizontally
                undistort_png_name = 'left_undistorted.png'
            else:
                generated_background_img = self.right_background
                undistort_png_name = 'right_undistorted.png'

            params = IntrinsicParameters()
            height, width = generated_background_img.shape[:2]
            input_size = params.input_size
            cast_size = params.cast_size
            undistorted_image, _ = goose_lib.undistort(generated_background_img, params.IntrinsicMatrix, params.distortionCoeff, input_size=input_size, cast_size=cast_size, output_size=input_size, output_camera_matrix=True)

            largest_contour = self.find_green_area(undistorted_image)
            epsilon = 0.006 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            approx = approx.reshape((-1, 2))  # Flatten the approximation
            centroid = np.mean(approx, axis=0)
            right_side_points = approx[approx[:, 0] >= centroid[0]-centroid[0]*0.2]

            intersection_points = self.find_all_intersections(right_side_points, 300)
            intersection_points = [p for p in intersection_points if 0 <= p[0] <= width and 0 <= p[1] <= height]

            primary_front_corner = max(intersection_points, key=lambda p: p[1])
            secondary_back_corner = min(intersection_points, key=lambda p: p[0])
            primary_back_corner = max([point for point in intersection_points if point[0] >= centroid[0] and point[1] <= centroid[1]], key=lambda p: p[0])

            corner_points = {'primary_front_corner': primary_front_corner, 'primary_back_corner': primary_back_corner, 'secondary_back_corner': secondary_back_corner}

            if self.align == 1:
                vis_img = cv2.flip(undistorted_image.copy(), 1)
                cv2.imwrite(os.path.join(self.project_dir, 'backgrounds', undistort_png_name), vis_img)
                for key, point in corner_points.items():
                    # important point mirroring
                    x,y = point
                    mirrored_x = width - x  # Calculate mirrored x-coordinate
                    point =  (mirrored_x, y)
                    corner_points[key] = point
            else: cv2.imwrite(os.path.join(self.project_dir, 'backgrounds', undistort_png_name), undistorted_image)
            
            return corner_points if self.is_this_valid_triangle(corner_points) else None
        else: raise Exception("You need to run CornerDetector.get_top_points first")
    
    def get_easy_side_points(self):
        if self.align is not None:
            if self.align == 1:
                original_bg_image = self.right_background 
                generated_background_img = cv2.flip(original_bg_image, 1) # 1 horizontally
                undistort_png_name = 'right_undistorted.png'
            else:
                generated_background_img = self.left_background
                undistort_png_name = 'left_undistorted.png'

            params = IntrinsicParameters()
            height, width = params.height, params.width
            input_size = params.input_size
            cast_size = params.cast_size
            undistorted_image, _ = goose_lib.undistort(generated_background_img, params.IntrinsicMatrix, params.distortionCoeff, input_size=input_size, cast_size=cast_size, output_size=input_size, output_camera_matrix=True)

            largest_contour = self.find_green_area(undistorted_image.copy(), visualize=True)
            epsilon = 0.006 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            approx = approx.reshape((-1, 2))  # Flatten the approximation
            centroid = np.mean(approx, axis=0)
            left_side_points = approx[approx[:, 0] <= centroid[0]+centroid[0]*0.2]

            primary_front_corner = max(left_side_points, key=lambda p: p[1]).tolist()
            secondary_back_corner = max(left_side_points, key=lambda p: p[0]).tolist()
            primary_back_corner = min([point for point in left_side_points if point[0] <= centroid[0] and point[1]<= centroid[1]], key=lambda p: p[0]).tolist()

            corner_points = {'primary_front_corner': primary_front_corner, 'primary_back_corner': primary_back_corner, 'secondary_back_corner': secondary_back_corner}
            
            if self.align == 1:
                vis_img = cv2.flip(undistorted_image.copy(), 1)
                cv2.imwrite(os.path.join(self.project_dir, 'backgrounds', undistort_png_name), vis_img)
                for key, point in corner_points.items():
                    # important point mirroring
                    x,y = point
                    mirrored_x = width - x  # Calculate mirrored x-coordinate
                    point =  (mirrored_x, y)
                    corner_points[key] = point
            else: cv2.imwrite(os.path.join(self.project_dir, 'backgrounds', undistort_png_name), undistorted_image)

            return corner_points if self.is_this_valid_triangle(corner_points) else None 
        else: raise Exception("You need to run CornerDetector.get_top_points first")
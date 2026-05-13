from ultralytics import YOLO
import decord as de
import numpy as np
import torch
import cv2
import os
import sys
import json
import pickle
import re
import time

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from corner_detection import CornerDetector
import VIS_sidewalk_src_cc_utils_cyb as goose_lib
from ruff_sync_demo_by_Gergo import Synchronizer as sync
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import utils.utils as utils
from sort import Sort

def get_current_time():
    return time.strftime("%Y-%m-%d_%H-%M-%S") + " "

class PositionDetecor:
    def __init__(self, right_video_path, left_video_path, top_video_path, project_dir, yolo_side_model_path, yolo_top_model_path):
        for path, name in [(right_video_path, "Right video"), (left_video_path, "Left video"), (top_video_path, "Top video")]:
            if not os.path.isfile(path):
                raise ValueError(f"{name} path does not point to a file: {path}")
            if not path.lower().endswith(".mp4"):
                raise ValueError(f"{name} path must be an .mp4 file: {path}")
            
        for path, name in [(yolo_side_model_path, "YOLO side model"), (yolo_top_model_path, "YOLO top model")]:
            if not os.path.isfile(path):
                raise ValueError(f"{name} path does not point to a file: {path}")
            if not path.lower().endswith(".pt"):
                raise ValueError(f"{name} path must be a .pt file: {path}")
        
        self.right_video_path = right_video_path
        self.left_video_path = left_video_path
        self.top_video_path = top_video_path

        self.project_dir = project_dir
        os.makedirs(self.project_dir, exist_ok=True)

        self.yolo_paths = {'right': yolo_side_model_path, 'left': yolo_side_model_path, 'top': yolo_top_model_path}

        # camera calibration
        self.H = []
        self.inv_rotation_matrices = {'right': None, 'left': None}
        self.params = utils.IntrinsicParameters()

        #init
        self.video_readers = {'right': None, 'left': None, 'top': None}
        
        self.top_points = None
        self.right_points = None
        self.left_points = None

        self.right_background = None
        self.left_background = None
        self.top_background = None

        self.sync_lists = None

        self.file_paths_masks = {'right': "", 'left': "", 'top': ""}
        self.text_file_paths_coords = {'right': "", 'left': "", 'top': ""}
        self.final_coords_file = ""

    # SYNC START
    def strip_zeros(self, *lists):
        # Transpose lists into a list of tuples (to keep elements in sync)
        filtered = [(tup) for tup in zip(*lists) if tup[0] != 0]
        # Transpose back to separate lists
        return [list(t) for t in zip(*filtered)] if filtered else [[]] * len(lists)
    # END

    # PROJECTION START
    def tx(self, x):
        return (x - self.params.OptimalIntrinsicMatrix[0,2])/self.params.OptimalIntrinsicMatrix[0,0]
    def ty(self, y):
        return (y - self.params.OptimalIntrinsicMatrix[1,2])/self.params.OptimalIntrinsicMatrix[1,1]

    # pitch
    def calculate_rotation_x(self, c0,z,t):
        core = (c0 - z*t)/(c0*t + z)
        return np.arctan(core)

    # roll
    def calculate_rotation_z(self, tx1,ty1,tx2,ty2):
        core = (ty1 - ty2)/(tx1 - tx2)
        return np.arctan(core)

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def recalculate_orientation(self, x_axis_middle_for_proj, smooth_y, corner_points, left_back_corner, right_back_corner, dist_from_floor):
        primary_front_corner = corner_points['primary_front_corner']
        primary_back_corner = corner_points['primary_back_corner']
        secondary_back_corner = corner_points['secondary_back_corner']
        
        x0 = x_axis_middle_for_proj
        y0 = int(smooth_y[x0])
        
        x1 = left_back_corner[0]
        y1 = int(smooth_y[x1])

        x2 = right_back_corner[0]
        y2 = int(smooth_y[x2])
        
        c0 = dist_from_floor
        z = self.params.opposite_line_dist
        minih = 45
        minc = (0.0,0.0,0.0)
        minxh = 1000000000.0
        for h in range(45,46,5):
            dist_from_floor = h
            c0 = dist_from_floor
            mini = -1
            minx = 1000000000.0
            t_angular_rotation=(self.calculate_rotation_x(c0,z,self.ty(y0)),-0.75,self.calculate_rotation_z(self.tx(x1),self.ty(y1),self.tx(x2),self.ty(y2)))
            for i in range(1500):
                _,inv_rotation_mat = goose_lib.calculate_rotation_matrix(t_angular_rotation)
                projected_coordinate_1 = goose_lib.inverse_projection((primary_front_corner[0],primary_front_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
                projected_coordinate_2 = goose_lib.inverse_projection((primary_back_corner[0],primary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
                t_angular_rotation = (t_angular_rotation[0],t_angular_rotation[1] + 0.001,t_angular_rotation[2])
                if minx > np.abs(projected_coordinate_1[0] - projected_coordinate_2[0]):
                    mini = t_angular_rotation[1]
                    minx = np.abs(projected_coordinate_1[0] - projected_coordinate_2[0])
            cal_angular_rotation=(self.calculate_rotation_x(c0,z,self.ty(y0)),mini,self.calculate_rotation_z(self.tx(x1),self.ty(y1),self.tx(x2),self.ty(y2)))
            mini = -1
            minx = 1000000000.0
            t_angular_rotation=(cal_angular_rotation[0],cal_angular_rotation[1],self.calculate_rotation_z(self.tx(x1),self.ty(y1),self.tx(x2),self.ty(y2))-0.75)
            for i in range(1500):
                _,inv_rotation_mat = goose_lib.calculate_rotation_matrix(t_angular_rotation)
                projected_coordinate_1 = goose_lib.inverse_projection((secondary_back_corner[0],secondary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
                projected_coordinate_2 = goose_lib.inverse_projection((primary_back_corner[0],primary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
                t_angular_rotation = (t_angular_rotation[0],t_angular_rotation[1],t_angular_rotation[2] + 0.001)
                if minx > np.abs(projected_coordinate_1[2] - projected_coordinate_2[2]):
                    mini = t_angular_rotation[2]
                    minx = np.abs(projected_coordinate_1[2] - projected_coordinate_2[2])
            cal_angular_rotation=(cal_angular_rotation[0],cal_angular_rotation[1],mini)
            mini = -1
            minx = 100000000.0
            t_angular_rotation=(cal_angular_rotation[0]-0.75,cal_angular_rotation[1],cal_angular_rotation[2])
            for i in range(1500):
                _,inv_rotation_mat = goose_lib.calculate_rotation_matrix(t_angular_rotation)
                projected_coordinate_1 = goose_lib.inverse_projection((secondary_back_corner[0],secondary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
                projected_coordinate_2 = goose_lib.inverse_projection((primary_back_corner[0],primary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
                t_angular_rotation = (t_angular_rotation[0] + 0.001,t_angular_rotation[1],t_angular_rotation[2])
                if projected_coordinate_1[2] < 0.0 or projected_coordinate_2[2] < 0.0:
                    continue
                if minx > np.abs(self.params.opposite_line_dist * 2.0 - (projected_coordinate_1[2] + projected_coordinate_2[2])):
                    mini = t_angular_rotation[0]
                    minx = np.abs(projected_coordinate_1[2] - projected_coordinate_2[2])
            cal_angular_rotation=(mini,cal_angular_rotation[1],cal_angular_rotation[2])
            projection_mat,rotation_mat, translation_vec ,inv_rotation_mat = goose_lib.calculate_extrinsic_matrix(cal_angular_rotation,(0.0,0.0,0.0))
            x_offset = goose_lib.inverse_projection((primary_back_corner[0],primary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,self.params.t)[0]
            proj_res = goose_lib.projection_func((x_offset,float(h),self.params.opposite_line_dist-500.0),self.params.OptimalIntrinsicMatrix,rotation_mat,(0.0,0.0,0.0),self.params.distortionCoeff)
            if np.abs(primary_front_corner[0] - proj_res[0]) * np.abs(primary_front_corner[1] - proj_res[1]) < minxh:
                minih = h
                minc = cal_angular_rotation
                minxh = np.abs(primary_front_corner[0] - proj_res[0]) * np.abs(primary_front_corner[1] - proj_res[1])
        cal_angular_rotation = minc
        dist_from_floor = 45#125#minih
        c0 = dist_from_floor
        mini = -1
        minx = 1000000000.0
        t_angular_rotation=(self.calculate_rotation_x(c0,z,self.ty(y0)),-0.75,self.calculate_rotation_z(self.tx(x1),self.ty(y1),self.tx(x2),self.ty(y2)))
        for i in range(1500):
            _,inv_rotation_mat = goose_lib.calculate_rotation_matrix(t_angular_rotation)
            projected_coordinate_1 = goose_lib.inverse_projection((primary_front_corner[0],primary_front_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
            projected_coordinate_2 = goose_lib.inverse_projection((primary_back_corner[0],primary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
            t_angular_rotation = (t_angular_rotation[0],t_angular_rotation[1] + 0.001,t_angular_rotation[2])
            if minx > np.abs(projected_coordinate_1[0] - projected_coordinate_2[0]):
                mini = t_angular_rotation[1]
                minx = np.abs(projected_coordinate_1[0] - projected_coordinate_2[0])
        cal_angular_rotation=(self.calculate_rotation_x(c0,z,self.ty(y0)),mini,self.calculate_rotation_z(self.tx(x1),self.ty(y1),self.tx(x2),self.ty(y2)))
        mini = -1
        minx = 1000000000.0
        t_angular_rotation=(cal_angular_rotation[0],cal_angular_rotation[1],self.calculate_rotation_z(self.tx(x1),self.ty(y1),self.tx(x2),self.ty(y2))-0.75)
        for i in range(1500):
            _,inv_rotation_mat = goose_lib.calculate_rotation_matrix(t_angular_rotation)
            projected_coordinate_1 = goose_lib.inverse_projection((secondary_back_corner[0],secondary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
            projected_coordinate_2 = goose_lib.inverse_projection((primary_back_corner[0],primary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
            t_angular_rotation = (t_angular_rotation[0],t_angular_rotation[1],t_angular_rotation[2] + 0.001)
            if minx > np.abs(projected_coordinate_1[2] - projected_coordinate_2[2]):
                mini = t_angular_rotation[2]
                minx = np.abs(projected_coordinate_1[2] - projected_coordinate_2[2])
        cal_angular_rotation=(cal_angular_rotation[0],cal_angular_rotation[1],mini)
        mini = -1
        minx = 1000000000.0
        t_angular_rotation=(cal_angular_rotation[0]-0.75,cal_angular_rotation[1],cal_angular_rotation[2])
        for i in range(1500):
            _,inv_rotation_mat = goose_lib.calculate_rotation_matrix(t_angular_rotation)
            projected_coordinate_1 = goose_lib.inverse_projection((secondary_back_corner[0],secondary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
            projected_coordinate_2 = goose_lib.inverse_projection((primary_back_corner[0],primary_back_corner[1]),(1,dist_from_floor),self.params.OptimalIntrinsicMatrix,inv_rotation_mat,(0.0,0.0,0.0))
            t_angular_rotation = (t_angular_rotation[0] + 0.001,t_angular_rotation[1],t_angular_rotation[2])
            if minx > np.abs(self.params.opposite_line_dist * 2.0 - (projected_coordinate_1[2] + projected_coordinate_2[2])):
                mini = t_angular_rotation[0]
                minx = np.abs(projected_coordinate_1[2] - projected_coordinate_2[2])
        cal_angular_rotation=(mini,cal_angular_rotation[1],cal_angular_rotation[2])
        projection_mat,rotation_mat, translation_vec ,inv_rotation_mat = goose_lib.calculate_extrinsic_matrix(cal_angular_rotation,(0.0,0.0,0.0))
        return cal_angular_rotation, projection_mat, rotation_mat, translation_vec ,inv_rotation_mat, dist_from_floor#minih#

    def convert_to_numpy(self,obj):
        if isinstance(obj, list):  # If it's a list, assume it was originally an ndarray
            return np.array(obj)
        return obj  # Otherwise, return as is

    # END

    # POSITIONS WITH YOLO
    def has_feet(self, polygon_vertices, threshold, num_bottom_points=10):
        if polygon_vertices is None or len(polygon_vertices) < num_bottom_points:
            return False

        try:
            # Sort vertices by y-coordinate (descending) to get the lowest points first
            sorted_indices = np.argsort(polygon_vertices[:, 1])[::-1]
            bottom_points = polygon_vertices[sorted_indices[:num_bottom_points]]

            # Check if the y-range of the lowest points exceeds the threshold
            min_y = np.min(bottom_points[:, 1])
            max_y = np.max(bottom_points[:, 1])

            if max_y - min_y > threshold:
                # Further check: Ensure the point with min_y isn't an outlier extreme point
                # compared to the average x of the bottom points (simple heuristic)
                avg_x = np.mean(bottom_points[:, 0])
                min_y_point_x = bottom_points[np.argmin(bottom_points[:, 1]), 0]
                # If the lowest point is horizontally far from the average, it might be noise
                # This threshold might need tuning
                if abs(min_y_point_x - avg_x) < (np.max(bottom_points[:, 0]) - np.min(bottom_points[:, 0])) * 0.75:
                    return True

        except Exception as e:
            print(f"Error in has_feet_from_vertices: {e}")
            return False

        return False
    
    def keep_largest_component(self, mask):
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Find the largest component
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Ignore background label 0
            mask = (labels == largest_label).astype(np.uint8)

        return mask
    
    def transform_top_point(self, point, H):
        point = np.array([point[0], point[1], 1], dtype=np.float32)

        transformed_point = np.dot(H, point)

        x_transformed = transformed_point[0] / transformed_point[2]
        y_transformed = transformed_point[1] / transformed_point[2]

        return(x_transformed, y_transformed)
    
    def transform(self, original_point, new_origin):
        # Translate to new origin
        point = original_point - new_origin

        # Mirroring across X, Y, and Z axes, change X and Z and recude dimension by deleting Y
        mirrored_point = np.array([-point[2], -point[0]])
        return mirrored_point
    
    def calculate_file_name(self, directory, basename, extension='.txt'):
        filename = basename + extension
        counter = 2
        while os.path.exists(os.path.join(directory, filename)):
            filename = f"{basename}{counter}{extension}"
            counter += 1
        return filename
    # END

    # COORDINATE MATCHING START
    def load_YOLO_masks(self, side, save_vis=False):
        origo = goose_lib.inverse_projection(self.right_points['primary_back_corner'],(1,self.params.dist_from_floor),self.params.OptimalIntrinsicMatrix, self.inv_rotation_matrices['right'], self.params.t) if side == 'right' else goose_lib.inverse_projection(self.left_points['secondary_back_corner'],(1,self.params.dist_from_floor),self.params.OptimalIntrinsicMatrix, self.inv_rotation_matrices['left'], self.params.t)
        pickle_file_path = self.file_paths_masks[side]
        
        text_file_path =os.path.join(self.project_dir, "coordinates", side +'.txt')
        self.text_file_paths_coords[side] = text_file_path
        
        # VIS
        vis_out = None
        colors = []
        if save_vis:
            vis_output_dir = os.path.join(self.project_dir, "output")
            os.makedirs(vis_output_dir, exist_ok=True)
            vis_output_path = os.path.join(vis_output_dir, f"{side}_masks_visualization.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = self.params.width
            frame_height = self.params.height
            vis_out = cv2.VideoWriter(vis_output_path, fourcc, 25, (frame_width, frame_height)) 
            colors = [(0, 0, 255),(0, 255, 0),(255, 0, 0),(255, 255, 0),
                               (0, 255, 255), (255, 0, 255), (138,138,255),(0, 21, 200),
                               (71, 223, 20), (138,13,25),(200, 21, 20), (0, 223, 0),
                               (0,100,100), (40, 40, 40), (120,120, 120), (120, 40, 80),
                               (80, 120, 40), (255, 255, 255)]
        # ENDVIS
        
        with open(text_file_path, "w") as file:
            pass
        
        try:
            with open(pickle_file_path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Pickle file not found at {pickle_file_path}")
            return
        except Exception as e:
            print(f"Error loading pickle file {pickle_file_path}: {e}")
            return
        
        for (seq_idx, dat) in data.items():
            frame_idx = dat['frame_idx']
            boxes = dat.get('boxes')
            masks = dat.get('masks')

            if masks is None: # skip
                continue
            
            image_height = self.params.height
            image_width = self.params.width

            # VIS
            frame_to_draw = None
            overlay = None
            alpha = 0.4 # Mask transparency
            if save_vis:
                try:
                    # Read the frame using decord reader (assuming it's initialized)
                    if self.video_readers[side] is None:
                         print("Error: Video reader not initialized.")
                         save_vis = False # Disable vis if reader is missing
                    else:
                        frame_tensor = self.video_readers[side][frame_idx]
                        # Convert tensor to numpy BGR
                        frame_to_draw = frame_tensor.asnumpy() # Assuming RGB output
                        frame_to_draw = cv2.cvtColor(frame_to_draw, cv2.COLOR_RGB2BGR)
                        overlay = frame_to_draw.copy() # For transparent overlay
                except IndexError:
                    print(f"Warning: Could not read frame {frame_idx} for visualization. Skipping vis for this frame.")
                    frame_to_draw = None # Skip drawing if frame read fails
                except Exception as e:
                    print(f"Error reading frame {frame_idx} for visualization: {e}")
                    frame_to_draw = None
            #END VIS
            
            for midx, mask_string in enumerate(masks):
                if mask_string is None: 
                    continue
                yolo_string = mask_string
                polygon_data = utils.get_polygon_properties_from_yolo(yolo_string, image_width, image_height)
                if polygon_data is None: continue
                
                polygon_vertices, mask_center, lowest_poly_point = polygon_data

                #vis
                if save_vis and frame_to_draw is not None:
                    color = colors[midx]
                    try:
                        # Draw contour
                        contours = [polygon_vertices.reshape((-1, 1, 2))]
                        if contours: # Only draw if contours are found
                            cv2.drawContours(frame_to_draw, contours, -1, color, 2) # Draw outline on main image
                            # Draw transparent fill on overlay
                            #overlay[mask == 255] = color

                            # Optional: Add index label near mask centroid
                            moments = cv2.moments(contours[0]) # Use the first/largest contour
                            if moments["m00"] != 0:
                                cX = int(moments["m10"] / moments["m00"])
                                cY = int(moments["m01"] / moments["m00"])
                                cv2.putText(frame_to_draw, str(midx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    except Exception as e:
                        print(f"Error drawing mask {midx} on frame {frame_idx}: {e}")

                if len(polygon_vertices) > 0:  # the mask is not empty
                    if side != 'top':
                        inv_rotation_mat = self.inv_rotation_matrices[side]
                        
                        """Ha talál olyan részt a maszkban, amely konkáv tehát lehet láb, akkor annak a résznek az alját veszi lábként.
                        Egyéb esetben a maszk közepétől a bounding box magasságának a felét veszi lefelé, és az lesz a láb pont.
                        """

                        if self.has_feet(polygon_vertices, 7):
                            possible_feet = lowest_poly_point
                        else:
                            x, y, w, h = boxes[midx] 
                            possible_feet = (int(mask_center[0]), int(mask_center[1] + h / 2))
                        
                        possible_feet = lowest_poly_point
                        possible_feet_np = np.array([[possible_feet]], dtype=np.float32)
                        possible_feet_und = goose_lib.undistortPoints(possible_feet_np, self.params.IntrinsicMatrix, self.params.distortionCoeff, input_size=self.params.input_size, cast_size=self.params.cast_size, output_size=self.params.input_size)
                        possible_feet_und = possible_feet_und[0][0]
                        mask_3d = self.transform(goose_lib.inverse_projection(possible_feet_und,(1,self.params.dist_from_floor),self.params.OptimalIntrinsicMatrix, inv_rotation_mat, self.params.t), origo)
                        
                        if save_vis: cv2.circle(frame_to_draw, possible_feet, 10, (255, 255, 255), -1)
                    else:
                        if save_vis: cv2.circle(frame_to_draw, mask_center, 10, (255, 255, 255), -1)
                        mask_3d = self.transform_top_point(mask_center, self.H)
                    
                    with open(text_file_path, "a") as file:
                        file.write(f"{seq_idx} {frame_idx} {midx} {mask_3d[0]} {mask_3d[1]}\n")
            
            if save_vis and frame_to_draw is not None and overlay is not None:
                try:
                    # Apply the overlay with transparency
                    cv2.addWeighted(overlay, alpha, frame_to_draw, 1 - alpha, 0, frame_to_draw)
                    vis_out.write(frame_to_draw)
                except Exception as e:
                    print(f"Error writing frame {frame_idx} to video: {e}")

    def load_YOLO_coords(self):
        all_data = {}
        for side, file_path in self.text_file_paths_coords.items():
            with open(file_path, 'r') as f:
                for line in f:
                    match = re.match(r"(\d+) (\d+) (\d+) ([+-]?\d+\.?\d*) ([+-]?\d+\.?\d*)", line)
                    if match:
                        i_str, index_str, midx_str, coord1_str, coord2_str = match.groups()
                        frame_idx, index, midx = map(int, [i_str, index_str, midx_str])
                        coordinate = (float(coord1_str), float(coord2_str))
                        if frame_idx not in all_data:
                            all_data[frame_idx] = {'right': [], 'left': [], 'top': []}
                        all_data[frame_idx][side].append(coordinate)

        return all_data
    
    def match_bird_coordinates_three_sets(self, set_A, set_B, set_C, max_distance=100.0):
        A = np.array(set_A, dtype=np.float32)
        B = np.array(set_B, dtype=np.float32)
        C = np.array(set_C, dtype=np.float32)

        if C.shape[0] == 0:
            return [], [], [], list(range(A.shape[0])), list(range(B.shape[0])), list(range(C.shape[0]))

        # Step 1: Match A to C
        if A.shape[0] != 0:
            cost_matrix_AC = cdist(A, C, metric='euclidean')
            row_ind_AC, col_ind_AC = linear_sum_assignment(cost_matrix_AC)
            matched_AC_initial = [(int(A_idx), int(C_idx)) for A_idx, C_idx in zip(row_ind_AC, col_ind_AC) if cost_matrix_AC[A_idx, C_idx] <= max_distance]
            used_C_A = {c_idx for _, c_idx in matched_AC_initial}
            potential_AC = set(matched_AC_initial)
        else:
            potential_AC = set()
            used_C_A = set()

        # Step 2: Match B to C
        if B.shape[0] != 0:
            cost_matrix_BC = cdist(B, C, metric='euclidean')
            row_ind_BC, col_ind_BC = linear_sum_assignment(cost_matrix_BC)
            matched_BC_initial = [(int(B_idx), int(C_idx)) for B_idx, C_idx in zip(row_ind_BC, col_ind_BC) if cost_matrix_BC[B_idx, C_idx] <= max_distance and C_idx not in used_C_A]
            used_C_B = {c_idx for _, c_idx in matched_BC_initial}
            potential_BC = set(matched_BC_initial)
        else:
            potential_BC = set()
            used_C_B = set()

        # Step 3: Match (A, C) and (B, C) pairs
        matched_triplets = []
        matched_A = set()
        matched_B = set()
        matched_C_final = set()

        # Prioritize matches where a C is involved in both A-C and B-C
        for a_idx, c_idx_ac in list(potential_AC):
            for b_idx, c_idx_bc in list(potential_BC):
                if c_idx_ac == c_idx_bc and c_idx_ac not in matched_C_final:
                    matched_triplets.append((a_idx, b_idx, c_idx_ac))
                    matched_A.add(a_idx)
                    matched_B.add(b_idx)
                    matched_C_final.add(c_idx_ac)
                    potential_AC.discard((a_idx, c_idx_ac))
                    potential_BC.discard((b_idx, c_idx_bc))
                    break # Move to the next A-C pair after finding a common C

        # Handle remaining A-C matches
        matched_pairs_AC_with_B = []
        remaining_AC = list(potential_AC)
        for a_idx, c_idx in remaining_AC:
            if a_idx not in matched_A and c_idx not in matched_C_final:
                # To form a triplet, we need to find an unmatched B close to this (A, C)
                min_dist_B = float('inf')
                best_b_idx = -1
                for b_idx in range(B.shape[0]):
                    if b_idx not in matched_B:
                        dist = cdist([A[a_idx]], [B[b_idx]], metric='euclidean')[0][0]
                        if dist <= max_distance and dist < min_dist_B:
                            min_dist_B = dist
                            best_b_idx = b_idx
                if best_b_idx != -1:
                    matched_triplets.append((a_idx, best_b_idx, c_idx))
                    matched_A.add(a_idx)
                    matched_B.add(best_b_idx)
                    matched_C_final.add(c_idx)
                    potential_AC.discard((a_idx, c_idx))

        # Handle remaining B-C matches
        matched_pairs_BC_with_A = []
        remaining_BC = list(potential_BC)
        for b_idx, c_idx in remaining_BC:
            if b_idx not in matched_B and c_idx not in matched_C_final:
                # To form a triplet, we need to find an unmatched A close to this (B, C)
                min_dist_A = float('inf')
                best_a_idx = -1
                for a_idx in range(A.shape[0]):
                    if a_idx not in matched_A:
                        dist = cdist([B[b_idx]], [A[a_idx]], metric='euclidean')[0][0]
                        if dist <= max_distance and dist < min_dist_A:
                            min_dist_A = dist
                            best_a_idx = a_idx
                if best_a_idx != -1:
                    matched_triplets.append((best_a_idx, b_idx, c_idx))
                    matched_A.add(best_a_idx)
                    matched_B.add(b_idx)
                    matched_C_final.add(c_idx)
                    potential_BC.discard((b_idx, c_idx))

        # Remaining unmatched A-C pairs
        unmatched_pairs_AC = [pair for pair in potential_AC if pair[0] not in matched_A and pair[1] not in matched_C_final]

        # Remaining unmatched B-C pairs
        unmatched_pairs_BC = [pair for pair in potential_BC if pair[0] not in matched_B and pair[1] not in matched_C_final]

        # Find truly unmatched points (not in any triplet or remaining pair)
        matched_A_total = matched_A.union({a for a, _ in unmatched_pairs_AC})
        matched_B_total = matched_B.union({b for b, _ in unmatched_pairs_BC})
        matched_C_total = matched_C_final.union({c for _, c in unmatched_pairs_AC}).union({c for _, c in unmatched_pairs_BC})

        unmatched_A_final = [i for i in range(A.shape[0]) if i not in matched_A_total]
        unmatched_B_final = [j for j in range(B.shape[0]) if j not in matched_B_total]
        unmatched_C_final = [k for k in range(C.shape[0]) if k not in matched_C_total]

        return matched_triplets, unmatched_pairs_AC, unmatched_pairs_BC, unmatched_A_final, unmatched_B_final, unmatched_C_final

    def cast_point_to_vis(self, point):
        return tuple(int(coord) for coord in point)

    def match_coordinates(self, all_data, save_vis=False):
        final_path = self.calculate_file_name(os.path.join(self.project_dir, "coordinates"), 'final')
        self.final_coords_file = os.path.join(self.project_dir, "coordinates", final_path)
        colors = [(0, 0, 255),(0, 255, 0),(255, 0, 0),(255, 255, 0),
          (0, 255, 255), (255, 0, 255),
          (138,138,255),(0, 21, 200),(71, 223, 20),
          (138,13,25),(200, 21, 20),
          (0, 223, 0),(0,100,100), (40, 40, 40), 
          (120,120, 120), (120, 40, 80), (80, 120, 40), (255, 255, 255)]
        
        with open(self.final_coords_file, "w") as f: 
            pass

        if save_vis: 
            os.makedirs(os.path.join(self.project_dir, "output"), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(self.project_dir, "output_matched.mp4"), fourcc, 25, (400, 400))

        for frame_idx, data in all_data.items():
            if save_vis: vis_all = np.zeros((400,400,3), np.uint8)
            set_A = data['right']
            set_B = data['left']
            set_C = data['top']
            matched_pairs, matched_AC, matched_BC,  unmatched_A, unmatched_B, unmatched_C = self.match_bird_coordinates_three_sets(set_A, set_B, set_C, 50.0)
            for i in range(len(matched_pairs)):
                match = matched_pairs[i]
                point_A = np.array(set_A[match[0]])
                point_B = np.array(set_B[match[1]])
                point_C = np.array(set_C[match[2]])
                
                weight_C = 3
                weighted_sum = point_A + point_B + weight_C * point_C
                total_weight = 1 + 1 + weight_C
                final_pos = weighted_sum / total_weight
                with open(self.final_coords_file, 'a') as f: f.write(f"{frame_idx} {match[2]} {final_pos[0]} {final_pos[1]}\n")

                if save_vis: 
                    cv2.drawMarker(vis_all, self.cast_point_to_vis(point_A), colors[i], cv2.MARKER_SQUARE, 5, 2)
                    cv2.drawMarker(vis_all, self.cast_point_to_vis(point_B), colors[i], cv2.MARKER_TRIANGLE_UP, 5, 2)
                    cv2.drawMarker(vis_all, self.cast_point_to_vis(point_C), colors[i], cv2.MARKER_DIAMOND, 5, 2)
                    cv2.circle(vis_all, self.cast_point_to_vis(final_pos) ,2, (255,255,255),-1)

            for i in range(len(matched_AC)):
                match = matched_AC[i]
                point_A = np.array(set_A[match[0]])
                point_C = np.array(set_C[match[1]])

                weight_C = 2
                weighted_sum = point_A + weight_C * point_C
                final_pos = weighted_sum / (1+weight_C)
                with open(self.final_coords_file, 'a') as f: f.write(f"{frame_idx} {match[1]} {final_pos[0]} {final_pos[1]}\n")

                if save_vis:
                    cv2.drawMarker(vis_all, self.cast_point_to_vis(point_A), colors[len(matched_pairs) + i], cv2.MARKER_SQUARE, 5, 2)
                    cv2.drawMarker(vis_all, self.cast_point_to_vis(point_C), colors[len(matched_pairs) + i], cv2.MARKER_DIAMOND, 5, 2)
                    cv2.circle(vis_all, self.cast_point_to_vis(final_pos) ,2, (255,255,255),-1)

            for i in range(len(matched_BC)):
                match = matched_BC[i]
                point_B = np.array(set_B[match[0]])
                point_C = np.array(set_C[match[1]])

                weight_C = 2
                weighted_sum = point_B + weight_C * point_C
                final_pos = weighted_sum / (1+weight_C)
                with open(self.final_coords_file, 'a') as f: f.write(f"{frame_idx} {match[1]} {final_pos[0]} {final_pos[1]}\n")

                if save_vis:
                    cv2.drawMarker(vis_all, self.cast_point_to_vis(point_B), colors[len(matched_pairs) + len(matched_AC) + i], cv2.MARKER_TRIANGLE_UP, 5, 2)
                    cv2.drawMarker(vis_all, self.cast_point_to_vis(point_C), colors[len(matched_pairs) + len(matched_AC) + i], cv2.MARKER_DIAMOND, 5, 2)
                    cv2.circle(vis_all, self.cast_point_to_vis(final_pos) ,2, (255,255,255),-1)

            if save_vis: out.write(vis_all)
        if save_vis: out.release()
    # END

    # SORT START
    def run_sort(self):
        sort_filaname = self.calculate_file_name(os.path.join(self.project_dir, "coordinates"), "sort")
        output_file_path = os.path.join(self.project_dir, "coordinates", sort_filaname)
        total_frames = 0
        self.sort_file = output_file_path
        if os.path.exists(self.final_coords_file):
            mot_tracker = Sort(max_age=30, 
                                min_hits=1,
                                iou_threshold=0) #create instance of the SORT tracker
            
            seq_dets = np.loadtxt(self.final_coords_file, delimiter=' ')

            with open(output_file_path,'w') as out_file:
                if seq_dets.ndim == 1:  # Handle case where there's only one detection
                        seq_dets = np.array([seq_dets])

                max_frame = int(seq_dets[:, 0].max()) if seq_dets.size > 0 else 0
                
                for frame in range(max_frame):
                    frame += 1 #detection and frame numbers begin at 1
                    dets = seq_dets[seq_dets[:, 0]==frame, 2:]
                    
                    #convert coordinates to [x1,y1,x2,y2]
                    bounding_boxes = np.zeros((dets.shape[0], 5))  # [x_top_left, y_top_left, width, height, score]
                    bounding_boxes[:, 0] = dets[:, 0] - 10
                    bounding_boxes[:, 1] = dets[:, 1] - 10
                    bounding_boxes[:, 2] = dets[:, 0] + 10
                    bounding_boxes[:, 3] = dets[:, 1] + 10
                    bounding_boxes[:, 4] = 1
                    
                    total_frames += 1
                    trackers = mot_tracker.update(bounding_boxes)

                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
    # END


    # PIPELINE START
    def load_videos(self):
        # masodik GPU-ra rakjuk a videokat
        self.video_readers['right'] = de.VideoReader(self.right_video_path,ctx=de.gpu(0))
        self.video_readers['left'] = de.VideoReader(self.left_video_path,ctx=de.gpu(0))
        self.video_readers['top'] = de.VideoReader(self.top_video_path,ctx=de.gpu(0))

    # -- used outside --
    def get_corner_points_all(self):
        yield "Creating backgrounds for the videos"
        corner_detector = CornerDetector(self.project_dir, self.right_video_path, self.left_video_path, self.top_video_path)
        self.right_background, self.left_background, self.top_background = corner_detector.get_all_backgrounds()
        # top
        yield "Getting top corner points"
        self.top_points, self.H = corner_detector.get_top_points()

        # side views
        yield "Getting side corner points"
        if corner_detector.align == 1:
            self.left_points = corner_detector.get_difficult_side_points()
            self.right_points = corner_detector.get_easy_side_points()
        else:
            self.left_points = corner_detector.get_easy_side_points()
            self.right_points = corner_detector.get_difficult_side_points()

    def run_syncronization(self):
        file_path = os.path.join(self.project_dir, "sync", "sync_indexes.json")
        if os.path.exists(file_path):
            new_sync_lists = {'right': [], 'left': [], 'top': []}

            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                new_sync_lists['right'] = data['right']
                new_sync_lists['left'] = data['left']
                new_sync_lists['top'] = data['top']
        else:
            synchronizer = sync()
            synchronizer.add_video(self.right_video_path)
            synchronizer.add_video(self.left_video_path)
            synchronizer.add_video(self.top_video_path)

            for x in range(3):
                synchronizer.sync_video(x)

            new_sync_lists = self.strip_zeros(synchronizer.synchronized_frames[0], synchronizer.synchronized_frames[1], synchronizer.synchronized_frames[2])

            new_sync_lists = {
                "right": new_sync_lists[0],
                "left": new_sync_lists[1],
                "top": new_sync_lists[2]
            }

            with open(file_path, "w") as f:
                json.dump(new_sync_lists, f)

        return new_sync_lists

    def calculate_side_projection(self, corner_points, side):
        file_dir = os.path.join(self.project_dir, "camera")
        file_path = os.path.join(file_dir, side + ".json")
        if not os.path.isdir(os.path.join(self.project_dir, "camera")): os.mkdir(file_dir)
        if not os.path.exists(file_path):
            primary_side = 1 if side == "right" else 2
            x = np.asarray([corner_points['secondary_back_corner'][0],corner_points['primary_back_corner'][0]])
            y = np.asarray([corner_points['secondary_back_corner'][1],corner_points['primary_back_corner'][1]])
            det = 1 # 1: egyenes, 2: parabola, 3: harmadfokú
            fitted_curve = np.polyfit(x,y,det)
            smooth_x = []
            smooth_y = []
            for i in range(0,self.params.width,1):
                smooth_x.append(i)
                pos_smooth_y = 0.0
                for j in range(len(fitted_curve)):
                    pos_smooth_y += np.power(i,det - j) * fitted_curve[j]
                smooth_y.append(pos_smooth_y)

            x_axis_middle_for_proj = int(self.params.OptimalIntrinsicMatrix[0][-1])
            right_back_corner = corner_points['primary_back_corner'] if side == 'right' else corner_points['secondary_back_corner']
            left_back_corner = corner_points['secondary_back_corner'] if side == 'right' else corner_points['primary_back_corner']
            angularRotation, projection_mat, rotation_mat, translation_vec , inv_rotation_mat, dist_from_floor = self.recalculate_orientation(x_axis_middle_for_proj, smooth_y, corner_points, left_back_corner, right_back_corner, self.params.dist_from_floor)
            
            data = {
                "angularRotation": angularRotation,
                "projection_mat": self.convert_to_serializable(projection_mat),
                "rotation_mat": self.convert_to_serializable(rotation_mat),
                "translation_vec": self.convert_to_serializable(translation_vec),
                "inv_rotation_mat": self.convert_to_serializable(inv_rotation_mat)
            }

            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)

            self.inv_rotation_matrices[side] = inv_rotation_mat
        else:
            with open(file_path, "r") as file:
                camera_params = json.load(file)
            camera_params = {key: self.convert_to_numpy(value) for key, value in camera_params.items()}
            self.inv_rotation_matrices[side] = camera_params['inv_rotation_mat']

    def run_YOLO(self, side, save_video=False, recalculate_data=True):
        # for visualization
        colors = [(0, 0, 255),(0, 255, 0),(255, 0, 0),(255, 255, 0),
          (0, 255, 255), (255, 0, 255),
          (138,138,255),(0, 21, 200),(71, 223, 20),
          (138,13,25),(200, 21, 20),
          (0, 223, 0),(0,100,100), (40, 40, 40), 
          (120,120, 120), (120, 40, 80), (80, 120, 40), (255, 255, 255)]
        
        # model
        device = torch.device("cuda:0")
        model = YOLO(self.yolo_paths[side]).to(device)  # Load a custom trained model

        base_pickle_file_path = os.path.join(self.project_dir, "yolo results", side + ".pkl")
        self.file_paths_masks[side] = base_pickle_file_path
        conf = 0.7 if side == 'top' else 0.5
        if save_video:
            os.makedirs(os.path.join(self.project_dir, "output"), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(self.project_dir, "output", side + ".mp4"), fourcc, 25, (self.params.width, self.params.height))
        
        if recalculate_data or not os.path.isfile(base_pickle_file_path):
            filename = self.calculate_file_name(os.path.join(self.project_dir, "yolo results"), side, '.pkl')
            pickle_file_path = os.path.join(self.project_dir, "yolo results", filename)
            self.file_paths_masks[side] = pickle_file_path
            results_for_saving = {}
            for i in range(len(self.sync_lists[side])):
                results_for_saving[i] = {}
                index = self.sync_lists[side][i]
                results_for_saving[i]['frame_idx'] = index
                myimg = self.video_readers[side][index].asnumpy()
                myimg_orig = myimg.copy()
                myimg = cv2.cvtColor(myimg, cv2.COLOR_RGB2BGR)

                
                results = model.track(source=myimg, tracker="botsort.yaml", verbose=False, persist=True, conf=conf, iou=0.2)
                
                result  = results[0]

                boxes   = result.boxes
                masks = result.masks
                if boxes is not None and masks is not None:
                    yoloout_h, yoloout_w = masks.shape[1], masks.shape[2]
                    plotterimg = myimg.copy()

                    r_w = yoloout_w / plotterimg.shape[1]
                    r_h = yoloout_h / plotterimg.shape[0]  
                    scale = min(r_w, r_h)
                    
                    new_w = int(plotterimg.shape[1] * scale) 
                    new_h = int(plotterimg.shape[0] * scale)

                    pad_w = (yoloout_w - new_w) // 2
                    pad_h = (yoloout_h - new_h) // 2

                    crop_y_start = max(0, pad_h)
                    crop_y_end = min(yoloout_h, pad_h + new_h)
                    crop_x_start = max(0, pad_w)
                    crop_x_end = min(yoloout_w, pad_w + new_w)

                    if boxes is not None:
                        results_for_saving[i]['boxes'] = []
                        for box_idx, box in enumerate(boxes):
                            b = np.floor(box.xywh[0].cpu().numpy())  # get box coordinates in (left, top, right, bottom) format
                            x,y,w,h = int(b[0]-b[2]//2),int(b[1]-b[3]//2),int(b[2]),int(b[3])
                            results_for_saving[i]['boxes'].append([x,y,w,h])
                            if save_video: 
                                plotterimg= cv2.rectangle(plotterimg, (x,y),(x+w, y+h), colors[box_idx], 1, cv2.LINE_AA)
                            
                    
                    if masks is not None:
                        myoverlay = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
                        results_for_saving[i]['masks'] = []
                        for midx, mask in enumerate(masks):
                            mask_o = mask
                            mask = mask.data[0].to(dtype=torch.float32).cpu().numpy()
                            mask = mask.astype(np.uint8)
                            mask = self.keep_largest_component(mask)
                            cropped_mask = mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                            resized_mask = cv2.resize(cropped_mask, (self.params.width, self.params.height), interpolation=cv2.INTER_NEAREST)
                            yolo_format_mask = utils.mask_to_yolo_format(resized_mask, 0, self.params.width, self.params.height)
                            results_for_saving[i]['masks'].append(yolo_format_mask)
                            r,c = np.where(mask!=0)
                            myoverlay[r,c,:] = colors[midx]
                    if save_video:
                        cropped_mask = myoverlay[pad_h:myoverlay.shape[0] - pad_h, pad_w:myoverlay.shape[1] - pad_w]  # 361 640
                        resized_mask = cv2.resize(cropped_mask, (plotterimg.shape[1], plotterimg.shape[0]), interpolation=cv2.INTER_NEAREST)     
                        plotterimg= cv2.addWeighted(resized_mask, 0.5,  plotterimg, 0.8, 1.0)

                    if save_video: out.write(plotterimg)

            if save_video: out.release()
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(results_for_saving, f)

    def save_zones_for_coords(self):
        zones_file_name_txt = self.calculate_file_name(os.path.join(self.project_dir, "coordinates"), "zones")
        zones_file_name_pkl = self.calculate_file_name(os.path.join(self.project_dir, "coordinates"), "zones", '.pkl')
        zones_file_path_txt = os.path.join(self.project_dir, "coordinates", zones_file_name_txt)
        zones_file_path_pkl = os.path.join(self.project_dir, "coordinates", zones_file_name_pkl)
        if os.path.exists(self.final_coords_file):
            zone_count = {}

            with open(self.final_coords_file, 'r') as f:
                with open(zones_file_path_txt, 'w') as f_zones:
                    for line in f:
                        data = line.strip().split(' ')
                        frame_idx = int(data[0])
                        if frame_idx not in zone_count: zone_count[frame_idx] = {'zone1': 0, 'zone2': 0, 'zone3': 0, 'zone4': 0}
                        x_coord = float(data[2])
                        
                        if 0 <= x_coord <= 135:
                            zone_count[frame_idx]['zone3']+= 1
                        elif 136 <= x_coord <= 235:
                            zone_count[frame_idx]['zone2']+= 1
                        elif x_coord >= 236:
                            zone_count[frame_idx]['zone1']+= 1
                        else:
                            zone_count[frame_idx]['zone4']+= 1
                        
                    
            for frame_idx, zone_data in zone_count.items():
                with open(zones_file_path_txt, 'a') as f_zones:
                    try:
                        if frame_idx % 750 == 0:
                            myimg = self.video_readers['top'][frame_idx].asnumpy()
                            img_name = f"{frame_idx}{zone_count[frame_idx]['zone1']}{zone_count[frame_idx]['zone2']}{zone_count[frame_idx]['zone3']}{zone_count[frame_idx]['zone4']}.jpg"
                            cv2.imwrite(os.path.join(self.project_dir, "output", img_name), cv2.cvtColor(myimg, cv2.COLOR_RGB2BGR))
                    except:
                        pass
                    zone_line = f"{frame_idx}, zone1: {zone_count[frame_idx]['zone1']}, zone2: {zone_count[frame_idx]['zone2']}, zone3: {zone_count[frame_idx]['zone3']}, zone4: {zone_count[frame_idx]['zone4']}\n"
                    f_zones.write(zone_line)
            with open(zones_file_path_pkl, 'wb') as f:
                pickle.dump(zone_count, f)
    # END
    
    def get_positions_pipeline(self):
        start_time = time.time()
        # load the three videos
        yield get_current_time() + "Loading videos..."
        self.load_videos()

        # create directory for sync files
        yoloresult_path = os.path.join(self.project_dir, "yolo results")
        os.makedirs(yoloresult_path, exist_ok=True)
        coordinate_path = os.path.join(self.project_dir, "sync")
        os.makedirs(coordinate_path, exist_ok=True)
        
        
        # synchronization
        yield  get_current_time() + "Synchronize files..."
        self.sync_lists = self.run_syncronization()
        
        # projection
        yield get_current_time() + "Calculate right side"
        self.calculate_side_projection(self.right_points, "right")
        yield get_current_time() + "Calculate left side"
        self.calculate_side_projection(self.left_points, "left")

        # create directory for txt files
        coordinate_path = os.path.join(self.project_dir, "coordinates")
        os.makedirs(coordinate_path, exist_ok=True)
        
        
        #YOLO: we go through every video after each other and saving the data
        for side in ['right', 'left', 'top']:
            yield get_current_time() + "YOLO detection for " + side + " side"
            self.run_YOLO(side, recalculate_data=True, save_video=True)

        for side in ['right', 'left', 'top']:
            yield get_current_time() + "Loading YOLO masks and calculating coords for " + side + " side"
            self.load_YOLO_masks(side, save_vis=False)

        # loading the coordinates
        yield get_current_time() + "Loading coords"
        data = self.load_YOLO_coords()
    
        # calculating the coordinates
        yield get_current_time() + "Matching the coordinates"
        self.match_coordinates(data)
        
        # SORT 
        yield get_current_time() + "Using sort algorithm on the coordinates"
        self.run_sort()

        yield get_current_time()+ "Saving zones and birds"
        self.save_zones_for_coords()
        end_time = time.time()
        print("positions_pipeline:: Estimated time: ", end_time-start_time)
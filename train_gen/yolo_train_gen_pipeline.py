import torch
import sys
import cv2

import numpy as np
import supervision as sv
import matplotlib.pyplot as plt

import os
import random
import time

import decord as de
import yaml

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import utils.utils as utils

def get_current_time():
    return time.strftime("%Y-%m-%d_%H-%M-%S") + " "

class EmptyIntervalError(ValueError):
    """Raised when the start and end indices are the same."""
    pass

class EndIndexOutOfBounds(ValueError):
    """Raised when end index is greater than the length of the video"""
    pass

class InvalidInterval(ValueError):
    """Raised when the start index is greater than the end index."""
    pass

class YOLODataGenerator():
    CHUNK_SIZE = 300

    def __init__(self, video_path, start, end, max_ruffs, project_dir):
        self.video_path = video_path
        self.start_idx = utils.get_frame_idx_from_time(start)
        self.end_idx = utils.get_frame_idx_from_time(end)
        video_reader = de.VideoReader(video_path)
        de.bridge.set_bridge("torch")
        if self.end_idx > len(video_reader):
            seconds = len(video_reader)//25
            minutes, secs = divmod(seconds, 60)
            hours, mins = divmod(minutes, 60)
            raise EndIndexOutOfBounds(f"There are less frames than the specified end index. Video length: {hours:02d}:{mins:02d}:{secs:02d}")
        del video_reader
        if self.start_idx == self.end_idx:
            raise EmptyIntervalError("Start and end indices are the same, no interval selected.")
        if self.start_idx > self.end_idx:
            raise InvalidInterval("Start index is greater than end index.")
        self.new_start_idx = self.start_idx
        self.new_end_idx = self.start_idx + self.CHUNK_SIZE if self.start_idx + self.CHUNK_SIZE < self.end_idx else self.end_idx
        self.project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)
        self.labels_dir = os.path.join(project_dir, 'labels')
        self.images_dir = os.path.join(project_dir, 'images')
        self.overlays_dir = os.path.join(project_dir, 'overlay')
        self.yaml_path = ""
        self.max_ruffs = max_ruffs
        self.object_labels = []
        self.random_indexes = []

        # Creating labels
        self.create_labels()

        # Import pretrained sam2 checkpoint and model
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_checkpoint = "train_gen/checkpoints/sam2.1_hiera_large.pt"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        self.inference_state = self.predictor.init_state(video_path=self.video_path, start_idx=self.new_start_idx, end_idx=self.new_end_idx)
        self.first_image = utils.numpy_tensor_image(self.inference_state['images'][0], video_width=self.inference_state['video_width'], video_height=self.inference_state['video_height'])

    def overlay_mask_on_image(self, image, mask, obj_id):
        binary_mask = np.squeeze(mask.astype(bool))

        if binary_mask.shape != image.shape[:2]:
            raise ValueError("Mask and image dimensions do not match!")

        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color_rgba = np.array([*cmap(cmap_idx)[:3], 0.6])

        color_rgb = (color_rgba[:3] * 255).astype(np.uint8)
        alpha = color_rgba[3]

        overlayed_image = image.copy()

        for c in range(3):  # For each RGB channel
            overlayed_image[binary_mask, c] = (
                alpha * color_rgb[c] + (1 - alpha) * overlayed_image[binary_mask, c]
            ).astype(np.uint8)

        return overlayed_image
    
    def create_labels(self):
        for i in range(self.max_ruffs):
            label = f"ruff{i+1}"
            self.object_labels.append(label)

    def create_yaml(self):
        data = {
            "path": self.project_dir,
            "train": f"{self.project_dir}/images/train",
            "val": f"{self.project_dir}/images/val",
            "nc": 1,
            "names": {
                0: 'ruff',
            },
            "single_class": True,
            "overlap_mask": False,
            "mask_ration": 1,
            "shear": 0.3,
            "flipud": 0.3,
            "mosaic": 0.5,
            "copy_paste": 0.5,
            "crop_fraction": 0.5,
        }
        yaml_string = yaml.dump(data, sort_keys=False)
        file_path = os.path.join(self.project_dir, "data.yaml")
        self.yaml_path = file_path

        with open(file_path, 'w') as f:
            f.write(yaml_string)

    def are_any_masks_overlapping(self, masks, threshold):
        n_masks = len(masks)
        if n_masks < 2:
            return False  # Need at least two masks to check for overlap

        for i in range(n_masks):
            for j in range(i + 1, n_masks):
                mask1 = masks[i]
                mask2 = masks[j]

                if mask1.shape != mask2.shape:
                    continue

                intersection = mask1 & mask2
                intersection_area = np.sum(intersection)
                area1 = np.sum(mask1)
                area2 = np.sum(mask2)

                if area1 > 0 and (intersection_area / area1 * 100) >= threshold:
                    return True 
                if area2 > 0 and (intersection_area / area2 * 100) >= threshold:
                    return True

        return False

    def sam2_create_train_data(self):
        wrong_frames = []
        while self.new_start_idx != self.end_idx - 1:
            print("Starting new cycle: ", get_current_time())
            relative_chunk_size = self.new_end_idx - self.new_start_idx
            # segmentation for CHUNK_SIZE frame
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                yield out_frame_idx

            # saving the yolo data
            print("Saving data: ", get_current_time())
            for out_frame_idx in range(0, len(self.inference_state['images'])):

                # if the number of masks equals the number of the monitored birds
                not_null_masks =  len([(mid, mask) for (mid, mask) in video_segments[out_frame_idx].items() if mask.any()])
                overlapping_masks =  self.are_any_masks_overlapping([mask for (mid, mask) in  video_segments[out_frame_idx].items()], 20)
                if not_null_masks == self.max_ruffs and not overlapping_masks:
                    #if the index is in the random_indexes, the frame goes to val, otherwise train
                    destination = ''
                    if (out_frame_idx + self.new_start_idx) in self.random_indexes:
                        destination = 'val'
                    else:
                        destination = 'train'
                    

                    base_name = os.path.basename(self.video_path)
                    video_name, _ = os.path.splitext(base_name)
                    text_filename = os.path.join(self.labels_dir, destination , f"{video_name}_0000{self.new_start_idx+out_frame_idx}.txt")
                    jpg_filename = os.path.join(self.images_dir, destination , f"{video_name}_0000{self.new_start_idx+out_frame_idx}.jpg")
                    overlay_filename = os.path.join(self.overlays_dir, destination , f"{video_name}_0000{self.new_start_idx+out_frame_idx}.jpg")

                    # for saving the overlay
                    overlay_picture = utils.numpy_tensor_image(self.inference_state['images'][out_frame_idx], video_height=self.inference_state['video_height'], video_width=self.inference_state['video_width'])
                    
                    #saving the mask data
                    with open(text_filename, 'w') as file:
                        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                            overlay_picture = self.overlay_mask_on_image(overlay_picture, out_mask, out_obj_id)

                            binary_mask = np.squeeze(out_mask).astype(np.uint8)

                            num_labels, labels = cv2.connectedComponents(binary_mask)

                            if num_labels <= 1:
                                continue  # No object

                            # Find largest component
                            largest_label = 1 + np.argmax([
                                np.sum(labels == i) for i in range(1, num_labels)
                            ])

                            out_mask = (labels == largest_label).astype(out_mask.dtype) * out_mask.max()


                            yolo_line = utils.mask_to_yolo_format(out_mask, 0, self.inference_state['video_width'], self.inference_state['video_height'])
                            if yolo_line:  # Ensure valid YOLO lines
                                file.write(yolo_line + '\n')
                    
                    # saving the raw picture
                    picture = utils.numpy_tensor_image(self.inference_state['images'][out_frame_idx], video_height=self.inference_state['video_height'], video_width=self.inference_state['video_width'])
                    bgr_picture = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(jpg_filename, bgr_picture)

                    # saving the overlayed picture
                    overlay_bgr_picture = cv2.cvtColor(overlay_picture, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(overlay_filename, overlay_bgr_picture)
                    print("van olyan, amit kimentünk egyáltaláN?")
                else:
                    wrong_frames.append(out_frame_idx)
                    
            if not video_segments:
                print(f"Warning: No segments generated for chunk ending at {self.new_end_idx}. Stopping propagation." + get_current_time())
                yield self.end_idx # Stop processing
                break
            
            # continue with the same prompting
            prompts = {}
            last_segments = video_segments[relative_chunk_size - 1].items()

            overlapping_masks_last =  self.are_any_masks_overlapping([mask for (mid, mask) in  last_segments], 50)
            # if we lost the birds, stop with the propagation
            if len([(mid, mask) for (mid, mask) in last_segments if mask.any()]) < self.max_ruffs or overlapping_masks_last:
                print("Stopping propagation. Masks count: ", len([(mid, mask) for (mid, mask) in last_segments if mask.any()]), "Overlapping masks: ", overlapping_masks_last , get_current_time())
                yield self.end_idx
                break;

            # calculating new start and end idx
            self.new_start_idx = self.new_end_idx - 1
            self.new_end_idx = self.new_start_idx + self.CHUNK_SIZE if self.new_start_idx + self.CHUNK_SIZE <= self.end_idx else self.end_idx

            if self.new_start_idx == self.end_idx - 1:
                print("This is the end: ", get_current_time())
                yield self.end_idx
                break

            # resetting the inference state for the next x frame
            print(f'reloading the pictures between {self.new_start_idx} and {self.new_end_idx}')
            self.predictor.reset_state_and_images(self.inference_state, self.new_start_idx, self.new_end_idx)
            print(f'Inference state images len: {len(self.inference_state["images"])}')

            torch.cuda.empty_cache()
            for object_id, label in enumerate(self.object_labels, start=1):
                masks = [mask for out_obj_id, mask in last_segments if out_obj_id == object_id]

                if len(masks) == 0:
                    continue

                points = np.array([
                    [
                        np.mean(np.argwhere(mask),axis=0)[1],
                        np.mean(np.argwhere(mask),axis=0)[0]
                    ] for mask in masks[0]
                ], dtype=np.float32)
                labels = np.ones(len(points))

                prompts[object_id] = points, labels

                _, object_ids, mask_logits = self.predictor.add_new_points(
                    inference_state=self.inference_state,
                    frame_idx=0,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )


            print(f'New end idx: {self.new_end_idx}, end idx: {self.end_idx}', get_current_time())
            print(prompts)

        del self.inference_state
        del self.predictor
        torch.cuda.empty_cache()

    def create_train_data_pipeline(self, points_with_labels):
        # GPU context
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Creating directories
            try:
                os.makedirs(self.labels_dir, exist_ok=True)
                os.makedirs(os.path.join(self.labels_dir, "train"), exist_ok=True)
                os.makedirs(os.path.join(self.labels_dir, "val"), exist_ok=True)
                
                os.makedirs(self.images_dir, exist_ok=True)
                os.makedirs(os.path.join(self.images_dir, "train"), exist_ok=True)
                os.makedirs(os.path.join(self.images_dir, "val"), exist_ok=True)
                
                os.makedirs(self.overlays_dir, exist_ok=True)
                os.makedirs(os.path.join(self.overlays_dir, "train"), exist_ok=True)
                os.makedirs(os.path.join(self.overlays_dir, "val"), exist_ok=True)

            except OSError as e:
                print(f"Error creating directories: {e}")

            # creating random indexes for validating the model
            interval = self.end_idx - self.start_idx
            for i in range(interval//10):
                while(True):
                    random_number = random.randint(self.start_idx, self.end_idx)
                    if random_number not in self.random_indexes:
                        self.random_indexes.append(random_number)
                        break;
            
            for index, label in enumerate(points_with_labels):
                points = points_with_labels[label]
                points = np.array(points, dtype=np.float32)
                labels = np.ones(len(points), dtype=np.int32)
                _, object_ids, mask_logits = self.predictor.add_new_points(
                    inference_state=self.inference_state,
                    frame_idx=0,
                    obj_id=index,
                    points=points,
                    labels=labels,
                )
            
            # SAM2 pipeline
            for completed_frame in self.sam2_create_train_data():
                if completed_frame == self.end_idx:
                    yield completed_frame
                else: 
                    yield self.new_start_idx + completed_frame + 1
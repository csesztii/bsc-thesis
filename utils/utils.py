import numpy as np
import cv2
import datetime
import time
import re
from PIL import Image
import torch

class IntrinsicParameters:
    IntrinsicMatrix = np.asarray([[1195.91324,0.0,1344.0],[0.0,1195.93626,760.0],[0.0,0.0,1.0]])
    distortionCoeff = np.asarray([-0.29394954,0.08916018,-0.0019039,-0.00149538,-0.01191335])
    distortionCoeffZeros = np.asarray([0.0,0.0,0.0,0.0,0.0])
    t = (0.0,0.0,0.0)
    angular_rotation = (0.0,0.0,0.0)
    dist_from_floor = 45
    opposite_line_dist = 275.0
    width, height = 2688,1520
    input_size = (width, height)
    cast_size = (int(width*0.75), int(height*0.75))
    OptimalIntrinsicMatrix, _ = cv2.getOptimalNewCameraMatrix(IntrinsicMatrix, distortionCoeff, input_size, 0.0, cast_size)
    OptimalIntrinsicMatrix[0,2] = width / 2
    OptimalIntrinsicMatrix[1,2] = height / 2

def mask_to_yolo_format(binary_mask, class_index, image_width, image_height):
        # (squeeze to 2D)
        binary_mask_2d = binary_mask.squeeze()

        # boolean mask to uint8
        binary_mask_uint8 = (binary_mask_2d * 255).astype(np.uint8)

        # find contours
        contours, hierarchy = cv2.findContours(binary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None  # No contours found
        
        largest_contour = max(contours, key=cv2.contourArea)

        # normalize coordinates
        normalized_coordinates = []
        for point in largest_contour:
            x, y = point[0]  # Extract x, y coordinates
            norm_x = x / image_width
            norm_y = y / image_height
            normalized_coordinates.extend([norm_x, norm_y])

        # format: <class-index> <x1> <y1> <x2> <y2> ...
        yolo_format_line = f"{class_index} " + " ".join(f"{coord:.6f}" for coord in normalized_coordinates)
        
        return yolo_format_line

def yolo_polygon_to_numpy_mask(yolo_string, image_width, image_height):
    """
    Converts a YOLO polygon format string to a NumPy binary mask.

    Args:
        yolo_string (str): The YOLO polygon string (e.g., "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5").
                           It starts with the class index followed by normalized x, y coordinates.
        image_width (int): The width of the original image the polygon belongs to.
        image_height (int): The height of the original image the polygon belongs to.

    Returns:
        np.ndarray: A binary NumPy mask (height, width) with dtype uint8,
                    where the polygon area is 255 and the background is 0.
                    Returns None if the input string is invalid or has too few points.
    """
    parts = yolo_string.strip().split()

    if len(parts) < 7 or len(parts) % 2 == 0:
        return None

    coords_normalized = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)

    coords_pixel = (coords_normalized * np.array([image_width, image_height])).astype(np.int32)

    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    cv2.fillPoly(mask, [coords_pixel.reshape((-1, 1, 2))], color=255) # Fill with white

    return mask

def get_polygon_properties_from_yolo(yolo_string, image_width, image_height):
    """
    Parses YOLO polygon string, calculates vertices, centroid, and lowest point
    in pixel coordinates.

    Args:
        yolo_string (str): Space-separated normalized coordinates (x1 y1 x2 y2...).
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: (vertices, centroid, lowest_point) or None if parsing fails.
               vertices: Nx2 numpy array of pixel coordinates [[x1, y1], ...].
               centroid: (cx, cy) tuple of integers.
               lowest_point: (lx, ly) tuple of integers (point with max y).
    """
    try:
        points = yolo_string.strip().split()
        if len(points) < 3:
            # print(f"Warning: Not enough points in YOLO string: {yolo_string[:50]}...")
            return None # Need at least 3 points for a polygon

        vertices = np.array([float(p) for p in points[1:]]).reshape(-1, 2)
        # Denormalize
        vertices[:, 0] *= image_width
        vertices[:, 1] *= image_height
        vertices = vertices.astype(int) # Convert to integer pixel coordinates

        # Calculate centroid using moments for better accuracy with irregular polygons
        moments = cv2.moments(vertices)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            centroid = (centroid_x, centroid_y)
        else:
            # Fallback to mean if moments are zero (e.g., collinear points)
            centroid = tuple(np.mean(vertices, axis=0).astype(int))


        # Find lowest point (vertex with max y-coordinate)
        lowest_idx = np.argmax(vertices[:, 1])
        lowest_point = tuple(vertices[lowest_idx])

        return vertices, centroid, lowest_point
    except Exception as e:
        print(f"Error parsing YOLO string '{yolo_string[:50]}...': {e}")
        return None

def get_frame_idx_from_time(video_time, fps=25):
    pattern = r"^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)$"
    if not re.match(pattern, video_time):
        raise ValueError(f"Invalid time format: {video_time}. Expected HH:MM:SS")
    x = time.strptime(video_time,'%H:%M:%S')
    seconds = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    return int(seconds * fps)

def numpy_tensor_image(tensor_image, video_width=1920, video_height=1080, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)):
    tensor_image = tensor_image.cpu() * torch.tensor(img_std).view(3, 1, 1) + torch.tensor(img_mean).view(3, 1, 1)
    tensor_image = tensor_image.clamp(0, 1)
    np_image = tensor_image.permute(1, 2, 0).numpy()
    pil_image = Image.fromarray((np_image * 255).astype('uint8'))
    pil_image = pil_image.resize((video_width, video_height), Image.LANCZOS)
    np_resized_image = np.array(pil_image)
    return np_resized_image
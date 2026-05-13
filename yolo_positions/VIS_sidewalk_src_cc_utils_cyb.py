import numpy as np
import cv2 as cv
#==========================================
# Calibration
#==========================================
def undistort(img, mat_intri, coff_dis, input_size = (1920,1080),  cast_size = (1920,1080), output_size = (1920,1080), output_camera_matrix = False):
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mat_intri, coff_dis, input_size, 0.0, cast_size)
    newcameramtx[0][2] *= (output_size[0]/cast_size[0]) # skálázza az optikai középpontot az új képmérethez
    newcameramtx[1][2] *= (output_size[1]/cast_size[1])
    img = cv.resize(img, output_size)                   # skálázza a képet az új képmérethez
    udst = cv.undistort(img, mat_intri, coff_dis, None, newcameramtx)
    if output_camera_matrix == True:
        return udst, newcameramtx
    return udst

def undistortPoints(points_to_undistort, mat_intri, coff_dis, input_size = (1920,1080),  cast_size = (1920,1080), output_size = (1920,1080), output_camera_matrix = False):
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mat_intri, coff_dis, input_size, 0.0, cast_size)
    newcameramtx[0][2] *= (output_size[0]/cast_size[0])
    newcameramtx[1][2] *= (output_size[1]/cast_size[1])
    udst = cv.undistortPoints(points_to_undistort, mat_intri, coff_dis, None, newcameramtx)
    if output_camera_matrix == True:
        return udst, newcameramtx
    return udst
#==========================================
# Projection
#==========================================
def calculate_extrinsic_matrix(rot:tuple,pos:tuple) -> np.ndarray:
    rotation_matrix = cv.Rodrigues(rot)[0]
    inv_rotation_mat = np.linalg.inv(rotation_matrix)
    t = np.asarray([pos[0],pos[1],pos[2]])
    extrinsic_matrix = np.hstack([rotation_matrix,np.asarray([0.0,0.0,0.0]).reshape(3,1)])
    return extrinsic_matrix, rotation_matrix, t, inv_rotation_mat

def calculate_t(pos:tuple) -> np.ndarray:
    t = np.asarray([pos[0],pos[1],pos[2]])
    return t

def projection_func(pos, camera_mat, rotation_mat, translation_mat, dist):
    rp_pos, _ = cv.projectPoints(pos,rotation_mat,translation_mat,camera_mat,dist)
    rp_pos = rp_pos.squeeze()
    return (int(rp_pos[0]),int(rp_pos[1]))
def calculate_rotation_matrix(rot:tuple):
    return cv.Rodrigues(rot)[0], np.linalg.inv(cv.Rodrigues(rot)[0])

def inverse_projection(uv : tuple, constraint : tuple, ks : np.ndarray, ir : np.ndarray, t : np.ndarray) -> tuple:
    ps = (((uv[0] - ks[0,2]) / ks[0,0]),((uv[1] - ks[1,2]) / ks[1,1]),1.0) # the homogeneous ray intersecting the image plane
    irp = np.matmul(ir,ps) # multiplied by the inverse-rotation matrix

    yirp = constraint[1] / irp[constraint[0]]
    pirp = np.asarray([irp[0] * yirp,irp[1] * yirp,irp[2] * yirp]) # constrained to a known plane (y=80.0)
    tirp = pirp + t # translated to the camera's world-position
    return tirp
def distorted_inverse_projection(uv : tuple, constraint : tuple,ks : np.ndarray, ir : np.ndarray, t : np.ndarray,dist : np.ndarray):
    undistorted_point, new_camera_matrix = undistortPoints(uv,ks,dist,output_camera_matrix=True)
    undistorted_point=undistorted_point[0][0]
    inverse_projected_coord = inverse_projection(undistorted_point,constraint,new_camera_matrix,ir,t)
    return inverse_projected_coord
import streamlit as st
import os
import cv2
import traceback
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from yolo_positions_pipeline import PositionDetecor

def draw_corner_points_on_image(image: np.ndarray, points, color=(0, 0, 255), radius=20, thickness=-1):
    num_points = len(points)
    cmap = plt.get_cmap('hsv')
    
    image_copy = image.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
    for i, (key, point) in enumerate(points.items()):
        color = cmap(i / num_points)
        color = tuple(int(c * 255) for c in color[:3])
        cv2.circle(image_copy, point, radius, color, thickness)
    return image_copy

def show_point_editor(image, points_key):
    points = st.session_state[points_key]
    cols = st.columns(len(points))
    height, width, _ = image.shape
    new_points = {}
    for index, (name, point) in enumerate(points.items()):
        with cols[index]:
            st.write(f"**{name}**")
            x = st.number_input("X", value=point[0], key=f"x_{index}", label_visibility="collapsed" ,min_value=0, max_value=width)
            y = st.number_input("Y", value=point[1], key=f"y_{index}", label_visibility="collapsed" ,min_value=0, max_value=height)
            
            new_points[name] = [x,y]
    
    st.session_state[points_key] = new_points
    vis_img = draw_corner_points_on_image(image, st.session_state[points_key])
    st.image(Image.fromarray(vis_img), use_container_width=True)


def main():
    st.session_state.setdefault("process_button_clicked", False)
    st.session_state.setdefault("edit_right_points", False)
    st.session_state.setdefault("edit_left_points", False)
    st.session_state.setdefault("edit_top_points", False)
    st.session_state.setdefault("editing_started", False)
    st.session_state.setdefault("start_pipeline", False)
    
    st.header("YOLO bird tracking")

    st.markdown(
        """
        <style>
            .stApp {
                padding-left: 10px !important; /* Adjust the left padding */
                padding-right: 10px !important; /* Adjust the right padding */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Video Uploads")
        right_video = st.text_input("Enter right side video path", "")
        left_video = st.text_input("Enter left side video path", "")
        top_video = st.text_input("Enter top view video path", "")
        project_dir = st.text_input("Enter Project Directory", os.path.join(os.getcwd(), "tests", "project_dir_for_lek3"))
        #yolo_side_model_dir = st.text_input("Enter YOLO side model path", os.path.join(os.getcwd(), "models", "top.pt"))
        yolo_side_model_dir = st.text_input("Enter YOLO side model path", "/home/molnarester/sam2/yolo_training/side_training/train/weights/best.pt")
        #yolo_top_model_dir = st.text_input("Enter YOLO top model path", os.path.join(os.getcwd(), "models", "side.pt"))
        yolo_top_model_dir = st.text_input("Enter YOLO top model path", "/home/molnarester/sam2/yolo_training/train2/weights/best.pt")

        if st.button("Process Videos"):
            try:
                if left_video == "" or right_video == "" or top_video == "":
                    raise Exception("Please add paths from all views")
                
                if 'position_det' not in st.session_state:
                    if right_video is not None and left_video is not None and top_video is not None:
                        #st.session_state.position_det = PositionDetecor("/home/molnarester/project_ruff/snippets/right_1730_1830_short.mp4", "/home/molnarester/project_ruff/snippets/left_1730_1830_short.mp4", "/home/molnarester/project_ruff/snippets/top_1730_1830_short.mp4", project_dir, yolo_side_model_dir, yolo_top_model_dir)
                        st.session_state.position_det = PositionDetecor(right_video, left_video, top_video, project_dir, yolo_side_model_dir, yolo_top_model_dir)
                        st.session_state.process_button_clicked = True
                    else:
                        st.error("Please add paths from all views")

            except Exception as e:
                st.error(f"An error occured: {e}")
                traceback.print_exc()

    if st.session_state.process_button_clicked and not st.session_state.start_pipeline and hasattr(st.session_state, 'position_det'):
        for message in st.session_state.position_det.get_corner_points_all(): print(message)
        found_all_corner_points = True and st.session_state.position_det.right_points is not None and st.session_state.position_det.left_points is not None and st.session_state.position_det.top_points is not None

        vis_img_right = cv2.imread(os.path.join(project_dir, 'backgrounds', 'right_undistorted.png'))
        vis_img_left = cv2.imread(os.path.join(project_dir, 'backgrounds', 'left_undistorted.png'))
        vis_img_top = cv2.imread(os.path.join(project_dir, 'backgrounds', 'top_undistorted.png'))
        
        if st.session_state.editing_started or found_all_corner_points: 
            st.button("Submit Points", use_container_width=True, on_click=lambda: setattr(st.session_state, 'start_pipeline', True))
            st.write("If you finished editing the points, please submit for further processing.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Right side")
            if st.session_state.position_det.right_points is None:
                st.write("We couldn't detect the corner points. Please add the corner points manually.")
                button_label = "Add points"
                if not st.session_state.editing_started: st.session_state.right_points = {'primary_front_corner': [2000,1500], 'primary_back_corner': [2000,500], 'secondary_back_corner': [400,500]}
            else:
                vis_img = draw_corner_points_on_image(vis_img_right, st.session_state.position_det.right_points)
                st.image(Image.fromarray(vis_img), use_container_width=True)
                if not st.session_state.editing_started: st.session_state.right_points = st.session_state.position_det.right_points
                

                button_label = "Edit points" 
            
            
            if st.button(button_label, key="right"):
                st.session_state.edit_right_points = True
                st.session_state.edit_left_points = False
                st.session_state.edit_top_points = False
                st.session_state.editing_started = True

        with col2:
            st.subheader("Left side")
            if st.session_state.position_det.left_points is None:
                st.write("We couldn't detect the corner points. Please add the corner points manually.")
                button_label = "Add points"
                if not st.session_state.editing_started: st.session_state.left_points = {'primary_front_corner': [400,1500], 'primary_back_corner': [400,500], 'secondary_back_corner': [2000,500]}
            else:
                vis_img = draw_corner_points_on_image(vis_img_left, st.session_state.position_det.left_points)
                st.image(Image.fromarray(vis_img), use_container_width=True)
                if not st.session_state.editing_started: st.session_state.left_points = st.session_state.position_det.left_points

                button_label = "Edit points"

            found_all_corner_points = found_all_corner_points and st.session_state.position_det.left_points is not None
            if st.button(button_label, key="left"):
                st.session_state.edit_left_points = True
                st.session_state.edit_right_points = False
                st.session_state.edit_top_points = False
                st.session_state.editing_started = True
        
        with col3:
            st.subheader("Top")
            if st.session_state.position_det.top_points is None:
                st.write("We couldn't detect the corner points. Please add the corner points manually.")
                button_label = "Add points"
                if not st.session_state.editing_started: st.session_state.top_points = {'top_left': [400,400], 'top_right': [2000,400], 'bottom_right': [2000,1200], 'bottom_left': [400,1200]}
            else:
                vis_img = draw_corner_points_on_image(vis_img_top, st.session_state.position_det.top_points)
                st.image(Image.fromarray(vis_img), use_container_width=True)
                if not st.session_state.editing_started: st.session_state.top_points = st.session_state.position_det.top_points

                button_label = "Edit points"

            if st.button(button_label, key="top"):
                st.session_state.edit_top_points = True
                st.session_state.edit_left_points = False
                st.session_state.edit_right_points = False
                st.session_state.editing_started = True

        if st.session_state.edit_right_points:
            show_point_editor(vis_img_right, 'right_points')
        if st.session_state.edit_left_points:
            show_point_editor(vis_img_left, 'left_points')
        if st.session_state.edit_top_points:
            show_point_editor(vis_img_top, 'top_points')
    else:
        st.write("Please input the data in the sidebar and then proceed by clicking the 'Process Videos' button.")

    if st.session_state.start_pipeline:
        st.session_state.position_det.right_points = st.session_state.right_points
        st.session_state.position_det.left_points = st.session_state.left_points
        st.session_state.position_det.top_points = st.session_state.top_points

        with st.status("Run detection...", expanded=True):
            for task in st.session_state.position_det.get_positions_pipeline():
                st.write(task)

if __name__ == "__main__":
    main()
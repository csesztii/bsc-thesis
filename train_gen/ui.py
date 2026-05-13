import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st
import os

from streamlit_image_coordinates import streamlit_image_coordinates as img_cord
from yolo_train_gen_pipeline import YOLODataGenerator


def clear_session_state():
    """Clears all keys from Streamlit's session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.setdefault("done", False)
    st.session_state.setdefault("generator", None)
    st.session_state.setdefault("object_labels", [])
    st.session_state.setdefault("generation_started", False)
    st.session_state.setdefault("create_yaml", False)


def draw_points_on_image(image: np.ndarray, points: np.ndarray, color=(0, 0, 255), radius=10, thickness=-1):
    num_points = len(points)
    cmap = plt.get_cmap('hsv')
    
    image_copy = image.copy()
    for i, label in enumerate(points):
        for point in points[label]:
            color = cmap(i / num_points)
            color = tuple(int(c * 255) for c in color[:3])
            cv2.circle(image_copy, tuple(point), radius, color, thickness)
    return image_copy

def main():
    st.set_page_config(layout="wide") #important for columns
    st.markdown("""
        <style>
        .css-1y4p82d {
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    st.session_state.setdefault("done", False)
    st.session_state.setdefault("generator", None)
    st.session_state.setdefault("object_labels", [])
    st.session_state.setdefault("generation_started", False)
    st.session_state.setdefault("create_yaml", False)
    
    with col1:
        st.subheader("YOLO Data Generator Interface")
        
        if st.button("Reset All"):
            clear_session_state()
            # Force a rerun of the app to reflect the cleared state
            st.rerun()

        # File Selection
        video_file = st.text_input("Video file path", "/home/molnarester/project_ruff/snippets/Lek3_left_1700_1900.mp4")

        # Time Interval Input
        start_time = st.text_input("Start Time (HH:MM:SS)", "00:00:00")
        end_time = st.text_input("End Time (HH:MM:SS)", "00:00:30")

        # Number Input
        max_ruffs = st.number_input("Max number of ruffs:", min_value=1, value=7, max_value=7)

        # Project Directory Selection
        project_dir = st.text_input("Project Directory", os.getcwd() + "/project_dir")

        if not st.session_state.generation_started:
            done = st.button("OK")
            st.session_state.done = done

        if st.session_state.done and not st.session_state.generation_started:
            try:
                # Create the generator instance
                generator = YOLODataGenerator(video_file, start_time, end_time, max_ruffs, project_dir)

                st.session_state.generator = generator
                st.session_state.object_labels = generator.object_labels

                if not hasattr(st.session_state, 'points'):
                    st.session_state.points = {}
                    for i, label in enumerate(st.session_state.object_labels):
                        st.session_state.points[label] = [(i*100, i*100)]
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")

    
    with col2:
        if video_file is None:
            st.subheader("First Image from Video")
            st.write("Upload a video to see the first frame.")
        else:
            if not st.button("Generate Data") and not st.session_state.generation_started and st.session_state.generator is not None:
                if hasattr(st.session_state, 'generator') and hasattr(st.session_state.generator, 'first_image') and st.session_state.generator.first_image is not None:
                    create_yaml = st.checkbox("Create YAML file for training?")
                    
                    if 'create_yaml' not in st.session_state:
                        st.session_state.create_yaml = {}
                    st.session_state.create_yaml = create_yaml
                    
                    st.write("Click on the picture to select the ruffs")
                    st.session_state.selected_label = st.selectbox("Select label", st.session_state.object_labels)
                    
                    for (x,y) in st.session_state.points[st.session_state.selected_label]:
                        st.write(f"X: {x}  |  Y: {y}")
                                
                                
                    vis_img = draw_points_on_image(st.session_state.generator.first_image, st.session_state.points)
                    value = img_cord(Image.fromarray(vis_img), key=f"img_cord_{st.session_state.selected_label}", use_column_width="always")

                    if value is not None:
                        x, y = value['x'], value['y']
                        width = Image.fromarray(st.session_state.generator.first_image).width
                        height = Image.fromarray(st.session_state.generator.first_image).height

                        scale_x = width / value["width"]
                        scale_y = height / value['height']
                        x,y = int(x*scale_x), int(y*scale_y)
                        
                        if (x, y) not in st.session_state.points[st.session_state.selected_label]:
                            st.session_state.points[st.session_state.selected_label] = [(x, y)]
                            st.rerun()

                else: 
                    st.write("Click 'OK' to see the preview")
            else:
                if hasattr(st.session_state, 'generator') and hasattr(st.session_state.generator, 'first_image') and st.session_state.generator.first_image is not None:
                    try:
                        st.success("Data generation started.")
                        st.session_state.generation_started = True
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        
                        for completed_frame in st.session_state.generator.create_train_data_pipeline(st.session_state.points):
                            progress = int((completed_frame / st.session_state.generator.end_idx) * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Completed {completed_frame} / {st.session_state.generator.end_idx} frames ({progress}%)")

                        if st.session_state.create_yaml:
                            st.session_state.generator.create_yaml()

                        st.markdown(f"""
                            **Training data created in the following directories:**
                            * Labels: `{st.session_state.generator.labels_dir}`
                            * Images: `{st.session_state.generator.images_dir}`
                            * Overlays: `{st.session_state.generator.overlays_dir}`
                            """)
                        
                        if st.session_state.create_yaml :st.markdown(f"* **Yaml created:** `{st.session_state.generator.yaml_path}`")
                        
                    except Exception as e:
                        import traceback
                        st.error(f"An error occurred: {e}")
                        st.error(f"Error type: {type(e)}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                else: 
                    st.write("Click 'OK' to see the preview and start generating data")

if __name__ == "__main__":
    main()
import streamlit as st
import time

from yolo_train_pipeline import YOLOTrainer

class StreamlitTextRedirector:
    def __init__(self, text_area):
        self.text_area = text_area

    def write(self, message):
        self.text_area.write(message)

    def flush(self):
        pass

def main():
    st.title("YOLO Model Trainer")

    yaml_path = st.text_input("YAML Path", "/home/molnarester/sam2/yolo_training/data.yaml")
    project_dir = st.text_input("Project Directory", "/home/molnarester/sam2/yolo_training")
    yolo_model = st.selectbox("YOLO Model", ["yolov9c-seg.pt", "yolov9e-seg.pt"])

    st.markdown("You can find more information about YOLO segmentation models on the following site: [YOLOv9 Segmentation Models](https://docs.ultralytics.com/models/yolov9/#__tabbed_1_2)")
    
    if st.button("Start Training"):
        try:
            trainer = YOLOTrainer(yaml_path, project_dir, yolo_model)
            results = trainer.run_train()

            if results is None: 
                st.error("Unsuccessful training.")
            else:
                success_message = str(time.time()) + " Training completed successfully!"
                st.success(success_message)
        
        except Exception as e:
            st.error(f"Error initializing trainer: {e}")

if __name__ == "__main__":
    main()
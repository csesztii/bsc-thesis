from ultralytics import YOLO
import logging
import sys
import os

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message.strip())

    def flush(self):
        pass

class YOLOTrainer:
    def __init__(self, yaml_path, project_dir, yolo_model):
        self.yaml_path = yaml_path
        self.project_dir = project_dir
        os.makedirs(self.project_dir, exist_ok=True)
        self.model = YOLO(yolo_model) # yolov9c-seg.pt or yolov9e-seg
        self.error_txt_path = os.path.join(project_dir, 'errors.txt')
        self.output_txt_path = os.path.join(project_dir, 'output.txt')

        self.logger = logging.getLogger('YOLOTrainerLogger')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        formatter = logging.Formatter('%(asctime)s - %(message)s')

        output_handler = logging.FileHandler(self.output_txt_path, mode='a')
        output_handler.setLevel(logging.INFO)
        output_handler.setFormatter(formatter)
        output_handler.addFilter(lambda record: record.levelno == logging.INFO)
        self.logger.addHandler(output_handler)

        error_handler = logging.FileHandler(self.error_txt_path, mode='a')
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)

        sys.stdout = LoggerWriter(self.logger.info)
        sys.stderr = LoggerWriter(self.logger.warning)

    def run_train(self):
        self.logger.info(f"Starting training with model {self.model.ckpt_path}, data {self.yaml_path}")
        try:
            results = self.model.train(data=self.yaml_path, 
                    amp=False,
                    project=self.project_dir,
                    pretrained=True, 
                    epochs=100, 
                    save_period=5, 
                    device="cuda:1")
            self.logger.info("Training completed successfully.")
            return results
        except ValueError as e:
            self.logger.error(f"ValueError during training: {e}")
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during training: {e}")
        return None
import numpy as np
import cv2 as cv
import decord as de
from tqdm import tqdm
import pickle
import os
class Synchronizer:
    def __init__(self):
        self.video_paths = []
        self.videos = []
        self.fractional_timestamps = []
        self.current_frame = 0
        self.synchronized_frames = []
        self.summed_diff = 0.0
        self.diff_cnt = 0
        self.min_stamp = 10000000.0
        self.max_stamp = -10000000.0
        self.time_templates = []
        for t in range(10):
            img_path = os.getcwd() + "/yolo_positions/ruff_sync_demo_by_Gergo/stamp_templates/new_templates/stamp_"+str(t)+".png"
            template = cv.imread(img_path)
            template = cv.cvtColor(template,cv.COLOR_BGR2GRAY)
            self.time_templates.append(template)
    
    def what_digit(self,extract,time_templates):
        probabilities = []
        for c in range(10):
            res = cv.matchTemplate(extract,time_templates[c],cv.TM_CCOEFF_NORMED)
            probabilities.append(np.amax(res))
        return np.argmax(np.asarray(probabilities))
    
    def get_frame_timestamps(self,input_frame : np.ndarray,time_templates : list) -> np.int32:
        """Extracts the timestamp from the top-left corner of the frame."""
        time_extract = input_frame[:20,211:275]
        time_extract = cv.cvtColor(time_extract,cv.COLOR_BGR2GRAY)
        stamp = []
        stamp.append(self.what_digit(time_extract[:,0:8],time_templates))
        stamp.append(self.what_digit(time_extract[:,8:16],time_templates))
        stamp.append(self.what_digit(time_extract[:,24:32],time_templates))
        stamp.append(self.what_digit(time_extract[:,32:40],time_templates))
        stamp.append(self.what_digit(time_extract[:,48:56],time_templates))
        stamp.append(self.what_digit(time_extract[:,56:64],time_templates))
        print(stamp,end='\r')
        time_stamp = int(stamp[0]*10 + stamp[1]) * 3600 + int(stamp[2]*10 + stamp[3]) * 60 + int(stamp[4]*10 + stamp[5])
        return time_stamp
    
    def get_fractional_timestamps(self,input_video,id,time_templates,verbose=False):
        """Returns the fractional timestamps associated with each frame."""
        input_video.seek(0)
        time_stamp_dict = dict()
        frame_count = int(len(input_video)) - 25
        cnt = -1
        current_stamp = 0
        next_stamp = 0
        read_time_stamps = []
        for i in tqdm(range(0,frame_count)):
            frame = input_video[i].asnumpy()
            next_stamp = self.get_frame_timestamps(frame,time_templates)
            read_time_stamps.append((next_stamp,i))
            current_stamp = next_stamp
        accepted_frame_count = frame_count
        current_stamp = read_time_stamps[0][0]
        accepted_time_stamps = []
        for i in range(0,frame_count):
            if (read_time_stamps[i][0] != current_stamp) and (read_time_stamps[i][0] != current_stamp + 1):
                print(f"Invalid time stamp: {read_time_stamps[i][0]} at frame {read_time_stamps[i][1]}.")
                print(f"Expected: {current_stamp}||{current_stamp+1}")
            else:
                accepted_time_stamps.append((read_time_stamps[i][0],read_time_stamps[i][1]))
                current_stamp = read_time_stamps[i][0]
        accepted_frame_count = len(accepted_time_stamps)
        for i in tqdm(range(0,accepted_frame_count)):
            if time_stamp_dict.__contains__(accepted_time_stamps[i][0]) == False:
                time_stamp_dict[accepted_time_stamps[i][0]] = [0,[]]
            time_stamp_dict[accepted_time_stamps[i][0]][0] += 1
            time_stamp_dict[accepted_time_stamps[i][0]][1].append(accepted_time_stamps[i][1])
        frame_time_stamps = [[],[]]
        frame_per_sec = []
        total_frame_count = 0
        for k,v in time_stamp_dict.items():
            frame_per_sec.append(v[0])
            total_frame_count += v[0]
            for p in range(v[0]):
                frame_time_stamps[0].append(k + p/v[0])
                frame_time_stamps[1].append(v[1][p])
        frame_time_stamps = np.asarray(frame_time_stamps)
        return frame_time_stamps, frame_time_stamps[0][0], frame_time_stamps[0][len(frame_time_stamps[0])-1]
    
    def sync_frames(self,frame_timestamps_a : np.ndarray, frame_timestamps_b : np.ndarray, frames_b : np.ndarray,frame_a : np.int32) -> [np.int32, np.float32]:
        """Returns the frame from (b) closest to the frame from (a)."""
        return (int(frames_b[np.argmin(np.abs(frame_timestamps_b - frame_timestamps_a[frame_a]))]),np.amin(np.abs(frame_timestamps_b - frame_timestamps_a[frame_a])))
    
    def add_video(self, path : str):
        self.video_paths.append(path)
        de_vid = de.VideoReader(path, ctx = de.gpu(0))
        id = len(self.video_paths) - 1
        fts, fmin,fmax = self.get_fractional_timestamps(de_vid,id,self.time_templates)
        self.min_stamp = np.amin([fmin,self.min_stamp])
        self.max_stamp = np.amax([fmax,self.max_stamp])
        print(f"Synchronization signal :: from {self.min_stamp} to {self.max_stamp}")
        self.fractional_timestamps.append(fts)
    
    def next_frame(self):
        self.current_frame = self.current_frame + 1
    
    def set_frame(self, new_frame : int):
        self.current_frame = new_frame
    
    def get_next_frame(self, video_id : int, frame_id : int):
        sync_frame = self.synchronized_frames[video_id][frame_id]
    
    def sync_video(self, id):
        sync_signal = np.arange(self.min_stamp,self.max_stamp,0.04)
        self.synchronized_frames.append([])
        for i in range(0,sync_signal.shape[0]):
            synhronized_frame_id, fractional_difference = self.sync_frames(sync_signal,self.fractional_timestamps[id][0],self.fractional_timestamps[id][1],i)
            if synhronized_frame_id > 0 and synhronized_frame_id < self.fractional_timestamps[id][1].shape[0] - 1:
                self.summed_diff += fractional_difference
                self.diff_cnt += 1
            self.synchronized_frames[id].append(synhronized_frame_id)
    
    def load(self, path : str):
        with open(path,"rb") as stf:
            self.video_paths=pickle.load(stf)
            self.fractional_timestamps=pickle.load(stf)
            self.synchronized_frames=pickle.load(stf)
            self.summed_diff=pickle.load(stf)
            self.diff_cnt=pickle.load(stf)
            self.min_stamp=pickle.load(stf)
            self.max_stamp=pickle.load(stf)
        print(self.video_paths)
        print(self.synchronized_frames)
    
    def save(self,path : str):
        with open(path,"wb") as stf:
            pickle.dump(self.video_paths,stf)
            pickle.dump(self.fractional_timestamps,stf)
            pickle.dump(self.synchronized_frames,stf)
            pickle.dump(self.summed_diff,stf)
            pickle.dump(self.diff_cnt,stf)
            pickle.dump(self.min_stamp,stf)
            pickle.dump(self.max_stamp,stf)
    
    def __del__(self):
        pass
    
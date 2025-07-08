import cv2
import os
import moviepy.editor as mp

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        video = mp.VideoFileClip(self.file_path)
        audio = video.audio
        audio_path = os.path.splitext(self.file_path)[0] + "_audio.wav"
        if not os.path.exists(audio_path):
            audio.write_audiofile(audio_path)

        cap = cv2.VideoCapture(self.file_path)
        frames = []
        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     frames.append(frame)
        cap.release()

        data = {
            "audio_path": audio_path,
            "frames": frames
        }
        return data
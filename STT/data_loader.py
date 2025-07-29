from typing import Any
import cv2
import os
import moviepy.editor as mp

class DataLoader:
    @classmethod
    def load_data(cls, file_path: str)-> dict[str, Any]:
        video = mp.VideoFileClip(file_path)
        audio = video.audio
        audio_path = os.path.splitext(file_path)[0] + "_audio.wav"
        if not os.path.exists(audio_path):
            audio.write_audiofile(audio_path)

        cap = cv2.VideoCapture(file_path)
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
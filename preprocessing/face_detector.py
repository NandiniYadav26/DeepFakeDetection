# File containing classes used for face detection. 

import os
import torch
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List


import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from torch.utils.data import Dataset


class VideoFaceDetector(ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @property
    @abstractmethod
    def _batch_size(self) -> int:
        pass

    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass


# Class implementing the MTCNN performing face detection
class FacenetDetector(VideoFaceDetector):

    def __init__(self, device=None) -> None:
        super().__init__()

        device = device if device is not None else "cpu"

        self.detector = MTCNN(
            device=device,
            thresholds=[0.6, 0.7, 0.7],
            margin=0,
        )

    def _detect_faces(self, frames) -> List:
        batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
        if batch_boxes is None:
            return []
        return [b.tolist() if b is not None else None for b in batch_boxes]

    @property
    def _batch_size(self):
        return 32

# Class for managing videos on which to perform face detection. The video is divided into frames when returned by getitem().
class VideoDataset(Dataset):

    def __init__(self, videos) -> None:
        super().__init__()
        self.videos = videos

    def __getitem__(self, index: int):
        video = self.videos[index]
        capture = cv2.VideoCapture(video)

        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 25

        frames = OrderedDict()

    # Extract only 8 evenly spaced frames
        num_frames_to_extract = 8
        if frames_num < num_frames_to_extract:
            frame_indices = list(range(frames_num))
        else:
            step = frames_num // num_frames_to_extract
            frame_indices = [i * step for i in range(num_frames_to_extract)]

        for i in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = capture.read()
            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            
            frames[i] = frame

        capture.release()

        return video, list(frames.keys()), fps, list(frames.values())

    def __len__(self) -> int:
        return len(self.videos)

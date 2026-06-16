# The videos are given as input to the network for training and inference in the form of sequences of faces extracted from the frames. 
# Faces are detected using a MTCNN in order to extract one per second. In the case of multiple faces within the same frame, all faces are extracted.

import argparse
import json
import os
import numpy as np
from typing import Type

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
import face_detector
from face_detector import VideoDataset, VideoFaceDetector
import argparse

def simple_collate(batch):
    return batch

def process_videos(videos, detector_cls: Type[VideoFaceDetector], opt):
    
    detector = face_detector.__dict__[detector_cls](device="cpu")

    dataset = VideoDataset(videos)
    loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=0,
        batch_size=1,
        collate_fn=simple_collate
    )

    missed_videos = [] # Used to print videos with no detected faces
    
    # For each video in the dataset, detect faces
    for item in tqdm(loader): 
        result = {}
        video, indices, fps, frames = item[0]
        id = video.split(opt.data_path)[-1]
        out_dir = opt.output_path + id
        out_dir = out_dir.replace("video.mp4", '')

        # Skip already detected videos to improve speed
        if os.path.exists(out_dir) and "video.json" in os.listdir(out_dir):
            continue
        
        if fps == 0:
            print("Zero fps video", video)
            continue

        
        result.update({i : b for i, b in zip(indices, detector._detect_faces(frames))})

        # Save faces as json dictionary into output folder
        os.makedirs(out_dir, exist_ok=True)
        
        with open(os.path.join(out_dir, "video.json"), "w") as f:
            json.dump(result, f)
        
        # Check if some faces have been detected
        found_faces = False
        for key in result:
            if type(result[key]) == list:
                found_faces = True
                break

        if not found_faces:
            print("Faces not found", video)
            missed_videos.append(video)

    # Display the missed videos
    if len(missed_videos) > 0:
        print("The detector did not find faces inside the following videos:")
        print(missed_videos)
        print(len(missed_videos))
        print("We suggest to re-run the code decreasing the detector threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default="datasets/FaceForensics/splits/train.csv", type=str)
    parser.add_argument('--data_path', default='datasets/FaceForensics/raw', type=str)
    parser.add_argument('--output_path', default='datasets/FaceForensics/faces/', type=str)
    parser.add_argument("--detector_type", default="FacenetDetector", choices=["FacenetDetector"])
    parser.add_argument('--workers', default=0, type=int)

    opt = parser.parse_args()
    print(opt)

    # Read CSV split file (video_path label)
    df = pd.read_csv(opt.list_file, sep=" ", header=None)

    video_names = df[0].tolist()

    videos_paths = [
        os.path.join(opt.data_path, name)
        for name in video_names
    ]

    print("Total videos:", len(videos_paths))

    process_videos(videos_paths, opt.detector_type, opt)


if __name__ == "__main__":
    main()
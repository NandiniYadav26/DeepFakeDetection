import os
import json
import cv2
import argparse
from tqdm import tqdm

def extract_faces_from_video(video_path, json_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open:", video_path)
        return

    with open(json_path, "r") as f:
        boxes_dict = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for frame_idx_str, boxes in boxes_dict.items():
        frame_idx = int(frame_idx_str)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if boxes is None:
            continue

        for face_id, box in enumerate(boxes):
            if box is None:
                continue

            x1, y1, x2, y2 = map(int, box)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            identity_dir = os.path.join(output_dir, str(face_id))
            os.makedirs(identity_dir, exist_ok=True)

            save_path = os.path.join(identity_dir, f"{frame_idx}.jpg")
            cv2.imwrite(save_path, face_crop)

    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True,
                        help="Path to raw FaceForensics videos")
    parser.add_argument("--faces_path", type=str, required=True,
                        help="Path where video.json files are stored")
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.faces_path):
        if "video.json" in files:
            json_path = os.path.join(root, "video.json")

            relative_path = os.path.relpath(root, args.faces_path)
            video_name = os.path.basename(relative_path) + ".mp4"

            video_path = os.path.join(args.raw_path, relative_path)

            if not os.path.exists(video_path):
                print("Video not found:", video_path)
                continue

            print("Processing:", video_path)
            extract_faces_from_video(video_path, json_path, root)


if __name__ == "__main__":
    main()
import cv2
from glob import glob
import os
import re

from utils.recursive_config import Config


def main():
    config = Config()

    # Define the path where your images are stored
    directory_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    image_folder = os.path.join(str(directory_path), ending, "color")
    output_video = os.path.join(image_folder, "video.mp4")

    # Function to sort file names in natural order
    def natural_sort_key(file):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", file)
        ]

    # Get a list of all image files and sort them naturally
    images = sorted(glob(os.path.join(image_folder, "*.jpg")), key=natural_sort_key)

    # Check if images were found
    if not images:
        print("No images found in the specified directory.")
        exit()

    # Read the first image to get the size
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
    fps = 12  # Frames per second, adjust as needed
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate over the images and write them to the video
    for image_path in images:
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    print("Video created successfully!")


if __name__ == "__main__":
    main()

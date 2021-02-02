# from https://github.com/rolux

import os
import numpy as np
import sys
from tqdm import tqdm
from ffhq_dataset.face_alignment import image_align
# from ffhq_dataset.dlib_landmarks_detector import DlibLandmarksDetector
from ffhq_dataset.fa_landmarks_detector import FaLandmarksDetector

MIN_WIDTH = 400


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    RAW_IMAGES_DIR = sys.argv[1]
    ALIGNED_IMAGES_DIR = sys.argv[2]

    # landmarks_detector = DlibLandmarksDetector()
    landmarks_detector = FaLandmarksDetector()
    for img_name in tqdm([x for x in os.listdir(RAW_IMAGES_DIR) if x[0] not in '._']):
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            if face_landmarks is None:
                continue

            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)

            np_landmarks = np.array(face_landmarks)
            max_x, _ = np.max(np_landmarks, axis=0)
            min_x, _ = np.min(np_landmarks, axis=0)
            if (max_x - min_x) < MIN_WIDTH:
                # print(face_img_name, (max_x - min_x))
                continue

            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks)

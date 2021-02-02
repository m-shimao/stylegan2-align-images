import cv2
import dlib
import bz2
from tensorflow.keras.utils import get_file

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
MIN_WIDTH = 500
MAX_WIDTH = 2048


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


class DlibLandmarksDetector:
    def __init__(self, predictor_model_path=None):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        if predictor_model_path is None:
            predictor_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                       LANDMARKS_MODEL_URL, cache_subdir='temp'))

        self.detector = dlib.get_frontal_face_detector()  # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        # img = dlib.load_rgb_image(image)
        img = cv2.imread(image, 0)
        if img is None:
            return None

        w = img.shape[1]
        resize_flag = False
        if w < MIN_WIDTH:
            return None

        if w > MAX_WIDTH:
            img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            resize_flag = True

        dets = self.detector(img, 1)
        if dets is None or len(dets) == 0:
            return None

        if resize_flag:
            return [(item.x * 2, item.y * 2) for item in self.shape_predictor(img, dets[0]).parts()]
        else:
            return [(item.x, item.y) for item in self.shape_predictor(img, dets[0]).parts()]

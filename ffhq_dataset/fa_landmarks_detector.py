import face_alignment
import torch
from skimage import io


class FaLandmarksDetector:
    def __init__(self):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    def get_landmarks(self, image):
        img = io.imread(image)
        dets = self.detector.get_landmarks(img)
        if dets is None:
            return

        for detection in dets:
            yield detection

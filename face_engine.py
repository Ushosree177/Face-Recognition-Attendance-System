import os
import pickle
import time
from datetime import datetime
from typing import Callable

import cv2
import numpy as np

from config import (
    CASCADE_PATH, DATASET_DIR, TRAINER_DIR, UNKNOWN_DIR,
    DETECT_SCALE_FACTOR, DETECT_MIN_NEIGHBORS, DETECT_MIN_SIZE,
    IMG_SIZE, MODEL_FILE, LABEL_FILE, UNKNOWN_SAVE_COOLDOWN,
)


class FaceDetector:

    def __init__(self):
        self._cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(
                f"Could not load Haar cascade from:\n{CASCADE_PATH}"
            )

    def detect(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=DETECT_SCALE_FACTOR,
            minNeighbors=DETECT_MIN_NEIGHBORS,
            minSize=DETECT_MIN_SIZE,
        )
        if not isinstance(faces, np.ndarray):
            faces = []
        return gray, faces


class DatasetManager:

    @staticmethod
    def registered_users() -> list[str]:
        if not os.path.exists(DATASET_DIR):
            return []
        return sorted(
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        )

    @staticmethod
    def user_folder(user_id: str, name: str) -> str:
        safe = name.strip().replace(" ", "_")
        folder = os.path.join(DATASET_DIR, f"{user_id.strip()}_{safe}")
        os.makedirs(folder, exist_ok=True)
        return folder

    @staticmethod
    def images_in_folder(folder: str) -> list[str]:
        return sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))

    @staticmethod
    def save_face(folder: str, index: int, gray_face: np.ndarray) -> str:
        path = os.path.join(folder, f"{index:04d}.jpg")
        cv2.imwrite(path, cv2.resize(gray_face, IMG_SIZE))
        return path

    @staticmethod
    def dataset_summary() -> dict:
        users = DatasetManager.registered_users()
        total = 0
        for u in users:
            fp = os.path.join(DATASET_DIR, u)
            total += len(DatasetManager.images_in_folder(fp))
        return {"users": len(users), "images": total}


class LBPHTrainer:

    def train(
        self,
        progress_cb: Callable[[str, int, int], None],
        done_cb: Callable[[bool, str], None],
    ):
        faces: list[np.ndarray] = []
        labels: list[int] = []
        label_map: dict[int, str] = {}
        label_id = 0

        users = DatasetManager.registered_users()
        if not users:
            done_cb(False, "No registered users found in the dataset.")
            return

        for folder_name in users:
            folder_path = os.path.join(DATASET_DIR, folder_name)
            label_map[label_id] = folder_name
            images = DatasetManager.images_in_folder(folder_path)

            for i, img_name in enumerate(images):
                img = cv2.imread(
                    os.path.join(folder_path, img_name),
                    cv2.IMREAD_GRAYSCALE,
                )
                if img is not None:
                    faces.append(cv2.resize(img, IMG_SIZE))
                    labels.append(label_id)
                progress_cb(folder_name, i + 1, len(images))

            label_id += 1

        if not faces:
            done_cb(False, "No valid images found in dataset.")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))

        os.makedirs(TRAINER_DIR, exist_ok=True)
        recognizer.save(os.path.join(TRAINER_DIR, MODEL_FILE))
        with open(os.path.join(TRAINER_DIR, LABEL_FILE), "wb") as f:
            pickle.dump(label_map, f)

        done_cb(
            True,
            f"Trained on {len(faces)} images across {len(users)} user(s).",
        )


class LBPHRecognizer:

    def __init__(self):
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._label_map: dict[int, str] = {}
        self.is_loaded = False

    def load(self) -> bool:
        model_path = os.path.join(TRAINER_DIR, MODEL_FILE)
        label_path = os.path.join(TRAINER_DIR, LABEL_FILE)
        if os.path.exists(model_path) and os.path.exists(label_path):
            self._recognizer.read(model_path)
            with open(label_path, "rb") as f:
                self._label_map = pickle.load(f)
            self.is_loaded = True
            return True
        return False

    def predict(self, gray_face: np.ndarray, threshold: int,) -> tuple[str, float, bool]:
        if not self.is_loaded:
            return "Unknown", 999.0, False

        try:
            resized = cv2.resize(gray_face, IMG_SIZE)
            label_id, confidence = self._recognizer.predict(resized)

            if confidence < threshold:
                folder = self._label_map.get(label_id, "Unknown")
                parts = folder.split("_", 1)
                name = parts[1].replace("_", " ") if len(parts) > 1 else folder
                return name, confidence, True

        except Exception:
            pass

        return "Unknown", 999.0, False


class UnknownFaceStore:

    def __init__(self):
        os.makedirs(UNKNOWN_DIR, exist_ok=True)
        self._last_save: dict[str, float] = {}   # grid-cell key → epoch seconds

    def try_save(self, face_bgr: np.ndarray, grid_key: str) -> bool:
        now = time.time()
        if now - self._last_save.get(grid_key, 0) < UNKNOWN_SAVE_COOLDOWN:
            return False

        self._last_save[grid_key] = now
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(os.path.join(UNKNOWN_DIR, f"unknown_{ts}.jpg"), face_bgr)
        return True

    def count_today(self) -> int:
        today = datetime.now().strftime("%Y%m%d")
        try:
            return sum(1 for f in os.listdir(UNKNOWN_DIR) if today in f)
        except OSError:
            return 0

    def reset_cooldowns(self):
        self._last_save.clear()
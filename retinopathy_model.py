"""Two-stage diabetic retinopathy model.

Stage 1: a ResNet50 vision model (model.h5) grades a retina image and returns
         a probability for each of the five grades.
Stage 2: a random forest (models/stacker_rf.joblib) combines these five
         probabilities with patient data into the final grade.

Typical use:

    model = RetinopathyModel("model.h5", "models/stacker_rf.joblib")
    patient = load_patient("000c1434d8d7")
    image = load_image("images/000c1434d8d7.png")
    prediction = model.predict(image, patient)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import keras
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

GRADES: dict[int, str] = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
IMAGE_SIZE: tuple[int, int] = (320, 320)

# input columns of the random forest, in this order
CNN_FEATURES: list[str] = [f"cnn_p{k}" for k in range(5)]
PATIENT_FEATURES: list[str] = ["age", "male", "diabetes_type", "diabetes_years", "hba1c", "systolic_bp"]
FEATURES: list[str] = CNN_FEATURES + PATIENT_FEATURES

# a preprocessed retina image: float32 array of shape (320, 320, 3) with values in [0, 1]
Image = npt.NDArray[np.float32]
# probabilities for the five grades: float array of shape (5,), or (n, 5) for a batch
Probabilities = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Patient:
    """One row of data/train.csv or data/test.csv."""
    id_code: str            # image file name without .png
    age: int                # years
    gender: Literal["F", "M"]
    diabetes_type: Literal[1, 2]
    diabetes_years: int     # years since diabetes diagnosis
    hba1c: float            # average blood sugar over the last 3 months, in percent
    systolic_bp: int        # systolic blood pressure, in mmHg
    diagnosis: int | None = None   # grade 0-4 assigned by a clinician, if known

    def features(self) -> dict[str, float]:
        """The patient data in the form used by the random forest (see PATIENT_FEATURES)."""
        return {"age": self.age, "male": int(self.gender == "M"), "diabetes_type": self.diabetes_type,
                "diabetes_years": self.diabetes_years, "hba1c": self.hba1c, "systolic_bp": self.systolic_bp}


@dataclass(frozen=True)
class Prediction:
    grade: int                          # final grade 0-4 (stage 2)
    probabilities: Probabilities        # final probabilities, shape (5,)
    cnn_probabilities: Probabilities    # vision model probabilities, shape (5,)

    @property
    def grade_name(self) -> str:
        return GRADES[self.grade]

    @property
    def cnn_grade(self) -> int:
        """Grade the vision model alone would assign."""
        return int(self.cnn_probabilities.argmax())


def load_patients(data_dir: str | Path = "data") -> list[Patient]:
    """All patients from data/train.csv and data/test.csv."""
    df = pd.concat([pd.read_csv(Path(data_dir) / f) for f in ("train.csv", "test.csv")])
    fields = Patient.__dataclass_fields__
    return [Patient(**{k: v for k, v in row.items() if k in fields}) for row in df.to_dict("records")]


def load_patient(patient_id: str, data_dir: str | Path = "data") -> Patient:
    """The patient with the given id (the image file name without .png)."""
    for patient in load_patients(data_dir):
        if patient.id_code == patient_id:
            return patient
    raise KeyError(f"no patient with id {patient_id}")


def load_image(path: str | Path) -> Image:
    """Load and preprocess a retina image the same way as during training."""
    image = keras.utils.load_img(path, target_size=IMAGE_SIZE)
    return (keras.utils.img_to_array(image) / 255.0).astype(np.float32)


def stacker_features(cnn_probabilities: Probabilities, patients: Sequence[Patient]) -> pd.DataFrame:
    """Random forest input: one row per patient with the columns in FEATURES.

    cnn_probabilities has shape (n, 5), one row per patient.
    """
    features = pd.DataFrame(np.asarray(cnn_probabilities), columns=CNN_FEATURES)
    patient_features = pd.DataFrame([p.features() for p in patients], columns=PATIENT_FEATURES)
    return pd.concat([features, patient_features], axis=1)


class RetinopathyModel:
    def __init__(self, cnn_path: str | Path = "model.h5",
                 stacker_path: str | Path | None = "models/stacker_rf.joblib"):
        """Loads both models; with stacker_path=None only the vision model (for training the stacker)."""
        self.cnn: keras.Model = keras.models.load_model(cnn_path, compile=False)
        self.stacker: RandomForestClassifier | None = joblib.load(stacker_path) if stacker_path else None

    def cnn_probabilities(self, images: Sequence[Image]) -> Probabilities:
        """Stage 1 only: vision model probabilities, shape (n, 5)."""
        return self.cnn.predict(np.stack(images), verbose=0).astype(np.float64)

    def predict_batch(self, images: Sequence[Image], patients: Sequence[Patient]) -> list[Prediction]:
        """Both stages for a batch; images[i] belongs to patients[i]."""
        cnn_probs = self.cnn_probabilities(images)
        probs = self.stacker.predict_proba(stacker_features(cnn_probs, patients))
        return [Prediction(int(p.argmax()), p, c) for p, c in zip(probs, cnn_probs)]

    def predict(self, image: Image, patient: Patient) -> Prediction:
        """Both stages for a single patient."""
        return self.predict_batch([image], [patient])[0]

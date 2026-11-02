"""TabPFN stockout model fitted on a random sample of rows, predicting in chunks to fit in memory."""
import os

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from src.stockout.features import as_matrix


class TabPFNStockout:
    def __init__(self, train_rows, members, chunk_rows, seed, device="cuda"):
        self.train_rows, self.members, self.chunk_rows, self.seed, self.device = train_rows, members, chunk_rows, seed, device

    def fit(self, frame):
        sample = frame.sample(min(self.train_rows, len(frame)), random_state=self.seed)
        self.model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=self.device, random_state=self.seed, n_estimators=self.members)
        self.model.fit(as_matrix(sample), sample.label.to_numpy().astype(int))
        return self

    def predict(self, frame):
        inputs = as_matrix(frame)
        return np.concatenate([self.model.predict_proba(inputs[i:i + self.chunk_rows])[:, 1] for i in range(0, len(inputs), self.chunk_rows)])

import torch
import numpy as np

from agents.abnormal_activity.heatmap_generator import HeatmapGenerator
from agents.abnormal_activity.sequence_builder import SequenceBuilder
from models.abnormal_activity.cnn_lstm import CNNLSTM


class AbnormalActivityAgent:
    def __init__(self, frame_width, frame_height, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Components
        self.heatmap_generator = HeatmapGenerator(frame_width, frame_height)
        self.sequence_builder = SequenceBuilder(sequence_length=16)

        # Model
        self.model = CNNLSTM().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Threshold
        self.threshold = 0.5

    def process(self, scene_state):
        # Step 1: Heatmap
        heatmap = self.heatmap_generator.generate(scene_state)

        # Step 2: Sequence
        sequence = self.sequence_builder.update(heatmap)

        if sequence is None:
            return None  # Not enough frames yet

        # Step 3: Convert to tensor
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Step 4: Inference
        with torch.no_grad():
            prob = self.model(x).item()

        # Step 5: Decision
        is_abnormal = prob > self.threshold

        return {
            "abnormal": is_abnormal,
            "score": prob
        }
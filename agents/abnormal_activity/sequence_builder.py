from collections import deque
import numpy as np

class SequenceBuilder:
    def __init__(self, sequence_length=16):
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)

    def update(self, heatmap):
        """
        heatmap: (C, H, W)
        """
        self.buffer.append(heatmap)

        # If not enough frames yet
        if len(self.buffer) < self.sequence_length:
            return None

        # Convert to numpy array
        sequence = np.array(self.buffer)  # shape: (T, C, H, W)
        return sequence
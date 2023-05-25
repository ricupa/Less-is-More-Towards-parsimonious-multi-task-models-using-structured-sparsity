import torch
from torchmetrics import Metric




class InferenceTime(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("times", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, batch_time: float):
        self.times = torch.cat([self.times, torch.tensor([batch_time])])

    def compute(self):
        mean_time = torch.mean(self.times)
        std_dev = torch.std(self.times)
        return {"average_inference_time": mean_time, "std_dev": std_dev}


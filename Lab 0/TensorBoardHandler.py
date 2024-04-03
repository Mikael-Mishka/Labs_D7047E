import torch.utils.tensorboard as tb

class TensorBoardHandler():
    def __init__(self):
        self.writer = tb.SummaryWriter()

    # "loss/train", [2.0, 2, ...], epochs
    def log_scalar(self, tag, values, steps):
        # (name_of_metric, value, epoch)
        for value, epoch in zip(values, steps):
            self.writer.add_scalar(tag, value, epoch)
        self.close()


    def close(self):
        self.writer.close()
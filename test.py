import torch
import unittest
from cvae import *


class TestCVAE(unittest.TestCase):

    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = ConditionalVAE(1, 64*64, 128)

    def test_forward(self):
        x = torch.randn(16, 1, 64, 64)
        c = torch.randn(16, 64 * 64)
        y = self.model(x, c)
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 1, 64, 64)
        c = torch.randn(16, 64 * 64)
        result = self.model(x, y = c)
        loss = self.model.loss_function(*result, M_N = 0.005)
        print(loss)


if __name__ == '__main__':
    unittest.main()


#   model_params:
#   name: 'ConditionalVAE'
#   in_channels: 3
#   num_classes: 40
#   latent_dim: 128

# data_params:
#   data_path: "Data/"
#   train_batch_size: 64
#   val_batch_size:  64
#   patch_size: 64
#   num_workers: 4


# exp_params:
#   LR: 0.005
#   weight_decay: 0.0
#   scheduler_gamma: 0.95
#   kld_weight: 0.00025
#   manual_seed: 1265

# trainer_params:
#   gpus: [1]
#   max_epochs: 10

# logging_params:
#   save_dir: "logs/"
#   name: "ConditionalVAE"
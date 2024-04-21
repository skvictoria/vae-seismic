import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.image_sum = 0
        self.image_sq_sum = 0
        self.num_images = 0
        self.automatic_optimization = False

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              #optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        # Manual optimization
        self.manual_backward(train_loss['loss'])  # Backward pass
        self.optimizers().step()  # Optimizer step
        self.optimizers().zero_grad()  # Zero gradients

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            #optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
         # Manual optimization
        #self.manual_backward(val_loss['loss'])  # Backward pass
        #self.optimizers().step()  # Optimizer step
        #self.optimizers().zero_grad()  # Zero gradients

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    def on_validation_end(self) -> None:
        self.sample_images()
        # mean_image = self.image_sum / self.num_images
        # std_image = (self.image_sq_sum / self.num_images - mean_image ** 2).sqrt()  # variance = E[X^2] - (E[X])^2

        # # You can save or log these images as needed, e.g.,
        # self.logger.experiment.add_image('mean_image', mean_image, self.current_epoch)
        # self.logger.experiment.add_image('std_image', std_image, self.current_epoch)

        vutils.save_image(self.mean_image.data,
                          os.path.join(self.logger.log_dir , 
                                       "mean_image", 
                                       f"mean_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        
        vutils.save_image(self.std_image.data,
                          os.path.join(self.logger.log_dir , 
                                       "std_image", 
                                       f"std_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        # Optionally, you might want to clear the accumulators to free up memory
        #del self.image_sum
        #del self.image_sq_sum
        #torch.cuda.empty_cache()  # If using GPU
        #self.plot_latent_space(self.model, self.trainer.datamodule.test_dataloader(), self.curr_device)

    def plot_latent_space(self, vae_model, data_loader, device):
        vae_model.eval()
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for images, lbls in data_loader:
                images = images.to(device)
                # Get latent vector
                _, _, z, _ = vae_model(images)  # Adjust this depending on your model's output
                latent_vectors.append(z)
                labels.append(lbls)
        
        latent_vectors = torch.cat(latent_vectors, 0).cpu().numpy()
        labels = torch.cat(labels, 0).cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(latent_vectors)
        
        # Plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
        plt.show()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device).float()

        # Convert to numpy for plotting
        # Select the first image in the batch for simplicity
        image_to_plot = test_input[0].cpu().detach().permute(1, 2, 0).numpy()
        plt.imshow(image_to_plot, interpolation='nearest')
        plt.title("Sample Test Input")
        plt.colorbar()
        plt.axis('off')
        plot_save_path = os.path.join(self.logger.log_dir, "Plots", f"test_input_epoch_{self.current_epoch}.png")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(plot_save_path)
        plt.close()


        image_to_plot = test_label[0].cpu().detach().permute(1, 2, 0).numpy()
        plt.imshow(image_to_plot, interpolation='nearest')
        plt.title("Sample Test Input")
        plt.colorbar()
        plt.axis('off')
        plot_save_path = os.path.join(self.logger.log_dir, "Labels", f"test_input_epoch_{self.current_epoch}_1.png")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(plot_save_path)
        plt.close()

        image_to_plot = test_label[1].cpu().detach().permute(1, 2, 0).numpy()
        plt.imshow(image_to_plot, interpolation='nearest')
        plt.title("Sample Test Input")
        plt.colorbar()
        plt.axis('off')
        plot_save_path = os.path.join(self.logger.log_dir, "Labels", f"test_input_epoch_{self.current_epoch}_2.png")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(plot_save_path)
        plt.close()

        image_to_plot = test_label[2].cpu().detach().permute(1, 2, 0).numpy()
        plt.imshow(image_to_plot, interpolation='nearest')
        plt.title("Sample Test Input")
        plt.colorbar()
        plt.axis('off')
        plot_save_path = os.path.join(self.logger.log_dir, "Labels", f"test_input_epoch_{self.current_epoch}_3.png")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(plot_save_path)
        plt.close()

        image_to_plot = test_label[3].cpu().detach().permute(1, 2, 0).numpy()
        plt.imshow(image_to_plot, interpolation='nearest')
        plt.title("Sample Test Input")
        plt.colorbar()
        plt.axis('off')
        plot_save_path = os.path.join(self.logger.log_dir, "Labels", f"test_input_epoch_{self.current_epoch}_4.png")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(plot_save_path)
        plt.close()

        image_to_plot = test_label[4].cpu().detach().permute(1, 2, 0).numpy()
        plt.imshow(image_to_plot, interpolation='nearest')
        plt.title("Sample Test Input")
        plt.colorbar()
        plt.axis('off')
        plot_save_path = os.path.join(self.logger.log_dir, "Labels", f"test_input_epoch_{self.current_epoch}_5.png")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(plot_save_path)
        plt.close()
        
#         test_input, test_label = batch
        vutils.save_image(test_input.data,
                          os.path.join(self.logger.log_dir , 
                                       "Test_Input", 
                                       f"input_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        
        vutils.save_image(test_label.data,
                          os.path.join(self.logger.log_dir , 
                                       "Test_Label", 
                                       f"label_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        
        
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(64,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
            
            sampled_list = self.model.sample_fixed_y(64, self.curr_device, labels=test_label)
            images_tensor = torch.stack(sampled_list)  # Shape will be [128, 1, 64, 64]

            # Calculate the mean image
            self.mean_image = images_tensor.mean(dim=0)
            self.std_image = images_tensor.std(dim=0, unbiased=True)
            
            
        except Warning:
            pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'], weight_decay=self.params['weight_decay'])
        return optimizer

    # def configure_optimizers(self):

    #     optims = []
    #     scheds = []

    #     optimizer = optim.Adam(self.model.parameters(),
    #                            lr=self.params['LR'],
    #                            weight_decay=self.params['weight_decay'])
    #     optims.append(optimizer)
    #     # Check if more than 1 optimizer is required (Used for adversarial training)
    #     try:
    #         if self.params['LR_2'] is not None:
    #             optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
    #                                     lr=self.params['LR_2'])
    #             optims.append(optimizer2)
    #     except:
    #         pass

    #     try:
    #         if self.params['scheduler_gamma'] is not None:
    #             scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
    #                                                          gamma = self.params['scheduler_gamma'])
    #             scheds.append(scheduler)

    #             # Check if another scheduler is required for the second optimizer
    #             try:
    #                 if self.params['scheduler_gamma_2'] is not None:
    #                     scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
    #                                                                   gamma = self.params['scheduler_gamma_2'])
    #                     scheds.append(scheduler2)
    #             except:
    #                 pass
    #             return optims, scheds
    #     except:
    #         return optims
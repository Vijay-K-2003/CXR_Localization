# PyTorch Imports
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchmetrics

# Helper Imports
import os
import pandas as pd
from PIL import Image
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Lightning Imports
import pytorch_lightning as L
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Local Imports
import swinv2
import dataset

seed = 0
seed_everything(seed, workers=True)

device = torch.device("cpu")

print('Verification#####################################')
print('Seed			 : ', seed)
print('CUDA available   : ', torch.cuda.is_available())
print('CUDA devices	 : ', torch.cuda.device_count())
print('CUDA version	 : ', torch.__version__)
print('#################################################')

class SwinV2ObjectDetector(nn.Module):
	def __init__(self, num_classes=16):
		super().__init__()
		
		self.swin_backbone = swinv2.SwinTransformerV2(
			img_size=384,
			embed_dim=192,
			depths=[2, 2, 18, 2],
			num_heads=[6, 12, 24, 48],
			window_size=24,
			drop_path_rate=0.2,
			pretrained_window_sizes=[12, 12, 12, 6]
		)
		self.swin_backbone.head = nn.Identity()
		
		self.classifier = nn.Linear(1536, num_classes)
		
		self.regressor = nn.Linear(1536, 4)
		
	def forward(self, x):
		features = self.swin_backbone(x)
		print(f"Features shape: {features.shape}")
		
		class_scores = self.classifier(features)
		
		bbox_preds = self.regressor(features)
		
		return class_scores, bbox_preds

class NeuralNet(L.LightningModule):
	def __init__(self, model, num_classes, learning_rate):
		super().__init__()
		self.automatic_optimization=False
		self.num_classes = num_classes
		self.learning_rate = learning_rate
		self.model = model
		
		self.train_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes)
		self.val_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes)
		self.test_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes)

		self.classifier_loss = F.cross_entropy
		self.regressor_loss = F.smooth_l1_loss

		self.classifier_train_loss = []
		self.regressor_train_loss = []
		self.classifier_val_loss = []
		self.regressor_val_loss = []
		self.classifier_test_loss = []
		self.regressor_test_loss = []
		
		print(model)

		print("INIT SwinV2ObjectDetector#############################")
		print("Learning Rate 		:", learning_rate)
		print("Classes 				:", num_classes)
		print("Accuracy Metric 		:", self.train_acc)
		print("Classifier Loss 		:", self.classifier_loss)
		print("Regressor Loss	   :", self.regressor_loss)
		print("######################################################")

	def forward(self, images):
		return self.model(images)

	def training_step(self, batch, batch_idx):
		opt_classifier, opt_regressor = self.optimizers()

		images, labels, bboxs = batch
		
		label_preds, bbox_preds = self.forward(images)
		
		train_acc = self.train_acc(label_preds, labels)

		classifier_loss = self.classifier_loss(label_preds, labels)
		self.classifier_train_loss.append(classifier_loss.item())
		regressor_loss = self.regressor_loss(bbox_preds, bboxs)
		self.regressor_train_loss.append(regressor_loss.item())

		opt_classifier.zero_grad()
		self.manual_backward(classifier_loss)
		opt_classifier.step()

		opt_regressor.zero_grad()
		self.manual_backward(regressor_loss)
		opt_regressor.step()
		
		self.log_dict({
			'classifier_loss', classifier_loss.item(),
			'regressor_loss', regressor_loss.item()
		}, on_step=True, prog_bar=True)

	def on_train_epoch_end(self):
		classifier_train_mean_loss = torch.mean(torch.tensor(self.classifier_train_loss))
		regressor_train_mean_loss = torch.mean(torch.tensor(self.regressor_train_loss))

		train_acc = self.train_acc.compute()

		self.print('Epoch : 							', self.current_epoch)
		self.print('Train Classification accuracy : 	', train_acc)
		self.print('Classifier Train Mean loss : 		', classifier_train_mean_loss)
		self.print('Regressor Train Mean loss : 		', regressor_train_mean_loss)

		self.classifier_train_loss = []
		self.regressor_train_loss = []
		self.train_acc.reset()

	def validation_step(self, batch, batch_idx):
		images, labels, bboxs = batch

		label_preds, bbox_preds = self.forward(images)

		classifier_loss = self.classifier_loss(label_preds, labels)
		self.classifier_val_loss.append(classifier_loss.item())
		regressor_loss = self.regressor_loss(bbox_preds, bboxs)
		self.regressor_val_loss.append(regressor_loss.item())

		val_acc = self.val_acc(label_preds, labels)

		return classifier_loss + regressor_loss

	def on_validation_epoch_end(self):
		classifier_val_mean_loss = torch.mean(torch.tensor(self.classifier_val_loss))
		regressor_val_mean_loss = torch.mean(torch.tensor(self.regressor_val_loss))
		val_acc = self.val_acc.compute()

		self.log('classifier_val_loss', classifier_val_mean_loss.item(), on_epoch=True, sync_dist=True)
		self.log('regressor_val_loss', regressor_val_mean_loss.item(), on_epoch=True, sync_dist=True)
		self.log('val_acc', val_acc.item(), on_epoch=True, sync_dist=True)

		self.print('Epoch								:', self.current_epoch)
		self.print('Train Classification accuracy	   :', val_acc)
		self.print('Classifier Train Mean loss			:', classifier_val_mean_loss)
		self.print('Regressor Train Mean loss			:', regressor_val_mean_loss)

		self.save_hyperparameters()

		self.val_acc.reset()
		self.classifier_val_loss = []
		self.regressor_val_loss = []

	# def test_step(self, batch, batch_idx):
	# 	images = batch
	# 	label_preds, bbox_preds = self.forward(images)
		
	# 	test_loss = self.loss(logits, labels)
	# 	self.test_loss.append(test_loss)

	# 	test_acc = self.test_acc(logits, labels)
		
	# 	return(test_loss)

	# def on_test_epoch_end(self):
	# 	test_loss = torch.mean(torch.tensor(self.test_loss))		
	# 	test_acc = self.test_acc.compute()

	# 	self.log('test_loss', test_loss.item(), on_epoch=True, sync_dist=True)
	# 	self.log('test_acc', test_acc.item(), on_epoch=True, sync_dist=True)

	# 	self.print('Epoch : 		', self.current_epoch)
	# 	self.print('Test accuracy : 	', test_acc.item())
	# 	self.print('Test mean loss : 	', test_loss.item())	

	# 	self.test_acc.reset()
	# 	self.test_loss = []
	
	def lr_scheduler_step(self, scheduler, metric):
		if metric:
			print('metric', metric)
			scheduler.step(metric)
		else:
			scheduler.step()
		
	def configure_optimizers(self):
		optimizer_classifier = optim.Adam(list(self.model.swin_backbone.parameters()) + list(self.model.classifier.parameters()), lr=self.learning_rate)
		optimizer_regressor = optim.Adam(list(self.model.swin_backbone.parameters()) + list(self.model.regressor.parameters()), lr=self.learning_rate)

		scheduler_classifier = optim.lr_scheduler.MultiStepLR(
			optimizer_classifier,
			milestones=[10, 15, 20],
			gamma=0.1,
			verbose=True
		)
		scheduler_regressor = optim.lr_scheduler.MultiStepLR(
			optimizer_regressor,
			milestones=[10, 15, 20],
			gamma=0.1,
			verbose=True
		)

		return (
			{
				"optimizer": optimizer_classifier,
				"lr_scheduler": {
					"scheduler": scheduler_classifier,
					"monitor": "classifier_val_loss",
					"interval": "epoch",
					"frequency": 1
				}
			}, 
			{
				"optimizer": optimizer_regressor,
				"lr_scheduler": {
					"scheduler": scheduler_regressor,
					"monitor": "regressor_val_loss",
					"interval": "epoch",
					"frequency": 1
				}
			}
		)

img_dir = './VinBigDataCXR/train'
csv_file = './VinBigDataCXR/train.csv'

data_module = dataset.VinBigDataCXRDatamodule(img_dir=img_dir, csv_file=csv_file, batch_size=16, num_workers=4)

data_module.setup(stage="fit")

train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()

num_classes = 16
learning_rate = 1e-4

swinv2_model = SwinV2ObjectDetector(num_classes=num_classes)

lightning_model = NeuralNet(model=swinv2_model, num_classes=num_classes, learning_rate=learning_rate)

logger = TensorBoardLogger("logs", name="swinv2_object_detection")

checkpoint_callback = ModelCheckpoint(
	monitor="regressor_val_loss",
	dirpath="./model_history/checkpoints/",
	filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
	save_top_k=1,
	mode="min",
	every_n_epochs=1,
)

lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = Trainer(
	max_epochs=5,
	devices=1,
	accelerator="gpu",
	max_time=timedelta(hours=1),
	logger=logger,
	callbacks=[checkpoint_callback, lr_monitor],
	precision=16,
)

trainer.fit(lightning_model, train_dataloader, val_dataloader)

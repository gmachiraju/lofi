from collections import OrderedDict
from functools import partial
import torch.nn as nn
from torchvision.models.vision_transformer import ConvStemConfig, Encoder
from torch import nn
from typing import Callable, List, Optional
import torch
from .embed_patches import train_on_Z
import math
import argparse
import torch.optim as optim

# Constants/defaults
#-----------
print_every = 10
val_every = 20
#-----------


class AttentionHead(nn.Module):
    """An attention head resembling the Vision Transformer attention."""
    def __init__(
        self,
        height: int,
        width: int,
        num_layers: int,
        num_heads: int,
        input_dim: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        """Attention head that follows ViT configuration.
        Args:
            input_dim: Dimensions of the input embedding.
                If this is different than ``hidden_dim`` a projection layer will be created.
        """
        super().__init__()

        # Have each head operate on each chunk.
        if num_heads == "chunks":
            num_heads = height * width

        self.proj = None
        if input_dim != hidden_dim:
            self.proj = nn.Linear(input_dim, hidden_dim)

        seq_length = height * width

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: The embedding - shape (batch, dim, h, w)
        """
        n, dim, height, width = z.shape

        z = z.reshape(n, dim, height*width)

        if self.proj is not None:
            z = self.proj(z)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        z = z.permute(0, 2, 1)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, z], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def build_model(args):
	pass

def build_dataloader(args):
	pass

def build_lr_scheduler(args):
	pass


def train_classifier(model, device, optimizer, args, save_embeds_flag=False):
	"""
	Train a model on image data using the PyTorch Module API.

	Inputs:
	- model: A PyTorch Module giving the model to train.
	- optimizer: An Optimizer object we will use to train the model
	- epochs: (Optional) A Python integer giving the number of epochs to train for

	Returns: Nothing, but prints model accuracies during training.
	"""
	# Logging with Weights & Biases
	#-------------------------------
	os.environ["WANDB_MODE"] = "online"
	experiment = "PatchCNN-" + args.model_class + "-" + args.dataset_name
	wandb.init(project=experiment, entity="selfsup-longrange")	
	wandb.config = {
	  "learning_rate": LEARN_RATE,
	  "epochs": args.num_epochs,
	  "batch_size": args.batch_size,
	  **args
	}

	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
	print("hyperparams:\n" + "="*30)
	print("Adam optimizer learn rate:", args.learn_rate)
	print("Starting training procedure shortly...\n")

	if args.model_to_load is None:
		train_losses = []
	else:
		train_losses = utils.deserialize(args.model_path + "/" + args.string_details + "_trainloss.obj") # same as train loss

	model = model.to(device=device)

	att_flag = False
	if args.model_class == "VGG_att":
		att_flag = True
	print("Now training:", args.model_class)

	if args.model_class.startswith("VGG") == True:
		if args.model_class != "VGG_att": # this errors 
			# summary(model, input_size=(args.channel_dim, args.patch_size, args.patch_size)) # print model
			pass
	
	train_loader = DataLoaderCustom(args)
	model.train()

	# files loaded differently per model class
	for e in range(0 + args.prev_epoch, args.num_epochs + args.prev_epoch):
		print("="*30 + "\n", "Beginning epoch", e, "\n" + "="*30)
	
		if save_embeds_flag == True:
			embed_dict = defaultdict(list)

		for t, (fxy, x, y) in enumerate(train_loader):
			x = torch.from_numpy(x)
			y = torch.from_numpy(y)
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)

			if args.model_class == "VGG_att":
				[scores, c1, c2, c3] = model(x)
			elif args.model_class in ["VGG19", "VGG19_bn"]:
				scores = model(x)

			train_loss = F.cross_entropy(scores, y)
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()

			if t % print_every == 0:
				print('Iteration %d, train loss = %.4f' % (t + print_every, train_loss.item()))
				preds, probs, num_correct, num_samples = check_mb_accuracy(scores, y)
				acc = float(num_correct) / num_samples
				print('minibatch training accuracy: %.4f' % (acc * 100))
				# more logging
				wandb.log({"loss": train_loss})

			train_losses.append(train_loss.item())
			gc.collect()
			scheduler.step()

			if save_embeds_flag == True:
				embed_dict = store_embeds(embed_dict, fxy, x, model, args, att_flag)
				serialize(embed_dict, args.cache_path + "/" + args.string_details + "-curr_embeddings_train.obj")
			# save embeddings every 4 epochs for standard classifiers
			if save_embeds_flag == True and ((e+1) % 4 == 0):
				serialize(embed_dict, args.cache_path + "/" + args.string_details + "-epoch" + str(e) + "-embeddings_train.obj")
				
		# save model per epoch
		torch.save(model, args.model_path + "/" + args.string_details + "_epoch%s.pt" % e)
		# Future: check val acc every epoch

		# cache the losses every epoch
		serialize(train_losses, args.model_path + "/" + args.string_details + "_trainloss.obj")
		fig = plt.plot(train_losses, c="blue", label="train loss")
		plt.savefig(args.model_path + "/"  + args.string_details + "_trainloss.png", bbox_inches="tight")

		# more logging
		wandb.log({"end-of-epoch loss": train_loss})

		# save model per epoch
		print("saving model for epoch", e, "\n")
		torch.save(model, args.model_path + "/" + args.string_details + "_epoch%s.pt" % e)
		# serialize(train_losses, args.model_path + "/" + args.string_details + "_trainloss.obj")
		# fig = plt.plot(train_losses, c="blue", label="train loss")
		# plt.savefig(args.model_path + "/"  + args.string_details + "_trainloss.png", bbox_inches="tight")
		torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, args.model_path + "/" + args.string_details + "_epoch%s.sd" % e)
		torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, args.model_path + "/" + args.string_details + ".sd")
		# always keep a backup
		torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, args.model_path + "/BACKUP-" + args.string_details + ".sd")

	# full model save
	torch.save(model, args.model_path + "/" + args.string_details + "_full.pt")
	return train_losses


def main():

	# ARGPARSE
	#==========
	parser = argparse.ArgumentParser()
	parser.add_argument('--description', default="no-description", type=str, help='Description of your experiement, with no spaces. E.g. VGG19_bn-random_loading-label_inherit-bce_loss-on_MFL-1')
	parser.add_argument('--model_class', default=None, type=str, help='Select one of: VGG19/VGG19_bn/VGG_att.')
	parser.add_argument('--dataset_name', default=None, type=str, help="What you want to name your dataset. For pre-defined label dictionaries, use: u54codex to search utils.")
	parser.add_argument('--dataloader_type', default="stored", type=str, help="Type of data loader: stored vs otf (on-the-fly).")
	parser.add_argument('--toy_flag', default=False, action="store_true", help="T/F for a smaller training dataset for rapid experimentation. Default is True.")
	parser.add_argument('--overfit_flag', default=False, action="store_true", help="T/F for intentional overfitting. Run name is modified to reflect this. Default is False.")
	parser.add_argument('--use_gpu', action="store_true")
	
    # hyperparameters
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs to train for. Default is 10.')
	parser.add_argument('--batch_size', default=36, type=int, help="Batch size. dDfault is 36.")
    parser.add_argument('--lr', default=5e-4, type=float, help="Learning rate. Default is 5e-4")

	# parameters for patches
	parser.add_argument('--loss', default="bce", type=str, help="Patch loss function. Default is bce. Future support for uncertainty.")

	# paths
	parser.add_argument('--data_path', default=None, type=str, required=True, help="Dataset path. If patches, will use stored data loader, if images, will use OTF data loader.")
	parser.add_argument('--patchlist_path', default=None, type=str, help="Patch list path. This is a cached result of the preprocess.py script.")
	parser.add_argument('--labeldict_path', default=None, type=str, help="Label dictionary path. This is a cached result of the preprocess.py script.")
	parser.add_argument('--model_path', default=None, required=True, type=str, help="Where you'd like to save the models.")
	parser.add_argument('--cache_path', default=None, type=str, help="Where you'd like to save the model outputs.")

	args = parser.parse_args()

	# PRINTS
	#========
	print("\nBEGINNING TRAINING MODEL:", args.model_class + "\n" + "="*60)
	print("We get to train on...\n" + "-"*60)
	if args.patchlist_path:
		patch_list = deserialize(args.patchlist_path)
		print("train set size (#unique patches):", len(patch_list))
		del patch_list # we can deserialize in other functions
	
	print("of patch size:", args.patch_size)
	print("train set unique images:", len(label_dict))

	# SET-UP
	#========
	print("GPU detected?", torch.cuda.is_available())
	if args.use_gpu and torch.cuda.is_available():
		device = torch.device('cuda')
		print("\nNote: gpu available & selected!")
	else:
		device = torch.device('cpu')
		print("\nNote: gpu NOT available!")

	# define hyperparameters, etc.
	if isinstance(args.hyperparameters, float):
		alpha = args.hyperparameters
	elif type(args.hyperparameters) == list:
		eta1 = 0.05
		eta2 = 0.05 # not used for now

	if args.patch_loss == "bce":
		num_classes = 2

	setattr(args, "string_details", args.description)
	
	# MODEL INSTANTIATION 
	#=====================
    model = build_model(args)

	# OPTIMIZER INSTANTIATION
    # TODO: Add more optimizers if needed.
	#=========================
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

	# load state dicts
	# https://pytorch.org/tutorials/beginner/saving_loading_models.html
	if isinstance(args.model_to_load, str) and args.model_to_load.endswith(".sd"):
		print("Detected a previously trained model to continue training on! Initiating warm start from state dict...")
		checkpoint = torch.load(args.model_to_load)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.to(device)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		prev_epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		print("previous training loss:", loss)
		print("loading state dict for optimizer as well")

	setattr(args, "prev_epoch", prev_epoch)

	# TRAINING ROUTINE
	#==================

if __name__ == "__main__":
	main()


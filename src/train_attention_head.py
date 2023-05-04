from collections import OrderedDict
from functools import partial
import torch.nn as nn
from torchvision.models.vision_transformer import ConvStemConfig, Encoder
from torch import nn
from typing import Callable, List, Optional
import torch
from dataloader import EmbedDataset

from train import check_mb_accuracy
from embed_patches import train_on_Z
import math
import argparse
import torch.optim as optim
import wandb
from utils import deserialize, serialize
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc


# Constants/defaults
# -----------
print_every = 10
val_every = 20
# -----------

# Sizes for the different embeddings
EMBEDDING_SIZES = {"clip": 512, "plip": 512, "tile2vec": 128, "vit": 1024}


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

        if hasattr(self.heads, "pre_logits") and isinstance(
            self.heads.pre_logits, nn.Linear
        ):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(
                self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in)
            )
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
        z = z.reshape(n, dim, height * width)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        z = z.permute(0, 2, 1)

        if self.proj is not None:
            z = self.proj(z)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, z], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def get_device(args):
    print("GPU detected?", torch.cuda.is_available())
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nNote: gpu available & selected!")
    else:
        device = torch.device("cpu")
        print("\nNote: gpu NOT available!")
    return device


def build_model(args, num_classes):
    embedding_size = EMBEDDING_SIZES[args.embedding]

    return AttentionHead(
        height=124,
        width=124,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        input_dim=embedding_size,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        attention_dropout=args.dropout,
        num_classes=num_classes,
    )


def build_dataloader(args, is_train: bool):
    embedding_dir = {
        "clip": "Zs_clip",
        "plip": "Zs_plip",
        "vit": "Zs_vit",
        "tile2vec": "Zs",
    }[args.embedding]
    data_dir = os.path.join("/home/data/tinycam/train", embedding_dir)
    dataset = EmbedDataset(
        data_dir=data_dir,
        label_dict_path=args.labeldict_path,
        split_list=None,
        mode="Zs_full",
        kmeans_model=None,
        arm=None,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


def build_lr_scheduler(args):
    pass


def train_classifier(model, device, optimizer, args):
    """
    Train a model on image data using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    # Logging with Weights & Biases
    # -------------------------------
    os.environ["WANDB_MODE"] = "online"
    experiment = "AttentionMiner-" + args.embedding + "-" + "pathology"
    config = {
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        **vars(args),
    }
    wandb.init(project="selfsup-longrange", name=experiment, config=config)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    print("hyperparams:\n" + "=" * 30)
    print("Adam optimizer learn rate:", args.lr)
    print("Starting training procedure shortly...\n")

    if args.model_to_load is None:
        train_losses = []
    else:
        train_losses = deserialize(
            args.model_path + "/" + args.string_details + "_trainloss.obj"
        )  # same as train loss

    model = model.to(device=device)

    train_loader = build_dataloader(args, is_train=True)
    model.train()

    # files loaded differently per model class
    for e in range(0 + args.prev_epoch, args.num_epochs + args.prev_epoch):
        print("=" * 30 + "\n", "Beginning epoch", e, "\n" + "=" * 30)

        for t, (x, y) in enumerate(train_loader):
            # n, h, w, c -> n, c, h, w
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)

            train_loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print(
                    "Iteration %d, train loss = %.4f"
                    % (t + print_every, train_loss.item())
                )
                preds, probs, num_correct, num_samples = check_mb_accuracy(scores, y)
                acc = float(num_correct) / num_samples
                print("minibatch training accuracy: %.4f" % (acc * 100))
                # more logging
                wandb.log({"loss": train_loss})

            train_losses.append(train_loss.item())
            gc.collect()
            scheduler.step()

        # save model per epoch
        torch.save(
            model, args.model_path + "/" + args.string_details + "_epoch%s.pt" % e
        )
        # Future: check val acc every epoch

        # cache the losses every epoch
        serialize(
            train_losses, args.model_path + "/" + args.string_details + "_trainloss.obj"
        )
        # fig = plt.plot(train_losses, c="blue", label="train loss")
        # plt.savefig(
        #     args.model_path + "/" + args.string_details + "_trainloss.png",
        #     bbox_inches="tight",
        # )

        # more logging
        wandb.log({"end-of-epoch loss": train_loss})

        # save model per epoch
        print("saving model for epoch", e, "\n")
        torch.save(
            model, args.model_path + "/" + args.string_details + "_epoch%s.pt" % e
        )
        # serialize(train_losses, args.model_path + "/" + args.string_details + "_trainloss.obj")
        # fig = plt.plot(train_losses, c="blue", label="train loss")
        # plt.savefig(args.model_path + "/"  + args.string_details + "_trainloss.png", bbox_inches="tight")
        torch.save(
            {
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            args.model_path + "/" + args.string_details + "_epoch%s.sd" % e,
        )
        torch.save(
            {
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            args.model_path + "/" + args.string_details + ".sd",
        )
        # always keep a backup
        torch.save(
            {
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            args.model_path + "/BACKUP-" + args.string_details + ".sd",
        )

    # full model save
    torch.save(model, args.model_path + "/" + args.string_details + "_full.pt")
    return train_losses


def main():
    # ARGPARSE
    # ==========
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--description",
        default="no-description",
        type=str,
        help="Description of your experiement, with no spaces. E.g. VGG19_bn-random_loading-label_inherit-bce_loss-on_MFL-1",
    )
    parser.add_argument(
        "--model_class",
        default=None,
        type=str,
        help="Select one of: VGG19/VGG19_bn/VGG_att.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="What you want to name your dataset. For pre-defined label dictionaries, use: u54codex to search utils.",
    )
    parser.add_argument(
        "--dataloader_type",
        default="stored",
        type=str,
        help="Type of data loader: stored vs otf (on-the-fly).",
    )
    parser.add_argument(
        "--toy_flag",
        default=False,
        action="store_true",
        help="T/F for a smaller training dataset for rapid experimentation. Default is True.",
    )
    parser.add_argument(
        "--overfit_flag",
        default=False,
        action="store_true",
        help="T/F for intentional overfitting. Run name is modified to reflect this. Default is False.",
    )
    parser.add_argument("--use_gpu", action="store_true")

    # hyperparameters
    parser.add_argument(
        "--num_epochs",
        default=10,
        type=int,
        help="Number of epochs to train for. Default is 10.",
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size. Default is 2."
    )
    parser.add_argument(
        "--lr", default=5e-4, type=float, help="Learning rate. Default is 5e-4"
    )

    # model hyperparameters
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of attention layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--mlp_dim", type=int, default=64, help="feed forward mlp dim")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout")

    # parameters for patches
    parser.add_argument(
        "--loss",
        default="bce",
        type=str,
        help="Patch loss function. Default is bce. Future support for uncertainty.",
    )

    # paths
    parser.add_argument(
        "--embedding",
        choices=["clip", "plip", "tile2vec", "vit"],
        required=True,
        type=str,
        help="The images to use",
    )
    parser.add_argument(
        "--labeldict_path",
        default=None,
        type=str,
        help="Label dictionary path. This is a cached result of the preprocess.py script.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        required=True,
        type=str,
        help="Where you'd like to save the models.",
    )
    parser.add_argument(
        "--cache_path",
        default=None,
        type=str,
        help="Where you'd like to save the model outputs.",
    )
    parser.add_argument(
        "--model_to_load",
        default=None,
        type=str,
        help="starting point for model to load",
    )

    args = parser.parse_args()

    # PRINTS
    # ========
    print("\nBEGINNING TRAINING MODEL w/ Attention miner" + "\n" + "=" * 60)
    # print("Train set unique images:", len(label_dict))

    # SET-UP
    # ========
    device = get_device(args)

    setattr(args, "string_details", args.description)

    # MODEL INSTANTIATION
    # =====================
    model = build_model(args, num_classes=2)

    # OPTIMIZER INSTANTIATION
    # TODO: Add more optimizers if needed.
    # =========================
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load state dicts
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    prev_epoch = 0
    if isinstance(args.model_to_load, str) and args.model_to_load.endswith(".sd"):
        print(
            "Detected a previously trained model to continue training on! Initiating warm start from state dict..."
        )
        checkpoint = torch.load(args.model_to_load)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        prev_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print("previous training loss:", loss)
        print("loading state dict for optimizer as well")

    setattr(args, "prev_epoch", prev_epoch)

    # TRAINING ROUTINE
    # ==================
    train_classifier(model, device=device, optimizer=optimizer, args=args)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import os
from config import get_arguments
from models import ClusterNet, TAE
from load_data import get_loader
from sklearn.metrics import roc_auc_score


def pretrain_autoencoder(args, verbose=True):
    """
    function for the autoencoder pretraining
    """
    print("Pretraining autoencoder... \n")

    ## define TAE architecture
    tae = TAE(args)
    tae = tae.to(args.device)

    ## MSE loss
    loss_ae = nn.MSELoss()
    ## Optimizer
    optimizer = torch.optim.Adam(tae.parameters(), lr=args.lr_ae)
    tae.train()

    for epoch in range(args.epochs_ae):
        all_loss = 0
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            optimizer.zero_grad()
            z, x_reconstr = tae(inputs)
            loss_mse = loss_ae(inputs.squeeze(1), x_reconstr)

            loss_mse.backward()
            all_loss += loss_mse.item()
            optimizer.step()
        if verbose:
            print(
                "Pretraining autoencoder loss for epoch {} is : {}".format(
                    epoch, all_loss / (batch_idx + 1)
                )
            )

    print("Ending pretraining autoencoder. \n")
    # save weights
    torch.save(tae.state_dict(), args.path_weights_ae)


def initalize_centroids(X):
    """
    Function for the initialization of centroids.
    """
    X_tensor = torch.from_numpy(X).type(torch.FloatTensor).to(args.device)
    model.init_centroids(X_tensor)


def kl_loss_function(input, pred):
    out = input * torch.log((input) / (pred))
    return torch.mean(torch.sum(out, dim=1))


def train_ClusterNET(epoch, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    train_loss = 0
    all_preds, all_gt = [], []
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.type(torch.FloatTensor).to(args.device)
        all_gt.append(labels.cpu().detach())
        optimizer_clu.zero_grad()
        z, x_reconstr, Q, P = model(inputs)
        loss_mse = loss1(inputs.squeeze(), x_reconstr)
        loss_KL = kl_loss_function(P, Q)

        total_loss = loss_mse + loss_KL
        total_loss.backward()
        optimizer_clu.step()

        preds = torch.max(Q, dim=1)[1]
        all_preds.append(preds.cpu().detach())
        train_loss += total_loss.item()
    if verbose:
        print(
            "For epoch ",
            epoch,
            " Loss is : %.3f" % (train_loss / (batch_idx + 1)),
        )
    all_gt = torch.cat(all_gt, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return (
        preds.detach().cpu().numpy(),
        max(
            roc_auc_score(all_gt, all_preds),
            roc_auc_score(all_gt, 1 - all_preds),
        ),
        train_loss / (batch_idx + 1),
    )


def training_function(args, verbose=True):

    """
    function for training the DTC network.
    """
    ## initialize clusters centroids
    initalize_centroids(X_scaled)

    ## train clustering model
    max_roc_score = 0
    print("Training full model ...")
    for epoch in range(args.max_epochs):
        preds, roc_score, train_loss = train_ClusterNET(
            epoch, args, verbose=verbose
        )
        if roc_score > max_roc_score:
            max_roc_score = roc_score
            patience = 0
        else:
            patience += 1
            if patience == args.max_patience:
                break

    torch.save(model.state_dict(), args.path_weights_main)
    return max_roc_score


if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()
    args.path_data = args.path_data.format(args.dataset_name)
    if not os.path.exists(args.path_data):
        os.makedirs(args.path_data)

    path_weights = args.path_weights.format(args.dataset_name)
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)

    args.path_weights_ae = os.path.join(path_weights, "autoencoder_weight.pth")
    args.path_weights_main = os.path.join(
        path_weights, "full_model_weigths.pth"
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, X_scaled = get_loader(args)
    pretrain_autoencoder(args)

    model = ClusterNet(args)
    model = model.to(args.device)
    loss1 = nn.MSELoss()
    optimizer_clu = torch.optim.SGD(
        model.parameters(), lr=args.lr_cluster, momentum=args.momentum
    )

    max_roc_score = training_function(args)

    print("maximum roc score is {}".format(max_roc_score))

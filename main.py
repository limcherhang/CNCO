import torch
import argparse
import random

from func import _logger
import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Implementation Script")
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", default=0.0001, type=float, help="Learning Rate for training"
    )
    parser.add_argument(
        "--opt", default="CNCO", type=str, help="optimizer for training"
    )
    parser.add_argument("--epochs", default=50, type=int, help="num epochs")
    parser.add_argument(
        "--gamma", default=1, type=float, help="learning rate of ConOpt"
    )
    parser.add_argument(
        "--alpha", default=0.1, type=float, help="rho of Negative Curvature"
    )
    parser.add_argument(
        "--load_model_path",
        default="",
        type=str,
        help="load model path of our pretrained GAN model",
    )
    parser.add_argument("--seed", default=999, type=int, help="random seed number")
    parser.add_argument(
        "--is_fid", default=0, type=int, help="0= no count fid, 1= count fid"
    )
    parser.add_argument(
        "--is_eigen",
        default=0,
        type=int,
        help="is calculate eigenvalue?0=no plot, 1=plot. Note that --is_eigen only for SimGA and ConOpt",
    )
    parser.add_argument(
        "--generated_num",
        default=50,
        type=int,
        help="how many number of images generated, note that must <=batch_size",
    )
    parser.add_argument(
        "--fid_range",
        default=50,
        type=int,
        help="count FID score after fid_range iteration",
    )
    parser.add_argument(
        "--log_folder", default="./log", type=str, help="log folder path"
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    manualSeed = args.seed
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    log_folder = args.log_folder
    opt = args.opt
    logger = _logger.get_logger(log_folder, f"{opt}_batch_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_alpha_{args.alpha}_load_model_{args.load_model_path}_is_fid_{args.is_fid}_is_eigen_{args.is_eigen}.log")

    gan = model.GAN(args)
    gan.train()

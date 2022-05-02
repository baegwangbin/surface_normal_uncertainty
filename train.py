import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed

import utils.utils as utils
from utils.losses import compute_loss


def train(model, args, device):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    should_write = ((not args.distributed) or args.rank == 0)
    if should_write:
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # dataloader
    if args.dataset_name == 'nyu':
        from data.dataloader_nyu import NyuLoader
        train_loader = NyuLoader(args, 'train').data
        test_loader = NyuLoader(args, 'test').data
    else:
        raise Exception('invalid dataset name')

    # define losses
    loss_fn = compute_loss(args)

    # optimizer
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [{"params": m.get_1x_lr_params(), "lr": args.lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": args.lr}]
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay, lr=args.lr)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              max_lr=args.lr,
                                              epochs=args.n_epochs,
                                              steps_per_epoch=len(train_loader),
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)

    # cudnn setting
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    # start training
    total_iter = 0
    model.train()
    for epoch in range(args.n_epochs):
        if args.rank == 0:
            t_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train")
        else:
            t_loader = train_loader

        for data_dict in t_loader:
            optimizer.zero_grad()
            total_iter += args.batch_size_orig

            # data to device
            img = data_dict['img'].to(device)
            gt_norm = data_dict['norm'].to(device)
            gt_norm_mask = data_dict['norm_valid_mask'].to(device)

            # forward pass
            if args.use_baseline:
                norm_out = model(img)
                loss = loss_fn(norm_out, gt_norm, gt_norm_mask)
                norm_out_list = [norm_out]
            else:
                norm_out_list, pred_list, coord_list = model(img, gt_norm_mask=gt_norm_mask, mode='train')
                loss = loss_fn(pred_list, coord_list, gt_norm, gt_norm_mask)

            loss_ = float(loss.data.cpu().numpy())
            if args.rank == 0:
                t_loader.set_description(f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train. Loss: {'%.5f' % loss_}")
                t_loader.refresh()

            # back-propagate
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # lr scheduler
            scheduler.step()

            # visualize
            if should_write and ((total_iter % args.visualize_every) < args.batch_size_orig):
                utils.visualize(args, img, gt_norm, gt_norm_mask, norm_out_list, total_iter)

            # save model
            if should_write and ((total_iter % args.validate_every) < args.batch_size_orig):
                model.eval()
                target_path = args.exp_model_dir + '/checkpoint_iter_%010d.pt' % total_iter
                torch.save({"model": model.state_dict(),
                            "iter": total_iter}, target_path)
                print('model saved / path: {}'.format(target_path))
                validate(model, args, test_loader, device, total_iter, args.eval_acc_txt)
                model.train()

                # empty cache
                torch.cuda.empty_cache()

    if should_write:
        model.eval()
        target_path = args.exp_model_dir + '/checkpoint_iter_%010d.pt' % total_iter
        torch.save({"model": model.state_dict(),
                    "iter": total_iter}, target_path)
        print('model saved / path: {}'.format(target_path))
        validate(model, args, test_loader, device, total_iter, args.eval_acc_txt)

        # empty cache
        torch.cuda.empty_cache()

    return model


__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
def _unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out

    
def validate(model, args, test_loader, device, total_iter, where_to_write, vis_dir=None):
    with torch.no_grad():
        total_normal_errors = None
        for data_dict in tqdm(test_loader, desc="Loop: Validation"):

            # data to device
            img = data_dict['img'].to(device)
            gt_norm = data_dict['norm'].to(device)
            gt_norm_mask = data_dict['norm_valid_mask'].to(device)

            # forward pass
            if args.use_baseline:
                norm_out = model(img)
            else:
                norm_out_list, _, _ = model(img, gt_norm_mask=gt_norm_mask, mode='test')
                norm_out = norm_out_list[-1]

            # upsample if necessary
            if norm_out.size(2) != gt_norm.size(2):
                norm_out = F.interpolate(norm_out, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)

            pred_norm = norm_out[:, :3, :, :]  # (B, 3, H, W)
            pred_kappa = norm_out[:, 3:, :, :]  # (B, 1, H, W)

            prediction_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
            prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
            E = torch.acos(prediction_error) * 180.0 / np.pi

            mask = gt_norm_mask[:, 0, :, :]
            if total_normal_errors is None:
                total_normal_errors = E[mask]
            else:
                total_normal_errors = torch.cat((total_normal_errors, E[mask]), dim=0)

        total_normal_errors = total_normal_errors.data.cpu().numpy()
        metrics = utils.compute_normal_errors(total_normal_errors)
        utils.log_normal_errors(metrics, where_to_write, first_line='total_iter: {}'.format(total_iter))
        return metrics


# main worker
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # define model
    if args.use_baseline:
        from models.baseline import NNET
    else:
        from models.NNET import NNET
    model = NNET(args)

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    train(model, args, device=args.gpu)


if __name__ == '__main__':
    # Arguments ########################################################################################################
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    # directory
    parser.add_argument('--exp_dir', default='./experiments', type=str, help='directory to store experiment results')
    parser.add_argument('--exp_name', default='exp00_test', type=str, help='experiment name')
    parser.add_argument('--visible_gpus', default='01', type=str, help='gpu to use')

    # model architecture
    parser.add_argument('--architecture', default='GN', type=str, help='{BN, GN}')
    parser.add_argument("--use_baseline", action="store_true", help='use baseline encoder-decoder (no pixel-wise MLP, no uncertainty-guided sampling')
    parser.add_argument('--sampling_ratio', default=0.4, type=float)
    parser.add_argument('--importance_ratio', default=0.7, type=float)

    # loss function
    parser.add_argument('--loss_fn', default='UG_NLL_ours', type=str, help='{L1, L2, AL, NLL_vMF, NLL_ours, UG_NLL_vMF, UG_NLL_ours}')

    # training
    parser.add_argument('--n_epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--validate_every', default=5000, type=int, help='validation period')
    parser.add_argument('--visualize_every', default=1000, type=int, help='visualization period')
    parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")
    parser.add_argument("--workers", default=12, type=int, help="Number of workers for data loading")

    # optimizer setup
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--lr', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--same_lr', default=False, action="store_true", help="Use same LR for all param groups")
    parser.add_argument('--grad_clip', default=0.1, type=float)
    parser.add_argument('--div_factor', default=25.0, type=float, help="Initial div factor for lr")
    parser.add_argument('--final_div_factor', default=10000.0, type=float, help="final div factor for lr")

    # dataset
    parser.add_argument("--dataset_name", default='nyu', type=str, help="{nyu, scannet}")

    # dataset - preprocessing
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)

    # dataset - augmentation
    parser.add_argument("--data_augmentation_color", default=True, action="store_true")
    parser.add_argument("--data_augmentation_hflip", default=True, action="store_true")
    parser.add_argument("--data_augmentation_random_crop", default=False, action="store_true")

    # read arguments from txt file
    if sys.argv.__len__() == 2 and '.txt' in sys.argv[1]:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.num_threads = args.workers
    args.mode = 'train'

    # create experiment directory
    args.exp_dir = args.exp_dir + '/{}/'.format(args.exp_name)
    args.exp_model_dir = args.exp_dir + '/models/'    # store model checkpoints
    args.exp_vis_dir = args.exp_dir + '/vis/'         # store training images
    args.exp_log_dir = args.exp_dir + '/log/'         # store log
    utils.make_dir_from_list([args.exp_dir, args.exp_model_dir, args.exp_vis_dir, args.exp_log_dir])
    print(args.exp_dir)

    utils.save_args(args, args.exp_log_dir + '/params.txt')  # save experiment parameters
    args.eval_acc_txt = args.exp_log_dir + '/eval_acc.txt'

    # train
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(args.visible_gpus))

    args.world_size = 1
    args.rank = 0
    nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    args.batch_size_orig = args.batch_size

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
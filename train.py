from data_utils.dataset import SoundDataset
from utils.utils import AverageMeter, save_model, Logger
from networks.damas_fista_net import DAMAS_FISTA_Net
from utils.config import Config
from utils.utils import get_search_freq, neighbor_2_zero, find_match_source

import argparse
import sys
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import time
import math


def parse_option():
    # Parameter Setting
    parser = argparse.ArgumentParser(description='Training the DAMAS-FISTA-Net for Real-time Acoustic Beamforming')

    # Dataset/Folder Parameter
    parser.add_argument('--print_freq', type=int, default=1, help='Frequency of print statements')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of model saves')
    parser.add_argument('--train_dir', help='Directory for training data', default='./data_split/One_train.txt', type=str)
    parser.add_argument('--test_dir', help='Directory for testing data', default='./data_split/One_test.txt', type=str)
    parser.add_argument('--label_dir', help='Directory for data label', default='./dataset/label/', type=str)
    parser.add_argument('--save_folder', dest='save_folder', help='Directory to save models', default='./save_models/', type=str)

    # Network Parameter
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='Starting epoch number (useful for restarts)')
    parser.add_argument('--val_epochs', default=20, type=int, metavar='N', help='Number of epochs between validations')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='Total number of epochsn')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for the dataloader')
    parser.add_argument('--learning_rate', '--learning-rate', default=1e-3, type=float, metavar='LR', help='Initial learning rate')
    parser.add_argument('--MultiStepLR', action='store_true', help='Use MultiStepLR scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum factor')
    parser.add_argument('--weight_decay', '--wd', default=1e-2, type=float, metavar='W', help='Weight decay (default: 1e-2)')
    parser.add_argument('--LayNo', default=5, type=int, help='Number of iteration layers')

    # Acoustic Parameter (Pre-calculate)
    parser.add_argument('--micro_array_path', default='./data_precomputed/56_spiral_array.mat', type=str, help='Path to microphone array file')
    parser.add_argument('--wk_reshape_path', default='./data_precomputed/wk_reshape.npy', type=str, help='Path to reshaped wk_reshape file')
    parser.add_argument('--A_path', default='./data_precomputed/A.npy', type=str, help='Path to A matrix file')
    parser.add_argument('--ATA_path', default='./data_precomputed/ATA.npy', type=str, help='Path to ATA matrix file')
    parser.add_argument('--L_path', default='./data_precomputed/ATA_eigenvalues.npy', type=str, help='Path to eigenvalues file')
    parser.add_argument('--more_source', action='store_true', help='Enable multiple sound sources')
    parser.add_argument('--config', default='./utils/config.yml', type=str, help='Path to configuration file')
    parser.add_argument('--source_num', default=1, type=int, help='Number of sources')

    args = parser.parse_args()
    record_time = time.localtime(time.time())

    return record_time, args


def set_loader(args, config):
    # Training data loader
    train_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.train_dir, args.label_dir, config['z_dist'], args.micro_array_path, args.wk_reshape_path,
                     args.A_path, args.ATA_path, args.L_path, args.frequencies, args.config,
                     more_source=args.more_source, train=True), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Testing data loader
    test_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.test_dir, args.label_dir, config['z_dist'], args.micro_array_path, args.wk_reshape_path,
                     args.A_path, args.ATA_path, args.L_path, args.frequencies, args.config,
                     more_source=args.more_source), batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    return train_dataloader, test_dataloader


def set_model(args):
    # Model Loading
    model = DAMAS_FISTA_Net(args.LayNo)

    if torch.cuda.is_available():
        model.cuda()
        cudnn.benchmark = True

    return model


def set_optimizer(args, model):
    # Adam Optimizer Setting
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)

    return optimizer


def adjust_learning_rate(args, optimizer):
    # Define the strategy for adjusting the learning rate
    if args.MultiStepLR:
        # Decreasing learning rate
        args.learning_rate *= 0.95

        # Update the learning rate for each parameter group in the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
            print('lr=', param_group['lr'])


def train(train_dataloader, model, optimizer, epoch, args):
    # Training your network
    model.train()

    losses = AverageMeter()

    for idx, (CSM_K, wk_reshape_K, A_K, ATA_K, L_K, _, label, _) in enumerate(train_dataloader):

        if torch.cuda.is_available():
            CSM_K = CSM_K.cuda(non_blocking=True)
            wk_reshape_K = wk_reshape_K.cuda(non_blocking=True)
            A_K = A_K.cuda(non_blocking=True)
            ATA_K = ATA_K.cuda(non_blocking=True)
            L_K = torch.Tensor(L_K).cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        # Compute loss
        output = model(CSM_K, wk_reshape_K, A_K, ATA_K, L_K)
        loss = torch.sum((output - label) ** 2)
        bsz = label.shape[0]
        losses.update(loss.item(), bsz)

        # Adam optimizer update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Showing train information
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss={loss.val:.8f} \t'
                  'mean={loss.avg:.8f}\t'.format(epoch, idx + 1, len(train_dataloader), loss=losses))

        sys.stdout.flush()


def test(test_dataloader, model, args, config):
    # Testing your network
    model.eval()

    # Record list
    location_bias_list = list()
    power_bias_list = list()
    time_list = list()

    # Define scanning area
    scanning_area_X = np.arange(config['scan_x'][0], config['scan_x'][1] + config['scan_resolution'],
                                config['scan_resolution'])
    scanning_area_Y = np.arange(config['scan_y'][0], config['scan_y'][1] + config['scan_resolution'],
                                config['scan_resolution'])

    # Start Testing: Model Inference
    with torch.no_grad():

        for idx, (CSM, wk_reshape, A, ATA, L, label, _) in enumerate(test_dataloader):

            # Parameter Loading
            if torch.cuda.is_available():
                CSM = CSM.cuda(non_blocking=True)
                wk_reshape = wk_reshape.cuda(non_blocking=True)
                A = A.cuda(non_blocking=True)
                ATA = ATA.cuda(non_blocking=True)
                L = L.cuda(non_blocking=True)

            output = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()

            # Start scan-frequency beamforming
            for K in range(len(args.frequencies)):

                # Setting parameters corresponding to the frequency K
                CSM_K = CSM[:, :, :, K]
                wk_reshape_K = wk_reshape[:, :, :, K]
                A_K = A[:, :, :, K]
                ATA_K = ATA[:, :, :, K]
                L_K = L[:, K]
                L_K = torch.unsqueeze(L_K, 1).to(torch.float64)
                L_K = torch.unsqueeze(L_K, 2).to(torch.float64)

                # Accumulate the frequency components
                if output is None:
                    output = model(CSM_K, wk_reshape_K, A_K, ATA_K, L_K)
                else:
                    output += model(CSM_K, wk_reshape_K, A_K, ATA_K, L_K)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()

            # Record time
            now_time = end_time - start_time
            time_list.append(now_time)

            # Loading label and model output
            b_gt = label.cpu().numpy()
            b_hat = output.cpu().numpy()
            b_gt = np.squeeze(np.squeeze(b_gt, 0), 1)
            b_hat = np.squeeze(np.squeeze(b_hat, 0), 1)

            # Initialize the true and estimated locations
            L_hat = np.zeros((args.source_num, 3))
            L_gt = np.zeros((args.source_num, 3))

            # Find the source location
            for i in range(args.source_num):

                # Coordinate of the ground truth
                max_gt = max(b_gt)
                b_gt_index = np.where(b_gt == max_gt)[0][0]

                y_hat_pos = math.ceil((b_gt_index + 1) / len(scanning_area_X))
                x_hat_pos = (b_gt_index + 1) - (y_hat_pos - 1) * len(scanning_area_X)
                x_gt = scanning_area_X[x_hat_pos - 1]
                y_gt = scanning_area_Y[y_hat_pos - 1]

                # Coordinate of the estimated
                max_hat = max(b_hat)
                b_hat_index = np.where(b_hat == max_hat)[0][0]

                y_hat_pos = math.ceil((b_hat_index + 1) / len(scanning_area_X))
                x_hat_pos = (b_hat_index + 1) - (y_hat_pos - 1) * len(scanning_area_X)
                x_hat = scanning_area_X[x_hat_pos - 1]
                y_hat = scanning_area_Y[y_hat_pos - 1]

                # Estimated location
                L_hat[i][0] = x_hat
                L_hat[i][1] = y_hat
                L_hat[i][2] = max_hat

                # True location
                L_gt[i][0] = x_gt
                L_gt[i][1] = y_gt
                L_gt[i][2] = max_gt

                # Reconstruct the estimated and true beamforming map for bias calculation
                map_gt = b_gt.reshape(41, 41, order='F')
                map_hat = b_hat.reshape(41, 41, order='F')

                map_gt = neighbor_2_zero(map_gt, x_hat_pos - 1, y_hat_pos - 1)
                map_hat = neighbor_2_zero(map_hat, x_hat_pos - 1, y_hat_pos - 1)

                b_gt = map_gt.reshape(map_gt.size, order='F')
                b_hat = map_hat.reshape(map_hat.size, order='F')

            temp_gt_mat = L_gt

            # Find match source and calculate the bias
            for i in range(args.source_num):

                _, location_bias, power_bias, temp_gt_mat = find_match_source(L_hat[i], temp_gt_mat)
                location_bias_list.append(location_bias)
                power_bias_list.append(power_bias)

        print("Test: \t location_bias={:.4f} \t power_bias={:.4f}\t time={:.4f}"
              .format(np.mean(location_bias_list), np.mean(power_bias_list), np.mean(time_list)))


def main():
    # Parameter Setting
    record_time, args = parse_option()
    args.save_folder = args.save_folder + '{}/'.format(time.strftime('%m-%d-%H-%M', record_time))
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    sys.stdout = Logger(args.save_folder + "log.txt")

    # Select the frequency points in the scanning frequency band
    con = Config(args.config).getConfig()['base']
    _, _, _, _, args.frequencies = get_search_freq(con['N_total_samples'], con['scan_low_freq'], con['scan_high_freq'], con['fs'])

    # Build data loader
    train_dataloader, test_dataloader = set_loader(args, con)

    # Build model and criterion
    model = set_model(args)

    # Build optimizer
    optimizer = set_optimizer(args, model)

    # Start training...
    print('===========================================')
    print('Start Training the DAMAS-FISTA-Net...')
    print('===> Start Epoch {} to End Epoch {}'.format(args.start_epoch, args.epochs))

    # Training routine
    for epoch in range(1, args.epochs + 1):
        # Learning rate Strategy
        if args.MultiStepLR and epoch != 1:
            adjust_learning_rate(args, optimizer)

        # Train for one epoch
        time1 = time.time()
        train(train_dataloader, model, optimizer, epoch, args)
        time2 = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # Save model
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_epoch_{epoch}.pt'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

        # Evaluation
        if epoch % args.val_epochs == 0:
            test(test_dataloader, model, args, con)

    # Save the last model
    save_file = os.path.join(args.save_folder, 'last.pt')
    save_model(model, optimizer, args, args.epochs, save_file)


if __name__ == '__main__':
    main()

from networks.damas_fista_net import DAMAS_FISTA_Net
from utils.utils import pyContourf, Logger
from data_utils.dataset import SoundDataset
import numpy as np
import torch
import argparse
import os
import sys
import h5py
from utils.config import Config
import math
import time
from utils.utils import get_search_freq, neighbor_2_zero, find_match_source


def parse_option():
    parser = argparse.ArgumentParser(description='Testing the DAMAS-FISTA-Net for Real-time Acoustic Beamforming')

    # Dataset/Folder Parameter
    parser.add_argument('--test_dir', help='Directory for testing data', default='./data_split/One_test.txt', type=str)
    parser.add_argument('--vis_dir', help='Directory for saving the visualization results', default='./vis_results/', type=str)
    parser.add_argument('--output_dir', help='Directory for saving the output', default='./output_results/', type=str)
    parser.add_argument('--label_dir', help='Directory for data label', default='./dataset/label/', type=str)
    parser.add_argument('--ckpt', type=str, default='./save_models/08-20-15-16/last.pt', help='Path to the trained model')
    parser.add_argument('--show_pre_imaging_result', action='store_true', help='Whether show pre-imaging result')

    # Acoustic Parameter (Pre-calculate)
    parser.add_argument('--LayNo', default=5, type=int, help='Number of iteration layers')
    parser.add_argument('--micro_array_path', default='./data_precomputed/56_spiral_array.mat', type=str, help='Path to microphone array file')
    parser.add_argument('--wk_reshape_path', default='./data_precomputed/wk_reshape.npy', type=str, help='Path to reshaped wk_reshape file')
    parser.add_argument('--A_path', default='./data_precomputed/A.npy', type=str, help='Path to A matrix file')
    parser.add_argument('--ATA_path', default='./data_precomputed/ATA.npy', type=str, help='Path to ATA matrix file')
    parser.add_argument('--L_path', default='./data_precomputed/ATA_eigenvalues.npy', type=str, help='Path to eigenvalues file')
    parser.add_argument('--more_source', action='store_true', help='Enable multiple sound sources')
    parser.add_argument('--config', default='./utils/config.yml', type=str, help='Path to configuration file')
    parser.add_argument('--source_num', default=1, type=int, help='Number of sources')

    # Noise Parameter
    parser.add_argument('--add_noise', action='store_true', help='whether add gaussian noise to test')
    parser.add_argument('--dB_value', default=0, type=float, help='gaussian noise value')

    args = parser.parse_args()

    # Construct vis and output folder
    args.vis_dir += args.ckpt.split('/')[-2] + '/' + args.ckpt.split('/')[-1].split('.')[0] + '/'
    if not os.path.exists(args.vis_dir):
            os.makedirs(args.vis_dir)

    args.output_dir += args.ckpt.split('/')[-2] + '/' + args.ckpt.split('/')[-1].split('.')[0] + '/'
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    return args


def set_loader(args, config):
    # Testing data loader
    test_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.test_dir, args.label_dir, config['z_dist'], args.micro_array_path, args.wk_reshape_path, args.A_path, args.ATA_path, args.L_path, None, args.config, add_noise=args.add_noise, dB_value=args.dB_value, more_source=args.more_source),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return test_dataloader


def set_model(args):
    # Model Loading
    if not args.show_pre_imaging_result:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        state_dict = ckpt['model']

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

        model = DAMAS_FISTA_Net(args.LayNo)

        if torch.cuda.is_available():
            model = model.cuda()

        model.load_state_dict(state_dict)

    else:
        model = DAMAS_FISTA_Net(args.LayNo, args.show_pre_imaging_result)

        if torch.cuda.is_available():
            model = model.cuda()

    return model


def test(test_dataloader, model, args, config):
    # Testing your network
    model.eval()

    # Record list
    location_bias_list = list()
    power_bias_list = list()
    time_list = list()

    # Define scanning area
    scanning_area_X = np.arange(config['scan_x'][0], config['scan_x'][1] + config['scan_resolution'], config['scan_resolution'])
    scanning_area_Y = np.arange(config['scan_y'][0], config['scan_y'][1] + config['scan_resolution'], config['scan_resolution'])

    # Start Testing: Model Inference
    with torch.no_grad():

        for idx, (CSM, wk_reshape, A, ATA, L, label, sample_name) in enumerate(test_dataloader):

            # Parameter Loading
            sample_name = sample_name[0]
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

            # Visualization results
            vis_gt = b_gt.reshape(41, 41, order='F')
            vis_hat = b_hat.reshape(41, 41, order='F')
            f = h5py.File(args.output_dir + sample_name + '.h5','w')
            f['damas_fista_net_output'] = vis_hat
            f.close()
            pyContourf(vis_hat, vis_gt, args.vis_dir, sample_name)

            b_gt = np.squeeze(np.squeeze(b_gt, 0), 1)
            b_hat = np.squeeze(np.squeeze(b_hat, 0), 1)


            # Reconstruct the estimated and true beamforming map for bias calculation
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

                map_gt = neighbor_2_zero(map_gt, x_hat_pos-1, y_hat_pos-1)
                map_hat = neighbor_2_zero(map_hat, x_hat_pos-1, y_hat_pos-1)

                b_gt = map_gt.reshape(map_gt.size, order='F')
                b_hat = map_hat.reshape(map_hat.size, order='F')
             
            temp_gt_mat = L_gt

            # Find match source and calculate the bias
            for i in range(args.source_num):

                _, location_bias, power_bias, temp_gt_mat = find_match_source(L_hat[i], temp_gt_mat)
                location_bias_list.append(location_bias)
                power_bias_list.append(power_bias)

            print("Test: [{}/{}]\t location_bias={:.4f} \t power_bias={:.4f}\t time={:.4f}"
                  .format(idx + 1, len(test_dataloader), location_bias, power_bias, now_time))

        print('===========================================')
        print('Finish Testing the DAMAS-FISTA-Net...')
        print("Test: \t location_bias={:.4f} \t power_bias={:.4f}\t time={:.4f}"
              .format(np.mean(location_bias_list), np.mean(power_bias_list), np.mean(time_list)))
      

def main():
    # Parameter Setting
    args = parse_option()
    sys.stdout = Logger(args.vis_dir + "log.txt")

    # Select the frequency points in the scanning frequency ban
    con = Config(args.config).getConfig()['base']
    _, _, _, _, args.frequencies = get_search_freq(con['N_total_samples'], con['scan_low_freq'], con['scan_high_freq'], con['fs'])

    # Build data loader
    test_dataloader = set_loader(args, con)

    # Build model and criterion
    model = set_model(args)

    # Start testing...
    print('===========================================')
    print('Start Testing the DAMAS-FISTA-Net...')
    test(test_dataloader, model, args, con)

if __name__ == '__main__':
    main()
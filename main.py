import os
import csv
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence
from dataset import get_data_dict
from dataset import VideoFeatureDataset
from model import ASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter


class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params,
                 event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device, all_params):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess
        self.params = all_params

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting,
              soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq,
              log_train_results=True,
              early_stopping_patience=None, early_stopping_min_delta=0.001):  # Early Stopping 파라미터 추가

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')

        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)

        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)

        # CSV 파일 경로 설정
        csv_file_path = os.path.join(result_dir, f"{self.params['naming']}_results.csv")

        # CSV 헤더 작성 (파일이 존재하지 않을 경우)
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Dataset', 'Acc', 'Edit', 'F1@10', 'F1@25', 'F1@50'])

        # --- Early Stopping 변수 초기화 ---
        best_score = float('-inf')  # F1 스코어는 높을수록 좋으므로 -inf로 초기화
        patience_counter = 0
        best_model_state = None
        # ------------------------------------

        for epoch in range(restore_epoch + 1, num_epochs):

            self.model.train()

            epoch_running_loss = 0

            for _, data in enumerate(train_train_loader):

                feature, label, boundary, video = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)

                loss_dict = self.model.get_training_loss(feature,
                    event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion,
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                total_loss = 0

                for k, v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k, v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()

                epoch_running_loss += total_loss.item()

                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1

            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')

            if result_dir:
                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                if result_dir:
                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')

                # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
                for mode in ['decoder-agg']:  # Default: decoder-agg. The results of decoder-noagg are similar

                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None)

                    # TensorBoard 로깅 (기존 코드)
                    if result_dir:
                        for k, v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir,
                                             f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    for k, v in test_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Test-{k} {v}')

                    # --- CSV 파일로 결과 저장 로직 추가 (test_test_dataset 결과) ---
                    with open(csv_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            epoch + 1,
                            f'{mode}-Test',
                            test_result_dict['Acc'],
                            test_result_dict['Edit'],
                            test_result_dict['F1@10'],
                            test_result_dict['F1@25'],
                            test_result_dict['F1@50']
                        ])
                    # ---------------------------------------------------

                    # --- Early Stopping 로직 적용 ---
                    current_score = test_result_dict['F1@25']  # F1@25를 모니터링 지표로 사용

                    if current_score > best_score + early_stopping_min_delta:
                        best_score = current_score
                        patience_counter = 0
                        best_model_state = copy.deepcopy(self.model.state_dict())  # 최적 모델 상태 저장
                        print(f"Epoch {epoch + 1}: Validation F1@25 improved to {best_score:.4f}. Saving best model.")
                    else:
                        patience_counter += 1
                        print(
                            f"Epoch {epoch + 1}: Validation F1@25 did not improve. Patience: {patience_counter}/{early_stopping_patience}")

                    if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                        print(
                            f"Early stopping triggered at epoch {epoch + 1}. No improvement for {early_stopping_patience} epochs.")
                        if best_model_state is not None:
                            # 최적 모델 로드 및 저장 (선택 사항: 나중에 사용하기 위해)
                            self.model.load_state_dict(best_model_state)
                            torch.save(self.model.state_dict(), f'{result_dir}/best_early_stop_model.model')
                            print(f"Best model saved to {result_dir}/best_early_stop_model.model")
                        break  # 학습 루프 종료
                    # ------------------------------------

                    if log_train_results:

                        train_result_dict = self.test(
                            train_test_dataset, mode, device, label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k, v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)

                            np.save(os.path.join(result_dir,
                                                 f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)

                        for k, v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')

                        # --- CSV 파일로 결과 저장 로직 추가 (train_test_dataset 결과) ---
                        with open(csv_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                epoch + 1,
                                f'{mode}-Train',
                                train_result_dict['Acc'],
                                train_result_dict['Edit'],
                                train_result_dict['F1@10'],
                                train_result_dict['F1@25'],
                                train_result_dict['F1@50']
                            ])
                        # ---------------------------------------------------
            # Early stopping이 트리거되면 바깥 for 루프도 종료되어야 함
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                break

        if result_dir:
            logger.close()

    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):

        assert (test_dataset.mode == 'test')
        assert (mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert (self.postprocess['type'] in ['median', 'mode', 'purge', None])

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None

        with torch.no_grad():

            feature, label, _, video = test_dataset[video_idx]

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device))
                          for i in range(len(feature))]  # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output = [self.model.ddim_sample(feature[i].to(device), seed)
                          for i in range(len(feature))]  # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [
                    self.model.ddim_sample(feature[len(feature) // 2].to(device), seed)]  # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert (output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:, :, :min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()

            if self.postprocess['type'] == 'median':  # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)

            output = restore_full_sequence(output,
                                           full_len=label.shape[-1],
                                           left_offset=left_offset,
                                           right_offset=right_offset,
                                           sample_rate=self.sample_rate
                                           )

            if self.postprocess['type'] == 'mode':  # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)

                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:

                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e + 1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e - 1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e - 1]
                            output[mid:ends[e]] = trans[e + 1]

            label = label.squeeze(0).cpu().numpy()

            assert (output.shape == label.shape)

            return video, output, label

    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None):

        assert (test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):

                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                pred = [self.event_list[int(i)] for i in pred]

                if result_dir:  # result_dir이 None이 아닐 때만 폴더 생성 및 파일 쓰기
                    if not os.path.exists(os.path.join(result_dir, 'prediction')):
                        os.makedirs(os.path.join(result_dir, 'prediction'))

                    file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                    file_ptr = open(file_name, 'w')
                    file_ptr.write('### Frame level recognition: ###\n')
                    file_ptr.write(' '.join(pred))
                    file_ptr.close()

        acc, edit, f1s = func_eval(
            label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list)

        result_dict = {
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2]
        }

        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    # Early Stopping 파라미터 추가
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Number of epochs to wait for improvement before stopping. Set to None to disable.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                        help='Minimum change to qualify as an improvement for early stopping.')

    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    feature_dir = os.path.join(root_data_dir, dataset_name, 'features')
    label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'train.split{split_id}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'test.split{split_id}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    train_data_dict = get_data_dict(
        feature_dir=feature_dir,
        label_dir=label_dir,
        video_list=train_video_list,
        event_list=event_list,
        sample_rate=sample_rate,
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    test_data_dict = get_data_dict(
        feature_dir=feature_dir,
        label_dir=label_dir,
        video_list=test_video_list,
        event_list=event_list,
        sample_rate=sample_rate,
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    # 결과 저장 경로를 /home/user/Downloads/result로 설정
    custom_result_dir = "/home/user/Downloads/result"

    # 결과 디렉토리가 없으면 생성
    if not os.path.exists(custom_result_dir):
        os.makedirs(custom_result_dir)

    # Trainer 초기화 시 all_params 인자 추가
    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params),
                      event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      all_params=all_params  # all_params 인자 전달
                      )

    # Trainer의 train 메서드에 custom_result_dir 및 early stopping 파라미터 전달
    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset,
                  loss_weights, class_weighting, soft_label,
                  num_epochs, batch_size, learning_rate, weight_decay,
                  label_dir=label_dir, result_dir=os.path.join(custom_result_dir, naming),
                  log_freq=log_freq, log_train_results=log_train_results,
                  early_stopping_patience=args.early_stopping_patience,  # arg로 받은 값 전달
                  early_stopping_min_delta=args.early_stopping_min_delta  # arg로 받은 값 전달
                  )

import os
import json
import copy

params_gtea = {
    "naming": "default",
    "root_data_dir": "./datasets",
    "dataset_name": "gtea",
    "split_id": 1,
    "sample_rate": 1,
    "temporal_aug": True,
    "encoder_params": {
        "use_instance_norm": False,
        "num_layers": 10,
        "num_f_maps": 64,
        "input_dim": 2048,
        "kernel_size": 5,
        "normal_dropout_rate": 0.5,
        "channel_dropout_rate": 0.5,
        "temporal_dropout_rate": 0.5,
        "feature_layer_indices": [
            5,
            7,
            9
        ]
    },
    "decoder_params": {
        "num_layers": 8,
        "num_f_maps": 24,
        "time_emb_dim": 512,
        "kernel_size": 5,
        "dropout_rate": 0.1,
    },
    "diffusion_params": {
        "timesteps": 1000,
        "sampling_timesteps": 25,
        "ddim_sampling_eta": 1.0,
        "snr_scale": 0.5,
        "cond_types": ['full', 'zero', 'boundary03-', 'segment=1', 'segment=1'],
        "detach_decoder": False,
    },
    "loss_weights": {
        "encoder_ce_loss": 0.5,
        "encoder_mse_loss": 0.1,
        "encoder_boundary_loss": 0.0,
        "decoder_ce_loss": 0.5,
        "decoder_mse_loss": 0.1,
        "decoder_boundary_loss": 0.1
    },
    "batch_size": 4,
    "learning_rate": 0.0005,
    "weight_decay": 1e-6,
    "num_epochs": 10001,
    "log_freq": 100,
    "class_weighting": True,
    "set_sampling_seed": True,
    "boundary_smooth": 1,
    "soft_label": 1.4,
    "log_train_results": False,
    "postprocess": {
        "type": "purge",
        "value": 3
    },
}

params_50salads = {
    "naming": "default",
    "root_data_dir": "./datasets",
    "dataset_name": "50salads",
    "split_id": 1,
    "sample_rate": 8,  # 입력 시퀀스의 시간 해상도 조절 -> 현재 값 => 초당 8프레임 샘플링
    "temporal_aug": True,
    "encoder_params": {
        "use_instance_norm": False,
        "num_layers": 10,  # 인코더 Depth -> underfitting 발생 시 깊이 늘리는 것도 방법
        "num_f_maps": 64,  # 위와 동일
        "input_dim": 2048,
        "kernel_size": 5,
        "normal_dropout_rate": 0.5,  # 모델 과적합 경향 있을 경우 정규화 강도 높일 수 있음
        "channel_dropout_rate": 0.5,
        "temporal_dropout_rate": 0.5,
        "feature_layer_indices": [
            5,
            7,
            9
        ]
    },
    "decoder_params": {
        "num_layers": 8,
        "num_f_maps": 24,
        "time_emb_dim": 512,
        "kernel_size": 7,
        "dropout_rate": 0.1,
    },
    "diffusion_params": {
        "timesteps": 1000,  # 늘리면 정교한 노이즈 제거 가능하지만 학습시간 길어짐
        "sampling_timesteps": 25,  # 값을 늘리면 품질과 정확도 향상 가능성 있음 -> F1 score, accuracy 영향
        "ddim_sampling_eta": 1.0,  # 0.0으로 바꿀경우 예측의 일관성과 안정성 향상으로 F1 score, accuracy 향상 가능성 존재
        "snr_scale": 1.0,
        "cond_types": [
            "full",
            "zero",
            "boundary05-",
            "segment=2",
            "segment=2"
        ],
        "detach_decoder": False,
    },
    "loss_weights": {
        "encoder_ce_loss": 0.5,
        "encoder_mse_loss": 0.1,
        "encoder_boundary_loss": 0.0,
        "decoder_ce_loss": 0.5,
        "decoder_mse_loss": 0.1,
        "decoder_boundary_loss": 0.1
    },
    "batch_size": 16,  # 32 or 64로 늘리면 한 번의 업데이트에 많은 샘플 사용으로 기울기 추정 안정화 가능 -> 학습속도 향상
    "learning_rate": 0.0001,  # 비교적 낮은 값으로 올리는 것도 방법
    "weight_decay": 0.0001,  # L2 Normalization 강도 결정 -> 과적합 방지용
    "num_epochs": 5001,
    "log_freq": 100,
    "class_weighting": True,
    "set_sampling_seed": True,
    "boundary_smooth": 20,
    "soft_label": None,
    "log_train_results": False,
    "postprocess": {
        "type": "median",  # W , 예측 후 결과 다듬는 단계 => value 값 조절을 통해 최적 값 찾는 것도 방법
        "value": 50  # W
    },
}

params_breakfast = {
    "naming": "default",
    "root_data_dir": "./datasets",
    "dataset_name": "breakfast",
    "split_id": 1,
    "sample_rate": 1,
    "temporal_aug": True,
    "encoder_params": {
        "use_instance_norm": False,
        "num_layers": 12,
        "num_f_maps": 256,
        "input_dim": 2048,
        "kernel_size": 5,
        "normal_dropout_rate": 0.5,
        "channel_dropout_rate": 0.1,
        "temporal_dropout_rate": 0.1,
        "feature_layer_indices": [
            7,
            8,
            9
        ]
    },
    "decoder_params": {
        "num_layers": 8,
        "num_f_maps": 128,
        "time_emb_dim": 512,
        "kernel_size": 5,
        "dropout_rate": 0.1
    },
    "diffusion_params": {
        "timesteps": 1000,
        "sampling_timesteps": 25,
        "ddim_sampling_eta": 1.0,
        "snr_scale": 0.5,
        "cond_types": [
            "full",
            "zero",
            "boundary03-",
            "segment=1",
            "segment=1"
        ],
        "detach_decoder": False,
    },
    "loss_weights": {
        "encoder_ce_loss": 0.5,
        "encoder_mse_loss": 0.025,
        "encoder_boundary_loss": 0.0,
        "decoder_ce_loss": 0.5,
        "decoder_mse_loss": 0.025,
        "decoder_boundary_loss": 0.1
    },
    "batch_size": 4,
    "learning_rate": 0.0001,
    "weight_decay": 0,
    "num_epochs": 1001,
    "log_freq": 20,
    "class_weighting": True,
    "set_sampling_seed": True,
    "boundary_smooth": 3,
    "soft_label": 4,
    "log_train_results": False,
    "postprocess": {
        "type": "median",
        "value": 15
    },
}

###################### GTEA #######################

split_num = 4

for split_id in range(1, split_num + 1):

    params = copy.deepcopy(params_gtea)

    params['split_id'] = split_id
    params['naming'] = f'GTEA-Trained-S{split_id}'

    if not os.path.exists('configs'):
        os.makedirs('configs')

    file_name = os.path.join('configs', f'{params["naming"]}.json')

    with open(file_name, 'w') as outfile:
        json.dump(params, outfile, ensure_ascii=False)

###################### 50salads #######################

split_num = 5

for split_id in range(1, split_num + 1):

    params = copy.deepcopy(params_50salads)

    params['split_id'] = split_id
    params['naming'] = f'50salads-Trained-S{split_id}'

    if not os.path.exists('configs'):
        os.makedirs('configs')

    file_name = os.path.join('configs', f'{params["naming"]}.json')

    with open(file_name, 'w') as outfile:
        json.dump(params, outfile, ensure_ascii=False)

###################### Breakfast #######################

split_num = 4

for split_id in range(1, split_num + 1):

    params = copy.deepcopy(params_breakfast)

    params['split_id'] = split_id
    params['naming'] = f'Breakfast-Trained-S{split_id}'

    if not os.path.exists('configs'):
        os.makedirs('configs')

    file_name = os.path.join('configs', f'{params["naming"]}.json')

    with open(file_name, 'w') as outfile:
        json.dump(params, outfile, ensure_ascii=False)

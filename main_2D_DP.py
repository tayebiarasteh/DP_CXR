"""
Created on Aug 6, 2022.
main_2D_DP.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms, models
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_DP import Training
from Prediction_DP import Prediction
from data.data_provider_UKA import UKA_data_loader_2D, mimic_data_loader_2D

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")






def main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', pretrained=False, resnetnum=50):
    """Main function for training + validation centrally
        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml"
        valid: bool
            if we want to do validation
        resume: bool
            if we are resuming training on a model
        augment: bool
            if we want to have data augmentation during training
        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
    # valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
    valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['physical_batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['physical_batch_size'],
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    # Changeable network parameters
    model = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnetnum, pretrained=pretrained)
    # model = ModuleValidator.fix(model)

    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.NAdam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']))

    trainer = Training(cfg_path, resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader)



def main_train_DP_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml", valid=False,
                  resume=False, experiment_name='name', pretrained=False, resnetnum=34):
    """Main function for training + validation using DPSGD

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=False)
    valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['DP']['logical_batch_size'],
                                            drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['physical_batch_size'],
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    # Changeable network parameters
    model = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnetnum, pretrained=pretrained)
    # model = ModuleValidator.fix(model)

    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.NAdam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']))

    errors = ModuleValidator.validate(model, strict=False)
    assert len(errors) == 0
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=2,
        target_epsilon=params['DP']['epsilon'],
        target_delta=float(params['DP']['delta']),
        max_grad_norm=params['DP']['max_grad_norm'])

    # model, optimizer, train_loader = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     noise_multiplier=params['DP']['noise_multiplier'],
    #     max_grad_norm=params['DP']['max_grad_norm'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint_DP(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names, privacy_engine=privacy_engine)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, privacy_engine=privacy_engine)
    trainer.train_epoch_DP(train_loader=train_loader, valid_loader=valid_loader)


def main_test_central_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml", experiment_name='central_exp_for_test', resnetnum=50):
    """Main function for multi label prediction without DP

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Changeable network parameters
    model = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnetnum)
    # model = ModuleValidator.fix(model)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['physical_batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model)
    average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision = predictor.evaluate_2D(test_loader)

    print('------------------------------------------------------'
          '----------------------------------')
    print(f'\t experiment: {experiment_name}\n')

    print(f'\t avg AUROC: {average_AUROC.mean() * 100:.2f}% | avg accuracy: {average_accuracy.mean() * 100:.2f}%'
    f' | avg specificity: {average_specificity.mean() * 100:.2f}%'
    f' | avg recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | avg precision: {average_precision.mean() * 100:.2f}% | avg F1: {average_f1_score.mean() * 100:.2f}%\n')

    print('Individual AUROC:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_AUROC[idx] * 100:.2f}%')

    print('\nIndividual accuracy:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_accuracy[idx] * 100:.2f}%')

    print('\nIndividual specificity scores:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_specificity[idx] * 100:.2f}%')

    print('\nIndividual sensitivity scores:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_sensitivity[idx] * 100:.2f}%')

    print('------------------------------------------------------'
          '----------------------------------')

    # saving the stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f'\t experiment: {experiment_name}\n\n' \
          f'avg AUROC: {average_AUROC.mean() * 100:.2f}% | avg accuracy: {average_accuracy.mean() * 100:.2f}% ' \
          f' | avg specificity: {average_specificity.mean() * 100:.2f}%' \
          f' | avg recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | avg precision: {average_precision.mean() * 100:.2f}% | avg F1: {average_f1_score.mean() * 100:.2f}%\n\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)

    msg = f'Individual AUROC:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_AUROC[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual accuracy:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_accuracy[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual specificity scores:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_specificity[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual sensitivity scores:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_sensitivity[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)




def main_test_DP_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml", experiment_name='central_exp_for_test', resnetnum=50):
    """Main function for multi label prediction with differential privacy

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Changeable network parameters
    model = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnetnum)
    # model = ModuleValidator.fix(model)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['physical_batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    errors = ModuleValidator.validate(model, strict=False)
    assert len(errors) == 0
    privacy_engine = PrivacyEngine()

    model, _, _ = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer, # not important during testing; you should only put a placeholder here
        data_loader=test_loader, # not important during testing; you should only put a placeholder here
        epochs=params['Network']['num_epochs'], # not important during testing; you should only put a placeholder here
        target_epsilon=params['DP']['epsilon'], # not important during testing; you should only put a placeholder here
        target_delta=float(params['DP']['delta']), # not important during testing; you should only put a placeholder here
        max_grad_norm=params['DP']['max_grad_norm']) # not important during testing; you should only put a placeholder here

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model_DP(model=model, privacy_engine=privacy_engine)
    average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision = predictor.evaluate_2D(test_loader)

    print('------------------------------------------------------'
          '----------------------------------')
    print(f'\t experiment: {experiment_name}\n')

    print(f'\t avg AUROC: {average_AUROC.mean() * 100:.2f}% | avg accuracy: {average_accuracy.mean() * 100:.2f}%'
    f' | avg specificity: {average_specificity.mean() * 100:.2f}%'
    f' | avg recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | avg precision: {average_precision.mean() * 100:.2f}% | avg F1: {average_f1_score.mean() * 100:.2f}%\n')

    print('Individual AUROC:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_AUROC[idx] * 100:.2f}%')

    print('\nIndividual accuracy:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_accuracy[idx] * 100:.2f}%')

    print('\nIndividual specificity scores:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_specificity[idx] * 100:.2f}%')

    print('\nIndividual sensitivity scores:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_sensitivity[idx] * 100:.2f}%')

    print('------------------------------------------------------'
          '----------------------------------')

    # saving the stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f'\t experiment: {experiment_name}\n\n' \
          f'avg AUROC: {average_AUROC.mean() * 100:.2f}% | avg accuracy: {average_accuracy.mean() * 100:.2f}% ' \
          f' | avg specificity: {average_specificity.mean() * 100:.2f}%' \
          f' | avg recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | avg precision: {average_precision.mean() * 100:.2f}% | avg F1: {average_f1_score.mean() * 100:.2f}%\n\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)

    msg = f'Individual AUROC:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_AUROC[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual accuracy:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_accuracy[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual specificity scores:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_specificity[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual sensitivity scores:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_sensitivity[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)



def load_pretrained_resnet(num_classes=2, resnet_num=34, pretrained=False):
    # Load a pre-trained model from config file

    # Load a pre-trained model from Torchvision
    if resnet_num == 9:
        model = models.resnet.ResNet(models.resnet.BasicBlock, [1, 1, 1, 1])
        in_features = model.fc.in_features
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        model.fc = torch.nn.Linear(in_features, num_classes)
        model.bn1 = torch.nn.GroupNorm(32, 64)
        model.layer1[0].bn1 = torch.nn.GroupNorm(32, 64)
        model.layer1[0].bn2 = torch.nn.GroupNorm(32, 64)
        model.layer2[0].bn1 = torch.nn.GroupNorm(32, 128)
        model.layer2[0].bn2 = torch.nn.GroupNorm(32, 128)
        model.layer2[0].downsample[1] = torch.nn.GroupNorm(32, 128)
        model.layer3[0].bn1 = torch.nn.GroupNorm(32, 256)
        model.layer3[0].bn2 = torch.nn.GroupNorm(32, 256)
        model.layer3[0].downsample[1] = torch.nn.GroupNorm(32, 256)
        model.layer4[0].bn1 = torch.nn.GroupNorm(32, 512)
        model.layer4[0].bn2 = torch.nn.GroupNorm(32, 512)
        model.layer4[0].downsample[1] = torch.nn.GroupNorm(32, 512)

        if pretrained:
            model.load_state_dict(torch.load('./pretraining.pth'))

        for param in model.parameters():
            param.requires_grad = True

    elif resnet_num == 18:
        if pretrained:
            model = models.resnet18(weights='DEFAULT')
        else:
            model = models.resnet18()
        for param in model.parameters():
            param.requires_grad = True
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, num_classes))  # for resnet 18

    elif resnet_num == 34:
        if pretrained:
            model = models.resnet34(weights='DEFAULT')
        else:
            model = models.resnet34()
        for param in model.parameters():
            param.requires_grad = True
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, num_classes))  # for resnet 34

    elif resnet_num == 50:
        if pretrained:
            model = models.resnet50(weights='DEFAULT')
        else:
            model = models.resnet50()
        for param in model.parameters():
            param.requires_grad = True
        model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, num_classes)) # for resnet 50

    return model





if __name__ == '__main__':
    delete_experiment(experiment_name='tempp', global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml")
    main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml",
                  valid=False, resume=False, augment=True, experiment_name='tempp', pretrained=True, resnetnum=9)
    # main_train_DP_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml",
    #               valid=False, resume=False, experiment_name='tempp', pretrained=True, resnetnum=9)
    # main_test_central_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml", experiment_name='central_UKA5k_3labels_imagenetpretrain_resnet50_groupnorm_lr2e5_batch16')
    # main_test_DP_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml", experiment_name='DP_UKA5k_8labels_imagenetpretrain_resnet50_lr5e5_decay1e5_epsilon500_maxnorm1.9_batch16_logibatch64')
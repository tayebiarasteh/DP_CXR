"""
Created on Aug 6, 2022.
main_2D_DP.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import models
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
import numpy as np

from config.serde import open_experiment, create_experiment
from Train_Valid_DP import Training
from Prediction_DP import Prediction
from data.data_provider_UKA import UKA_data_loader_2D

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")





def main_train_central_2D(global_config_path="DP_CXR/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', pretrained=False, resnet_num=50, mish=False, size256=False):
    """Main function for training + validation centrally
        Parameters
        ----------
        global_config_path: str
            always global_config_path="DP_CXR/config/config.yaml"
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

    train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, size256=size256)
    valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, size256=size256)

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
    model = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnet_num, pretrained=pretrained, mish=mish, size256=size256)
    model = ModuleValidator.fix(model)

    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.NAdam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']))

    trainer = Training(cfg_path, resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader)



def main_train_DP_2D(global_config_path="DP_CXR/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', pretrained=False, resnet_num=34, mish=False, size256=False):
    """Main function for training + validation using DPSGD

        Parameters
        ----------
        global_config_path: str
            always global_config_path="DP_CXR/config/config.yaml"

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

    train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
    valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=augment)

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
    model = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnet_num, pretrained=pretrained, mish=mish, size256=size256)
    model = ModuleValidator.fix(model)
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
        epochs=params['Network']['num_epochs'],
        target_epsilon=params['DP']['epsilon'],
        target_delta=float(params['DP']['delta']),
        max_grad_norm=params['DP']['max_grad_norm'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint_DP(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names, privacy_engine=privacy_engine)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, privacy_engine=privacy_engine)
    trainer.train_epoch_DP(train_loader=train_loader, valid_loader=valid_loader)


def main_test_central_2D(global_config_path="DP_CXR/config/config.yaml", experiment_name='central_exp_for_test', resnet_num=50):
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
    model = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnet_num)
    model = ModuleValidator.fix(model)

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



def main_test_DP_2D(global_config_path="DP_CXR/config/config.yaml", experiment_name='central_exp_for_test', resnet_num=50, mish=False, experiment_epoch_num=10):
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
    model = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnet_num, mish=mish)
    model = ModuleValidator.fix(model)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['physical_batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    optimizer = torch.optim.NAdam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']))

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
    predictor.setup_model_DP(model=model, privacy_engine=privacy_engine, epoch_num=experiment_epoch_num)
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




def main_test_2D_bootstrap(global_config_path="DP_CXR/config/config.yaml",
                                                 experiment_name1='central_exp_for_test',
                                                 experiment1_epoch_num=100, resnet_num=9, mish=True):
    """Main function for multi label prediction with DP

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    dataset_name = 'UKA-CXR'
    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']

    test_dataset = UKA_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params1['Network']['physical_batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    model1 = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnet_num, mish=mish)
    model1 = ModuleValidator.fix(model1)
    optimizer = torch.optim.NAdam(model1.parameters(), lr=float(params1['Network']['lr']),
                                 weight_decay=float(params1['Network']['weight_decay']))
    errors = ModuleValidator.validate(model1, strict=False)
    assert len(errors) == 0

    privacy_engine = PrivacyEngine()
    model1, _, _ = privacy_engine.make_private_with_epsilon(
        module=model1,
        optimizer=optimizer, # not important during testing; you should only put a placeholder here
        data_loader=test_loader, # not important during testing; you should only put a placeholder here
        epochs=params1['Network']['num_epochs'], # not important during testing; you should only put a placeholder here
        target_epsilon=params1['DP']['epsilon'], # not important during testing; you should only put a placeholder here
        target_delta=float(params1['DP']['delta']), # not important during testing; you should only put a placeholder here
        max_grad_norm=params1['DP']['max_grad_norm']) # not important during testing; you should only put a placeholder here

    index_list = []
    for counter in range(1000):
        index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))

    # Initialize prediction 1
    predictor1 = Prediction(cfg_path1, label_names)
    predictor1.setup_model_DP(model=model1, privacy_engine=privacy_engine, epoch_num=experiment1_epoch_num)

    delta = float(6e-6)
    epsilon = predictor1.privacy_engine.get_epsilon(delta)
    print(f"\n(ε = {epsilon:.2f}, δ = {delta})\n")
    msg = f"\n(ε = {epsilon:.2f}, δ = {delta})\n"
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)

    pred_array1, target_array1, gender_array1, age_array1 = predictor1.predict_only(test_loader)
    AUC_list1 = predictor1.bootstrapper(pred_array1.cpu().numpy(), target_array1.int().cpu().numpy(), index_list, dataset_name)



def main_test_2D_pvalue_out_of_bootstrap(global_config_path="DP_CXR/config/config.yaml",
                                                 experiment_name1='central_exp_for_test', experiment_name2='central_exp_for_test',
                                                 experiment1_epoch_num=100, experiment2_epoch_num=100, resnet_num=9, mish=True):
    """Main function for multi label prediction
    model1 must be DP model
    model2 must be non DP model

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    dataset_name = 'UKA-CXR'
    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']

    test_dataset = UKA_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params1['Network']['physical_batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    index_list = []
    for counter in range(1000):
        index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))

    model1 = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnet_num, mish=mish)
    model1 = ModuleValidator.fix(model1)
    optimizer = torch.optim.NAdam(model1.parameters(), lr=float(params1['Network']['lr']),
                                 weight_decay=float(params1['Network']['weight_decay']))
    errors = ModuleValidator.validate(model1, strict=False)
    assert len(errors) == 0
    privacy_engine = PrivacyEngine()
    model1, _, _ = privacy_engine.make_private_with_epsilon(
        module=model1,
        optimizer=optimizer, # not important during testing; you should only put a placeholder here
        data_loader=test_loader, # not important during testing; you should only put a placeholder here
        epochs=params1['Network']['num_epochs'], # not important during testing; you should only put a placeholder here
        target_epsilon=params1['DP']['epsilon'], # not important during testing; you should only put a placeholder here
        target_delta=float(params1['DP']['delta']), # not important during testing; you should only put a placeholder here
        max_grad_norm=params1['DP']['max_grad_norm']) # not important during testing; you should only put a placeholder here

    # Initialize prediction 1
    predictor1 = Prediction(cfg_path1, label_names)
    predictor1.setup_model_DP(model=model1, privacy_engine=privacy_engine, epoch_num=experiment1_epoch_num)

    delta = float(6e-6)
    epsilon = predictor1.privacy_engine.get_epsilon(delta)
    print(f"\n(ε = {epsilon:.2f}, δ = {delta})\n")
    msg = f"\n(ε = {epsilon:.2f}, δ = {delta})\n"
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)

    pred_array1, target_array1, gender_array1, age_array1 = predictor1.predict_only(test_loader)
    AUC_list1 = predictor1.bootstrapper(pred_array1.cpu().numpy(), target_array1.int().cpu().numpy(), index_list, dataset_name)

    model2 = load_pretrained_resnet(num_classes=len(weight), resnet_num=resnet_num, mish=False)
    # model2 = ModuleValidator.fix(model2)

    # Initialize prediction 2
    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2, label_names)
    predictor2.setup_model(model=model2, epoch_num=experiment2_epoch_num)
    pred_array2, target_array2, gender_array2, age_array2 = predictor2.predict_only(test_loader)
    AUC_list2 = predictor2.bootstrapper(pred_array2.cpu().numpy(), target_array2.int().cpu().numpy(), index_list, dataset_name)

    print('individual labels p-values:\n')
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            print(f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2')
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                print(f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1')
            else:
                print(f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label')

    print('\nAvg AUC of labels p-values:\n')
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)
    if ratio1 <= 0.05:
        print(f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average')
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)
        if ratio2 <= 0.05:
            print(f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average')
        else:
            print(f'\tp-value: {ratio1}; models NOT significantly different on average for all labels')


    msg = f'\n\nindividual labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            msg = f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2'
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                msg = f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1'
            else:
                msg = f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)


    msg = f'\n\nAvg AUC of labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)
    if ratio1 <= 0.05:
        msg = f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average'
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)
        if ratio2 <= 0.05:
            msg = f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average'
        else:
            msg = f'\tp-value: {ratio1}; models NOT significantly different on average for all labels'

    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)



def load_pretrained_resnet(num_classes=2, resnet_num=34, pretrained=False, mish=False, size256=False):
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

        if mish:
            activation = torch.nn.Mish()
            model.relu = activation
            model.layer1[0].relu = activation
            model.layer1[0].relu = activation
            model.layer2[0].relu = activation
            model.layer2[0].relu = activation
            model.layer3[0].relu = activation
            model.layer3[0].relu = activation
            model.layer4[0].relu = activation
            model.layer4[0].relu = activation

        if pretrained:
            if size256:
                model.load_state_dict(torch.load('DP_CXR/pretraining_resnet9_256.pth'))
            else:
                model.load_state_dict(torch.load('DP_CXR/pretraining_resnet9_512.pth'))

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
    main_train_DP_2D(global_config_path="DP_CXR/config/config.yaml",
                  valid=True, augment=False, resume=False, experiment_name='name', pretrained=True, resnet_num=9, mish=True, size256=False)
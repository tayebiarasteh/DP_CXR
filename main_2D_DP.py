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
from data.data_provider_UKA import UKA_data_loader_2D

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

    train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
    valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)

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
    # model = load_pretrained_swin_transformer(num_classes=len(weight), mode='b', pretrained=pretrained)
    model = ModuleValidator.fix(model)

    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume, label_names=label_names)
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
    valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)

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
    model = ModuleValidator.fix(model)

    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    errors = ModuleValidator.validate(model, strict=False)
    assert len(errors) == 0
    privacy_engine = PrivacyEngine()

    # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     epochs=params['num_epochs'],
    #     target_epsilon=params['DP']['epsilon'],
    #     target_delta=float(params['DP']['delta']),
    #     max_grad_norm=params['DP']['max_grad_norm'])

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=params['DP']['noise_multiplier'],
        max_grad_norm=params['DP']['max_grad_norm'])

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint_DP(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names, privacy_engine=privacy_engine)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, privacy_engine=privacy_engine)
    trainer.train_epoch_DP(train_loader=train_loader, valid_loader=valid_loader)




def main_test_central_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml", experiment_name='central_exp_for_test'):
    """Main function for multi label prediction

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
    model = load_pretrained_model(num_classes=len(weight), resnet_num=50)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model)
    average_f1_score, average_AUROC, average_accuracy, average_specifity, average_sensitivity, average_precision = predictor.evaluate_2D(test_loader)

    print('------------------------------------------------------'
          '----------------------------------')
    print(f'\t experiment: {experiment_name}\n')

    print(f'\t Average F1: {average_f1_score.mean() * 100:.2f}% | Average AUROC: {average_AUROC.mean() * 100:.2f}% | Average accuracy: {average_accuracy.mean() * 100:.2f}%'
    f' | Average specifity: {average_specifity.mean() * 100:.2f}%'
    f' | Average recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | Average precision: {average_precision.mean() * 100:.2f}%\n')

    print('Individual F1 scores:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_f1_score[idx] * 100:.2f}%')

    print('\nIndividual AUROC:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_AUROC[idx] * 100:.2f}%')

    print('------------------------------------------------------'
          '----------------------------------')

    # saving the stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f'\t experiment: {experiment_name}\n\n' \
          f'Average F1: {average_f1_score.mean() * 100:.2f}% | Average AUROC: {average_AUROC.mean() * 100:.2f}% | Average accuracy: {average_accuracy.mean() * 100:.2f}% ' \
          f' | Average specifity: {average_specifity.mean() * 100:.2f}%' \
          f' | Average recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | Average precision: {average_precision.mean() * 100:.2f}%\n\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)

    msg = f'Individual F1 scores:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_f1_score[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
    msg = f'\n\nIndividual AUROC:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_AUROC[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)





def load_pretrained_resnet(num_classes=2, resnet_num=34, pretrained=False):
    # Load a pre-trained model from config file
    # self.model.load_state_dict(torch.load(self.model_info['pretrain_model_path']))

    # Load a pre-trained model from Torchvision
    if resnet_num == 18:
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



def load_pretrained_swin_transformer(num_classes=2, mode='b', pretrained=False):
    # Load a pre-trained model from config file

    # Load a pre-trained model from Torchvision
    if mode == 't':
        if pretrained:
            model = models.swin_t(weights='DEFAULT')
        else:
            model = models.swin_t()
        for param in model.parameters():
            param.requires_grad = True
        model.head = torch.nn.Linear(768, num_classes)  # for Swin Transformer tiny

    elif mode == 's':
        if pretrained:
            model = models.swin_s(weights='DEFAULT')
        else:
            model = models.swin_s()
        for param in model.parameters():
            param.requires_grad = True
        model.head = torch.nn.Linear(768, num_classes)  # for Swin Transformer small

    elif mode == 'b':
        if pretrained:
            model = models.swin_b(weights='DEFAULT')
        else:
            model = models.swin_b()
        for param in model.parameters():
            param.requires_grad = True
        model.head = torch.nn.Linear(1024, num_classes)  # for Swin Transformer base

    return model






if __name__ == '__main__':
    # delete_experiment(experiment_name='temp', global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml")
    # main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml",
    #               valid=True, resume=False, augment=True, experiment_name='temp', pretrained=False, resnetnum=18)
    main_train_DP_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml",
                  valid=True, resume=True, experiment_name='temp', pretrained=False, resnetnum=18)
    # main_test_central_2D(global_config_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml", experiment_name='temp')
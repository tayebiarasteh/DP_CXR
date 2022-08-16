"""
Created on Aug 6, 2022.
Prediction_DP.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os.path
import torch.nn.functional as F
import numpy as np
import torchmetrics
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

from config.serde import read_config

epsilon = 1e-15



class Prediction:
    def __init__(self, cfg_path, label_names):
        """
        This class represents prediction (testing) process similar to the Training class.
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.label_names = label_names
        self.setup_cuda()


    def setup_cuda(self, cuda_device_id=0):
        """setup the device.
        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def setup_model(self, model, model_file_name=None):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model = model.to(self.device)

        # self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'], model_file_name)))
        self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "epoch775_" + model_file_name))



    def evaluate_2D(self, test_loader):
        """Testing 2D-wise.

        Parameters
        ----------

        Returns
        -------
        """
        self.model.eval()
        total_f1_score = []
        total_AUROC = []
        total_accuracy = []
        total_specifity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        # initializing the caches
        logits_with_sigmoid_cache = torch.Tensor([]).to(self.device)
        logits_no_sigmoid_cache = torch.Tensor([]).to(self.device)
        labels_cache = torch.Tensor([]).to(self.device)

        for idx, (image, label) in enumerate(tqdm(test_loader)):

            image = image.to(self.device)
            label = label.to(self.device)
            label = label.float()

            with torch.no_grad():
                output = self.model(image)

                output_sigmoided = F.sigmoid(output)
                output_sigmoided = (output_sigmoided > 0.5).float()

                # saving the logits and labels of this batch
                logits_with_sigmoid_cache = torch.cat((logits_with_sigmoid_cache, output_sigmoided))
                logits_no_sigmoid_cache = torch.cat((logits_no_sigmoid_cache, output))
                labels_cache = torch.cat((labels_cache, label))

        ############ Evaluation metric calculation ########

        # Metrics calculation (macro) over the whole set
        logits_with_sigmoid_cache = logits_with_sigmoid_cache.int().cpu().numpy()
        logits_no_sigmoid_cache = logits_no_sigmoid_cache.int().cpu().numpy()
        labels_cache = labels_cache.int().cpu().numpy()

        confusion = metrics.multilabel_confusion_matrix(labels_cache, logits_with_sigmoid_cache)

        try:
            # plotting the ROC curve
            for idx in range(len(confusion)):
                fpr, tpr, _ = metrics.roc_curve(labels_cache[:, idx], logits_no_sigmoid_cache[:, idx], pos_label=1)
                metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
                plt.grid()
                plt.title('label number: ' + str(idx))
                plt.show()
        except:
            print("hi")

        F1_disease = []
        accuracy_disease = []
        specifity_disease = []
        sensitivity_disease = []
        precision_disease = []

        for idx, disease in enumerate(confusion):
            TN = disease[0, 0]
            FP = disease[0, 1]
            FN = disease[1, 0]
            TP = disease[1, 1]
            F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
            accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
            specifity_disease.append(TN / (TN + FP + epsilon))
            sensitivity_disease.append(TP / (TP + FN + epsilon))
            precision_disease.append(TP / (TP + FP + epsilon))

        # Macro averaging
        total_f1_score.append(np.stack(F1_disease))
        try:
            total_AUROC.append(metrics.roc_auc_score(labels_cache, logits_no_sigmoid_cache, average=None))
        except:
            print('hi')
            pass
        total_accuracy.append(np.stack(accuracy_disease))
        total_specifity_score.append(np.stack(specifity_disease))
        total_sensitivity_score.append(np.stack(sensitivity_disease))
        total_precision_score.append(np.stack(precision_disease))

        average_f1_score = np.stack(total_f1_score).mean(0)
        average_AUROC = np.stack(total_AUROC).mean(0)
        average_accuracy = np.stack(total_accuracy).mean(0)
        average_specifity = np.stack(total_specifity_score).mean(0)
        average_sensitivity = np.stack(total_sensitivity_score).mean(0)
        average_precision = np.stack(total_precision_score).mean(0)

        self.plot_confusion_matrix(confusion)


        return average_f1_score, average_AUROC, average_accuracy, average_specifity, average_sensitivity, average_precision


    def plot_confusion_matrix(self, all_matrices, target_names=None,
                              title='Confusion matrix', cmap=None, normalize=False):
        """
        given a sklearn confusion matrix (cm), make a nice plot
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix
        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']
        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      plt.get_cmap('jet') or plt.cm.Blues
        normalize:    If False, plot the raw numbers
                      If True, plot the proportions
        """

        for cm in all_matrices:
            accuracy = np.trace(cm) / np.sum(cm).astype('float')
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap('Blues')

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            # plt.imshow(cm)
            plt.title(title)
            plt.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label\naccuracy={:0.2f}%; misclass={:0.2f}%'.format(accuracy*100, misclass*100))
            plt.show()
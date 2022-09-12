import random
import numpy as np
from copy import deepcopy
from typing import List, Union
import torch

import classifier
import utils

from .seg_trainer import seg_trainer


def harmonic_mean(base_iou, novel_iou):
    return 2 / (1. / base_iou + 1. / novel_iou)


class GIFS_seg_trainer(seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(GIFS_seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion,
                                               dataset_module, device)

        self.continual_vanilla_train_set = dataset_module.get_continual_vanilla_train_set(cfg)
        self.continual_aug_train_set = dataset_module.get_continual_aug_train_set(cfg)
        self.continual_test_set = dataset_module.get_continual_test_set(cfg)



        #########################################################################
        print('length of test set')
        print(len(self.continual_test_set))
        # print('\n printing max and min index 0 and 1 test image: ')
        # print(self.continual_aug_train_set[0][0].size())
        # print(self.continual_aug_train_set[1][0].size())
        # print(torch.max(self.continual_aug_train_set[0][0]))
        # print(torch.min(self.continual_aug_train_set[0][0]))
        # print(self.continual_aug_train_set[0][0].is_cuda)
        # print('\n printing max and min index 0 and 1 test label: ')
        # print(self.continual_aug_train_set[0][1].size())
        # print(self.continual_aug_train_set[1][1].size())
        # print(torch.max(self.continual_aug_train_set[0][1]))
        # print(torch.min(self.continual_aug_train_set[0][1]))
        # print(self.continual_aug_train_set[0][1].is_cuda)
        #########################################################################

        self.continual_test_loader = torch.utils.data.DataLoader(self.continual_test_set,
                                                                 batch_size=cfg.TEST.batch_size,
                                                                 shuffle=False,
                                                                 **self.loader_kwargs)

        self.test_base_iou = []
        self.test_novel_iou = []

    # self.train_one is inherited from seg trainer
    # self.val_one is inherited from seg trainer

    def test_one(self, device, num_runs=5):
        num_shots = self.cfg.TASK_SPECIFIC.GIFS.num_shots

        # Parse image candidates
        testing_label_candidates = self.train_set.dataset.invisible_labels

        vanilla_image_candidates = {}
        for l in testing_label_candidates:
            vanilla_image_candidates[l] = set(
                self.continual_vanilla_train_set.dataset.get_class_map(l))

        # To ensure only $num_shots$ number of examples are used
        image_candidates = {}
        for k_i in vanilla_image_candidates:
            image_candidates[k_i] = deepcopy(vanilla_image_candidates[k_i])
            for k_j in vanilla_image_candidates:
                if k_i == k_j: continue
                image_candidates[k_i] -= vanilla_image_candidates[k_j]
            image_candidates[k_i] = sorted(list(image_candidates[k_i]))

        # We use a total of $num_runs$ consistent random seeds.
        np.random.seed(1234)
        seed_list = np.random.randint(0, 99999, size=(num_runs,))

        # Meta Test!
        run_base_iou_list = []
        run_novel_iou_list = []
        run_total_iou_list = []
        run_harm_iou_list = []
        for i in range(num_runs):
            np.random.seed(seed_list[i])
            random.seed(seed_list[i])
            torch.manual_seed(seed_list[i])
            support_set = {}
            for k in image_candidates.keys():
                assert len(image_candidates[k]) > num_shots, "fewer samples than num_shots?"
                selected_idx = []
                iter_cnt = 0
                for _ in range(num_shots):
                    idx = random.choice(image_candidates[k])
                    while True:
                        iter_cnt += 1
                        if iter_cnt > num_shots + 20:
                            raise ValueError("Malformed image candidates?")
                        novel_img_chw, mask_hw = self.continual_vanilla_train_set[idx]
                        pixel_sum = torch.sum(mask_hw == k)
                        assert pixel_sum > 0, f"Sample {idx} does not contain class {k}"
                        # If the selected sample is bad (more than 1px) and has not been selected,
                        # we choose the example.
                        if pixel_sum != 1 and idx not in selected_idx:
                            selected_idx.append(idx)
                            break
                        else:
                            idx = random.choice(image_candidates[k])
                assert len(selected_idx) == num_shots
                support_set[k] = list(selected_idx)

            # get per-class IoU on the entire validation set based on results from the support set
            classwise_iou = self.continual_test_single_pass(support_set)

            novel_iou_list = []
            base_iou_list = []
            for i in range(len(classwise_iou)):
                if i in testing_label_candidates:
                    novel_iou_list.append(classwise_iou[i])
                else:
                    base_iou_list.append(classwise_iou[i])
            base_iou = np.mean(base_iou_list)
            novel_iou = np.mean(novel_iou_list)
            print("Base IoU: {:.4f} Novel IoU: {:.4f} Total IoU: {:.4f}".format(base_iou, novel_iou,
                                                                                np.mean(
                                                                                    classwise_iou)))
            # run_base_iou_list.append(base_iou)
            # run_novel_iou_list.append(novel_iou)
            # run_total_iou_list.append(np.mean(classwise_iou))
            # run_harm_iou_list.append(harmonic_mean(base_iou, novel_iou))
        # print("Results of {} runs with {} shots".format(num_runs, num_shots))
        # print("Base IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(run_base_iou_list), np.std(run_base_iou_list)))
        # print("Novel IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(run_novel_iou_list), np.std(run_novel_iou_list)))
        # print("Harmonic IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(run_harm_iou_list), np.std(run_harm_iou_list)))
        # print("Total IoU Mean: {:.4f} Std: {:.4f}".format(np.mean(run_total_iou_list), np.std(run_total_iou_list)))
        print("test base iou: ")
        print(self.test_base_iou)
        print("test novel iou: ")
        print(self.test_novel_iou)
        total_mean_iou = np.add(self.test_base_iou, self.test_novel_iou) / 2
        max_mean_iou_index = np.where(total_mean_iou == np.amax(total_mean_iou))[0][0]

        print("Results of {} runs in a non-few-shot setting".format(num_runs))
        # print("max Base IoU: {:.4f} max Novel IoU: {:.4f}".format(np.max(self.test_base_iou), np.max(self.test_novel_iou)))
        print("max Base IoU: {:.4f} max Novel IoU: {:.4f}".format(
            self.test_base_iou[max_mean_iou_index],
            self.test_novel_iou[max_mean_iou_index]))

        print("mean Base IoU: {:.4f} mean Novel IoU: {:.4f}".format(np.mean(self.test_base_iou),
                                                                    np.mean(self.test_novel_iou)))

    def classifier_weight_imprinting(self, base_id_list: List[int], novel_id_list: List[int],
                                     support_set: dict):
        """Use masked average pooling to initialize a new 1x1 convolutional HEAD for semantic segmentation

        The resulting classifier will produce per-pixel classification from class 0 (usually background)
        upto class max(max(base_class_idx), max(novel_class_idx)). If there is discontinuity in base_class_idx
        and novel_class_idx (e.g., base: [0, 1, 2, 4]; novel: [5, 6]), then the class weight of the non-used class
        will be initialized as full zeros.

        Args:
            base_id_list (List[int]): a sorted list containing base class id
            novel_id_list (List[int]): a sorted list containing novel class id
            supp_img_bchw (torch.Tensor): Normalized support set image tensor
            supp_mask_bhw (torch.Tensor): Complete segmentation mask of support set

        Returns:
            torch.Tensor: a weight vector that can be directly plugged back to
                data.weight of the 1x1 classification convolution
        """
        assert self.prv_backbone_net is not None
        assert self.prv_post_processor is not None
        max_cls = self.cfg.meta_testing_num_classes
        assert max_cls >= max(max(base_id_list), max(novel_id_list)) + 1
        assert self.prv_post_processor.pixel_classifier.class_mat.weight.data.shape[0] == len(
            base_id_list)

        ori_cnt = 0
        class_weight_vec_list = []
        for c in range(max_cls):
            if c in novel_id_list:
                # Aggregate all candidates in support set
                vec_list = []  # store MAP result for every image
                assert c in support_set
                for idx in support_set[c]:
                    img_chw, mask_hw = self.continual_vanilla_train_set[idx]
                    # novel class. Use MAP to initialize weight
                    supp_img_bchw_tensor = img_chw.view((1,) + img_chw.shape).to(self.device)
                    supp_mask_bhw_tensor = mask_hw.view((1,) + mask_hw.shape).to(self.device)
                    assert c in supp_mask_bhw_tensor
                    with torch.no_grad():
                        support_feature = self.prv_backbone_net(supp_img_bchw_tensor)
                        class_weight_vec = utils.masked_average_pooling(supp_mask_bhw_tensor == c,
                                                                        support_feature, True)
                        vec_list.append(class_weight_vec)
                class_weight_vec = torch.mean(torch.stack(vec_list), dim=0)
            elif c in base_id_list:
                # base class. Copy weight from learned HEAD
                class_weight_vec = self.prv_post_processor.pixel_classifier.class_mat.weight.data[
                    ori_cnt]
                ori_cnt += 1
            else:
                # not used class
                class_weight_vec = torch.zeros_like(
                    self.prv_post_processor.pixel_classifier.class_mat.weight.data[0])
            class_weight_vec = class_weight_vec.reshape((-1, 1, 1))  # C x 1 x 1
            class_weight_vec_list.append(class_weight_vec)

        classifier_weights = torch.stack(class_weight_vec_list)  # num_classes x C x 1 x 1
        return classifier_weights

    def finetune_backbone(self, base_class_idx, novel_class_idx, support_set):
        raise NotImplementedError

    def continual_test_single_pass(self, support_set):
        raise NotImplementedError

    def novel_adapt(self, base_class_idx, novel_class_idx, support_set):
        """Novel adapt for quantitative evaluation on dataset

        Args:
            base_class_idx (list of ints): ints of existing base classes
            novel_class_idx (list of ints): indices of novel class
            support_set (dict): dictionary with novel_class_idx as keys and dataset idx as values
        """
        max_cls = max(max(base_class_idx), max(novel_class_idx)) + 1
        self.post_processor = classifier.dispatcher(self.cfg, self.feature_shape,
                                                    num_classes=max_cls)
        self.post_processor = self.post_processor.to(self.device)
        # Aggregate weights
        aggregated_weights = self.classifier_weight_imprinting(base_class_idx, novel_class_idx,
                                                               support_set)
        self.post_processor.pixel_classifier.class_mat.weight.data = aggregated_weights

        # Optimization over support set to fine-tune initialized vectors
        if self.cfg.TASK_SPECIFIC.GIFS.fine_tuning:
            self.finetune_backbone(base_class_idx, novel_class_idx, support_set)

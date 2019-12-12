'''Copyright (C) 2019 Erik Linder-Norén, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changes to original version: Adopted to work with multi-layer pollen data
'''

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
from shutil import copyfile
from pathlib import Path    
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Sampler

import pandas as pd 

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import matplotlib
print(matplotlib.get_backend())
matplotlib.use('agg')

import copy
import cv2

"""
Test a sample of microscope, consisting of multiple stacks.
"""

class GroupSampler(Sampler):
    r"""Creates a batch for each sample (one sample consisting of multiple layers)

    Args:
        filenames (List): List of filenames of the dataset
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, filenames):
        img_files_df = pd.DataFrame(filenames,columns=['filename'])
        img_files_df['index'] = np.arange(len(filenames))
        print(img_files_df.head())
        img_files_df.sort_values('filename')
        img_files_df['group'] = img_files_df['filename'].apply(lambda x: os.path.splitext(os.path.basename(x))[0].split('-')[0])
        self.df = img_files_df
        self.groups = img_files_df['group'].unique()

    def __iter__(self):
        batch = []
        for idx in range(len(self.groups)):
            group = self.groups[idx]
            elements = self.df[self.df['group']==group]
            for i in range(len(elements)):
                index = elements.iloc[i]['index']
                batch.append(index)
            yield batch
            batch = []

    def __len__(self):
        return len(self.groups)


def merge_bb(box1, box2, iou_threshold, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    iou, box_index = iou.max(0)
    if iou >= iou_threshold:
        return [inter_rect_x1[box_index],inter_rect_y1[box_index],inter_rect_x2[box_index],inter_rect_y2[box_index]]
    else:
        return None
    return iou

def target_merge(targets,iou_threshold):
    if targets.shape[0] == 0:
        return targets

    ts = targets[0].unsqueeze(0)
    targets = targets[1:]
    for i in range(len(targets)):
        length = len(ts)
        overlap_found = False
        for j in range(length):
            bbox = merge_bb(targets[i,2:].unsqueeze(0), ts[j,2:].unsqueeze(0),iou_threshold)
            label_0 = targets[i,1]
            label_1 = ts[j,1]
            if bbox is not None and label_0 == label_1:
                ts[j,2:6] = torch.Tensor(bbox)      # merge
                overlap_found = True
        if not overlap_found:
            # we did not find an overlap, thus add this target to ts
            ts = torch.cat((ts,targets[i].unsqueeze(0)),0)   # add additional bb
    
    ts[:,0] = 0
    return ts

def output_merge(outputs,iou_threshold):
    
    # output has shape [batch_size, number_predictions, features]
    # features are 0:4 boundingboxes, 4 score, -1 label
    targets = torch.cat(outputs)    # [all_number_predictions, features]
    if targets.shape[0] == 0:
        return targets
        
    ts = targets[0].unsqueeze(0)
    targets = targets[1:]
    for i in range(len(targets)):
        length = len(ts)
        overlap_found = False
        for j in range(length):
            bbox = merge_bb(targets[i,:4].unsqueeze(0), ts[j,:4].unsqueeze(0),iou_threshold)
            label_0 = targets[i,-1]
            label_1 = ts[j,-1]
            if bbox is not None and label_0 == label_1:
                ts[j,:4] = torch.Tensor(bbox)      # merge
                if ts[j,4] < targets[i,4]:
                    ts[j,4] = targets[i,4]
                overlap_found = True

        if not overlap_found:
            # we did not find an overlap, thus add this target to ts
            ts = torch.cat((ts,targets[i].unsqueeze(0)),0)   # add additional bb
    return ts.unsqueeze(0)


def output_non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, area_thres=None, min_detections=1):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    if len(prediction) == 0:
        return prediction

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
#    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        if image_pred is None:
            continue

        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        
        if area_thres is not None:
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue

            x1, y1, x2, y2 = image_pred[:, 0], image_pred[:, 1], image_pred[:, 2], image_pred[:, 3]
            image_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            # image_area = image_area.abs()

            # if (image_area < area_thres[0]).sum() > 0:
            #     print(f'Discarded (too small) {(image_area < area_thres[0]).sum()} {image_area} {area_thres[0]}', )
            image_pred = image_pred[image_area >= area_thres[0]]
            image_area = image_area[image_area >= area_thres[0]]

            if not image_pred.size(0):
                continue

            image_pred = image_pred[image_area <= area_thres[1]]


            # If none are remaining => process next image
            if not image_pred.size(0):
                continue

        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            # print(invalid.sum(),invalid.shape)
            if invalid.sum() >= min_detections:
                keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def target_area_threshold(targets, area_thres):
    """
    """
    if targets.shape[0] == 0:
        return targets

    if area_thres is not None:
        x1, y1, x2, y2 = targets[:, 2], targets[:, 3], targets[:, 4], targets[:, 5]
        image_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        # image_area = image_area.abs()
        targets = targets[image_area >= area_thres[0]]
        target_area_threshold.num_removed += (image_area < area_thres[0]).sum()

        if not targets.size(0):
            return targets

        image_area = image_area[image_area >= area_thres[0]]
        targets = targets[image_area <= area_thres[1]]
        target_area_threshold.num_removed += (image_area > area_thres[1]).sum()

    return targets
target_area_threshold.num_removed=0

def target_non_max_suppression(targets, conf_thres=0.5, nms_thres=0.4, area_thres=None):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (index, label, x1, y1, x2, y2)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    if targets.shape[0] == 0:
        return targets

    targets[:, 0] = 0       # we are going to merge everythin into one image, thus index is always zero
    targets = [targets]

    output = [ torch.empty((0,6)) for _ in range(len(targets))]
    for image_i, image_pred in enumerate(targets):
        # # Filter out confidence scores below threshold
        # image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        
        # # If none are remaining => process next image
        # if not image_pred.size(0):
        #     continue

        if area_thres is not None:
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue

            x1, y1, x2, y2 = image_pred[:, 2], image_pred[:, 3], image_pred[:, 4], image_pred[:, 5]
            image_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            # image_area = image_area.abs()
            # TODO: find proper value for area_thres
            image_pred = image_pred[image_area >= area_thres[0]]

            if not image_pred.size(0):
                continue

            image_area = image_area[image_area >= area_thres[0]]

            image_pred = image_pred[image_area <= area_thres[1]]
            # if (image_area > area_thres[1]).sum() > 0:
            #     print(f'Discarded {(image_area > area_thres[1]).sum()} ground truth since it was too large')
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
        
        # Object confidence times class confidence
        # score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        # image_pred = image_pred[(-score).argsort()]
        # class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        # detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        detections = image_pred
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, 2:].unsqueeze(0), detections[:, 2:]) > nms_thres
            label_match = detections[0, 1] == detections[:, 1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 0:1] * 0 + 1  # hack to get the same shape with zeros
            # Merge overlapping bboxes by order of confidence
            detections[0, 2:] = (weights * detections[invalid, 2:]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    output = output[0]
    return output

def save_images_to_disk(path,img,detections,ground_truth,img_size,output_folder,plot_detections=True,plot_annotations=True,clip_labels=False,all_correct=None):
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    classes = ['pollen']

    # print("\nSaving images:")
    # Iterate through images and save plot of detections
    # for img_i, (path, img, detections) in enumerate(zip(img_paths, imgs, img_detections)):
    if True:
        # print("(%d) Image: '%s'" % (img_i, path))
        # annotations = ground_truth[ground_truth[:, 0] == img_i][:, 1:].cpu().numpy()
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        inch_size = fig.get_size_inches()

        # Draw bounding boxes and labels of detections
        if detections is not None and plot_detections:
            # Rescale boxes to original image

            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                color = 'green'
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none", clip_on=clip_labels,clip_box=ax.clipbox)
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                # plt.text(
                #     x1,
                #     y1,
                #     s=classes[int(cls_pred)],
                #     color="white",
                #     verticalalignment="top",
                #     clip_box=ax.clipbox, clip_on=clip_labels,
                #     bbox={"color": color, "pad": 0},
                # )

        
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"{output_folder}/{filename}.png", bbox_inches="tight", pad_inches=0.0)

        if ground_truth.size != 0 and plot_annotations:
            ground_truth[:,1:] = rescale_boxes(ground_truth[:,1:], img_size, img.shape[:2])
            for label, x1, y1, x2, y2, in ground_truth:
                box_w = x2 - x1
                box_h = y2 - y1

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor='black', facecolor="none", clip_on=clip_labels,clip_box=ax.clipbox,)
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                # plt.text(
                #     x1,
                #     y1+box_h,
                #     s=classes[int(label)]+'_gt',
                #     color="white",
                #     verticalalignment="top",
                #     clip_box=ax.clipbox, clip_on=clip_labels,
                #     bbox={"color": 'black', "pad": 0},
                # )

        if all_correct is True:
            plt.text(
                0,
                0,
                s='good',
                color="blue",
                transform=ax.transAxes,
                clip_box=ax.clipbox, clip_on=clip_labels,
            )
        elif all_correct is False:
            plt.text(
                0,
                0,
                s='bad',
                color="red",
                transform=ax.transAxes,
                clip_box=ax.clipbox, clip_on=clip_labels,
            )

        filename = path.split("/")[-1].split(".")[0]
        fig.set_size_inches(inch_size)
        plt.savefig(f"{output_folder}/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        plt.close()

def align_bb(images):
    for i in len(images):
        # Estimate perspective transform
        # Specify the number of iterations.
        number_of_iterations = 20;
        
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10;
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), images[i], M, cv2.MOTION_HOMOGRAPHY,criteria,None,1)
        w, h, _ = image.shape
        # Align image to first image
        image = cv2.warpPerspective(image, M, (h, w))

        # d_x = (M[0,0]*x + M[0,1]*y + M[0,2]) / (M[2,0]*x + M[2,1]*y + M[2,2])
        # d_y = (M[1,0]*x + M[1,1]*y + M[1,2]) / (M[2,0]*x + M[2,1]*y + M[2,2])


def evaluate(model, 
            path, 
            iou_thres, 
            conf_thres, 
            nms_thres, 
            img_size, 
            batch_size, 
            base_dir=None, 
            merge=False, 
            save_images=False, 
            save_for_verification=False,
            blur_thres=None):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False,base_dir=base_dir,blur_thres=blur_thres)

    sampler = GroupSampler(dataset.img_files)

    if merge:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, num_workers=1, collate_fn=dataset.collate_fn
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
        )


    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    target_area_threshold.num_removed = 0

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (index, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # One batch consists of all images belonging to one sample (e.g. having same timestamp)
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        batch_size = imgs.shape[0]
        # print('Batch size',imgs.shape[0])

        with torch.no_grad():
            outputs = model(imgs)
            # outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        length_thres = torch.Tensor([0.05*img_size, img_size])    # (min length, max length)
        area_thres = torch.Tensor([0.005*img_size*img_size, img_size*img_size])    # (min length, max length)
        # area_thres = length_thres*length_thres

        # print(targets) 60/1200
        # print(len(targets),len(outputs))
        # merge targets for each sample (prediction and ground truth)
        # targets = target_merge(targets,iou_thres)
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # print(targets.shape)
        targets = target_area_threshold(targets,area_thres)
        if merge:
            targets = target_non_max_suppression(targets, conf_thres=conf_thres, nms_thres=nms_thres, area_thres=area_thres)
        
        labels += targets[:, 1].tolist()
#         # print(targets.shape)
        outputs[..., :4] = xywh2xyxy(outputs[..., :4])
        # print(img_paths)
        # print(len(outputs))
        # print(len(outputs[0]))
        # outputs = torch.cat(outputs,dim=0)
        outputs = output_non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres,area_thres=area_thres,min_detections=0)

        # print(targets)
        # print(outputs)
        # outputs = torch.cat(outputs)
        # outputs = outputs.view(1,-1,outputs.shape[-1])
        # # print(outputs.shape)
        # outputs = [o for o in outputs if o is not None]
        output_mask = [o is not None for o in outputs]
        if merge:
#            index = index[output_mask]
            outputs = [o for o in outputs if o is not None]
            if len(outputs):
               outputs = torch.cat(outputs)
               outputs = outputs.view(1,-1,outputs.shape[-1])
               outputs = output_non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres,min_detections=batch_size*0.7)
#         # outputs = torch.cat(outputs)
#         # outputs = outputs.view(1,-1,outputs.shape[-1])
#         # print(outputs.shape)
#         # outputs = torch.cat(outputs).unsqueeze(0)
#         # outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        if blur_thres is not None:
            # for each output extract the segment
            # print(len(outputs))
            new_outputs = []
            for i in range(imgs.shape[0]):
                image = imgs[i]
                # print(image.max())
                # imag = cv2.cvtColor(image.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'../output/tmp/image_{batch_i}_{i}.png',imag*255) 
                output = outputs[i]
                if output is None:
                    new_outputs += [None]
                    continue
                mask = [False for _ in range(len(output))]
                bboxes = output[:,:4].clone() 
                bboxes = bboxes.round()
                bboxes[bboxes < 0] = 0
                bboxes[bboxes >= img_size] = img_size-1
                for j in range(bboxes.shape[0]):
                    x1, y1, x2, y2 = int(bboxes[j, 0]), int(bboxes[j, 1]), int(bboxes[j, 2]), int(bboxes[j, 3])
                    segments = image[:, y1:y2, x1:x2]
                    segments_gray = cv2.cvtColor(segments.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
                    lvar = cv2.Laplacian(segments_gray, cv2.CV_32F, ksize=3).var()
                    # cv2.imwrite(f'../output/tmp/out_{batch_i}_{i}_{j}_{lvar}.png',segments_gray*255)
                    if lvar > blur_thres:
                        mask[j] = True

                new_outputs += [output[mask]]
            outputs = new_outputs

            # cv2.Laplacian(image, cv2.CV_64F).var()  

        sample_metric = get_batch_statistics(copy.deepcopy(outputs), targets.clone(), iou_threshold=iou_thres)
        sample_metrics += sample_metric

        # Extract labels
        # Rescale target
        if save_images:
            os.makedirs('../output/images/',exist_ok=True)
            for i in range(imgs.shape[0]):
                # get the test results
                metric   = sample_metric[i]
                true_positives = metric[0]
                annotations = targets[targets[:, 0] == i][:, 1:].cpu().numpy()
                num_gt = len(annotations)
                num_pred = len(outputs[i]) if outputs[i] is not None else 0
                if np.array(true_positives).all() and num_pred == num_gt:
                    all_correct = True
                else:
                    all_correct = False

                # open the original files
                img_path        = Path(dataloader.dataset.img_files[index[i]])
                labels_path     = Path(dataloader.dataset.label_files[index[i]])
                save_images_to_disk(dataloader.dataset.img_files[index[i]],imgs[i].cpu().numpy().copy(),outputs[i],annotations,img_size,'../output/images/',clip_labels=True,all_correct=all_correct)

                if labels_path.exists():
                    copyfile(labels_path, '../output/images/'+labels_path.name)
        # print(sample_metrics)

        if save_for_verification:
            if merge:
                raise RuntimeError('Cannot use save_for_verification while `merge` is set to True')
            os.makedirs('../output/relabel/',exist_ok=True)
            os.makedirs('../output/correct/',exist_ok=True)
            assert len(index)==len(sample_metric)
            for i in range(len(outputs)):
                metric   = sample_metric[i]
                true_positives = metric[0]
                
                img_path        = Path(dataloader.dataset.img_files[index[i]])
                labels_path     = Path(dataloader.dataset.label_files[index[i]])

                # save false positives
                annotations = targets[targets[:, 0] == i][:, 1:].cpu().numpy()
                # print(img_path)
                # print(annotations)
                # print(len(true_positives))
                # print(len(annotations))

                if not np.array(true_positives).all() and true_positives.size > 0:
                    save_images_to_disk(str(img_path),imgs[i].cpu().numpy().copy(),outputs[i],annotations.copy(),img_size,'../output/relabel/',clip_labels=True)
                    #save_images_to_disk([str(img_path)],imgs[i:i+1].cpu().numpy().copy(),copy.deepcopy(outputs),targets.clone(),img_size,'../output/relabel/')
                    # copyfile(img_path, '../output/relabel/'+img_path.name)
                    if labels_path.exists():
                        copyfile(labels_path, '../output/relabel/'+labels_path.name)
                    # print(f'FP, copied file {img_path}')
                elif np.array(true_positives).all() and true_positives.size > 0:
                    save_images_to_disk(str(img_path),imgs[i].cpu().numpy().copy(),outputs[i],annotations.copy(),img_size,'../output/correct/',clip_labels=True)


                # save false negatives
                gt = len(annotations)
                num_pred = len(outputs[i]) if outputs[i] is not None else 0
                if num_pred != gt:
                    save_images_to_disk(str(img_path),imgs[i].cpu().numpy().copy(),outputs[i],annotations.copy(),img_size,'../output/relabel/',clip_labels=True)
                    # save_images_to_disk([str(img_path)],imgs[i:i+1].cpu().numpy().copy(),copy.deepcopy(outputs),targets.clone(),img_size,'../output/relabel/')
                    # copyfile(img_path, '../output/relabel/'+img_path.name)
                    if labels_path.exists():
                        copyfile(labels_path, '../output/relabel/'+labels_path.name)
                    # print(f'FN, copied file {img_path}')
        # if (targets[:,1]==1).any():
        #     break



    
    print('Number of targets removed',target_area_threshold.num_removed)
        # break
    # Concatenate sample statistics
    if len(sample_metrics) == 0: 
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))] 
    
    print('True positives',true_positives.sum())
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/test_20190523.data", help="path to data config file")
    parser.add_argument("--model_def", type=str, default="config/yolov3-pollen.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../weights/pollen_20190526.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="config/pollen/classes.names ", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--merge", action='store_true', help="if True every layer of a sample will be merged")
    parser.add_argument("--intheloop", action='store_true', help="stores false positives and false negatives")
    parser.add_argument("--save_images", action='store_true',  help="if True every layer will be saved")
    parser.add_argument("--blur_thres", type=float, default=None, help="laplacian variance threshold blurry image removal. If None, nothing will be removed")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    if "base_dir" in data_config:   
        base_dir_path = data_config["base_dir"]
    else:
        base_dir_path = None    
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path,map_location=device))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        base_dir=base_dir_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        merge=opt.merge,
        save_images=opt.save_images,
        save_for_verification=opt.intheloop,
        blur_thres=opt.blur_thres,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
    print('F1', f1)

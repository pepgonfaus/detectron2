# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .evaluator import DatasetEvaluator


class OpenLogoDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "supervision_type", meta.supervision, meta.split + ".txt")
        self._class_names = meta.thing_classes
        # assert meta.year in [2007, 2012], meta.year
        self._is_2007 = False
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.tolist()
                classes = instances.pred_classes.tolist()
                for box, score, cls in zip(boxes, scores, classes):
                    xmin, ymin, xmax, ymax = box
                    # The inverse of data loading logic in `datasets/pascal_voc.py`
                    xmin += 1
                    ymin += 1
                    self._predictions[cls].append(
                        f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                    )
            elif "proposals" in output:
                proposals = output["proposals"].to(self._cpu_device)
                boxes = proposals.proposal_boxes.tensor.numpy()
                scores = proposals.objectness_logits.tolist()

                for box, score in zip(boxes, scores):
                    xmin, ymin, xmax, ymax = box
                    # The inverse of data loading logic in `datasets/pascal_voc.py`
                    xmin += 1
                    ymin += 1
                    self._predictions["proposals"].append(
                        f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                    )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        if 0 and "proposals" in predictions:

            from detectron2.utils.logger import create_small_table

            self._logger.info("Evaluating bbox proposals ...")
            res = {}
            areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
            for limit in [100, 1000]:
                for area, suffix in areas.items():
                    stats = self._evaluate_box_proposals(
                        self._predictions, self._coco_api, area=area, limit=limit
                    )
                    key = "AR{}@{:d}".format(suffix, limit)
                    res[key] = float(stats["ar"].item() * 100)
            self._logger.info("Proposal metrics: \n" + create_small_table(res))
            return res

        else:
            self._logger.info(
                "Evaluating {} using {} metric. "
                "Note that results do not use the official Matlab API.".format(
                    self._dataset_name, 2007 if self._is_2007 else 2012
                )
            )

            with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
                res_file_template = os.path.join(dirname, "{}.txt")
                aps_cls = defaultdict()
                aps = defaultdict(list)  # iou -> ap per class
                # recs = defaultdict(list)  # iou -> ap per class
                # precs = defaultdict(list)  # iou -> ap per class
                categories = self._class_names
                if "proposals" in predictions:
                    categories = ["proposals"]
                for cls_id, cls_name in enumerate(categories):
                    if cls_name == "proposals":
                        cls_id = "proposals"
                    lines = predictions.get(cls_id, [""])

                    with open(res_file_template.format(cls_name), "w") as f:
                        f.write("\n".join(lines))

                    for thresh in range(50, 100, 5):
                        rec, prec, ap = voc_eval(
                            res_file_template,
                            self._anno_file_template,
                            self._image_set_path,
                            cls_name,
                            ovthresh=thresh / 100.0,
                            use_07_metric=self._is_2007,

                        )
                        aps[thresh].append(ap * 100)
                        # recs[thresh].append(rec * 100)
                        # precs[thresh].append(prec * 100)
                        # if thresh == 50:
                        #     aps_cls[cls_name] = ap * 100

            ret = OrderedDict()
            mAP = {iou: np.mean(x) for iou, x in aps.items()}
            # mREC = {iou: np.mean(x) for iou, x in recs.items()}
            # mPREC = {iou: np.mean(x) for iou, x in precs.items()}
            ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
            # for cls in aps_cls:
            #     ret["bbox"]["_AP50_"+cls] = aps_cls[cls]
            # ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75],
            #                "REC": np.mean(list(mREC.values())), "REC50": mREC[50], "REC75": mREC[75],
            #                "PREC": np.mean(list(mPREC.values())), "PREC50": mPREC[50], "PREC75": mPREC[75]}
            return ret




    # inspired from Detectron:
    # https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
    def _evaluate_box_proposals(self, dataset_predictions, _coco_api, thresholds=None, area="all", limit=None):
        """
        Evaluate detection proposal recall metrics. This function is a much
        faster alternative to the official COCO API recall evaluation code. However,
        it produces slightly different results.
        """
        from detectron2.structures import Boxes, BoxMode, pairwise_iou

        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {
            "all": 0,
            "small": 1,
            "medium": 2,
            "large": 3,
            "96-128": 4,
            "128-256": 5,
            "256-512": 6,
            "512-inf": 7,
        }
        area_ranges = [
            [0 ** 2, 1e5 ** 2],  # all
            [0 ** 2, 32 ** 2],  # small
            [32 ** 2, 96 ** 2],  # medium
            [96 ** 2, 1e5 ** 2],  # large
            [96 ** 2, 128 ** 2],  # 96-128
            [128 ** 2, 256 ** 2],  # 128-256
            [256 ** 2, 512 ** 2],  # 256-512
            [512 ** 2, 1e5 ** 2],
        ]  # 512-inf
        assert area in areas, "Unknown area range: {}".format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = []
        num_pos = 0

        for prediction_dict in dataset_predictions:
            predictions = prediction_dict["proposals"]

            # sort predictions in descending order
            # TODO maybe remove this and make it explicit in the documentation
            inds = predictions.objectness_logits.sort(descending=True)[1]
            predictions = predictions[inds]

            ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
            anno = coco_api.loadAnns(ann_ids)
            gt_boxes = [
                BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                for obj in anno
                if obj["iscrowd"] == 0
            ]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
            gt_boxes = Boxes(gt_boxes)
            gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

            if len(gt_boxes) == 0 or len(predictions) == 0:
                continue

            valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
            gt_boxes = gt_boxes[valid_gt_inds]

            num_pos += len(gt_boxes)

            if len(gt_boxes) == 0:
                continue

            if limit is not None and len(predictions) > limit:
                predictions = predictions[:limit]

            overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

            _gt_overlaps = torch.zeros(len(gt_boxes))
            for j in range(min(len(predictions), len(gt_boxes))):
                # find which proposal box maximally covers each gt box
                # and get the iou amount of coverage for each gt box
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)

                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ovr, gt_ind = max_overlaps.max(dim=0)
                assert gt_ovr >= 0
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert _gt_overlaps[j] == gt_ovr
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            # append recorded iou coverage level
            gt_overlaps.append(_gt_overlaps)
        gt_overlaps = torch.cat(gt_overlaps, dim=0)
        gt_overlaps, _ = torch.sort(gt_overlaps)

        if thresholds is None:
            step = 0.05
            thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
        recalls = torch.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {
            "ar": ar,
            "recalls": recalls,
            "thresholds": thresholds,
            "gt_overlaps": gt_overlaps,
            "num_pos": num_pos,
        }


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        if classname == "proposals":
            R = recs[imagename]
        else:
            R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

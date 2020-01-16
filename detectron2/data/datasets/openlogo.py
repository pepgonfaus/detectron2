# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

import tqdm

__all__ = ["register_openlogo"]

# fmt: off
CLASS_NAMES = ['3m', 'abus', 'accenture', 'adidas', 'adidas1', 'adidas_text', 'airhawk', 'airness', 'aldi', 'aldi_text',
               'alfaromeo', 'allett', 'allianz', 'allianz_text', 'aluratek', 'aluratek_text', 'amazon', 'amcrest',
               'amcrest_text', 'americanexpress', 'americanexpress_text', 'android', 'anz', 'anz_text', 'apc',
               'apecase', 'apple', 'aquapac_text', 'aral', 'armani', 'armitron', 'aspirin', 'asus', 'at_and_t',
               'athalon', 'audi', 'audi_text', 'axa', 'bacardi', 'bankofamerica', 'bankofamerica_text', 'barbie',
               'barclays', 'base', 'basf', 'batman', 'bayer', 'bbc', 'bbva', 'becks', 'bellataylor', 'bellodigital',
               'bellodigital_text', 'bem', 'benrus', 'bershka', 'bfgoodrich', 'bik', 'bionade', 'blackmores',
               'blizzardentertainment', 'bmw', 'boeing', 'boeing_text', 'bosch', 'bosch_text', 'bottegaveneta',
               'bridgestone', 'bridgestone_text', 'budweiser', 'budweiser_text', 'bulgari', 'burgerking',
               'burgerking_text', 'calvinklein', 'canon', 'carglass', 'carlsberg', 'carters', 'cartier', 'caterpillar',
               'chanel', 'chanel_text', 'cheetos', 'chevrolet', 'chevrolet_text', 'chevron', 'chickfila', 'chimay',
               'chiquita', 'cisco', 'citi', 'citroen', 'citroen_text', 'coach', 'cocacola', 'coke', 'colgate',
               'comedycentral', 'converse', 'corona', 'corona_text', 'costa', 'costco', 'cpa_australia', 'cvs',
               'cvspharmacy', 'danone', 'dexia', 'dhl', 'disney', 'doritos', 'drpepper', 'dunkindonuts', 'ebay', 'ec',
               'erdinger', 'espn', 'esso', 'esso_text', 'evernote', 'facebook', 'fedex', 'ferrari', 'firefox',
               'firelli', 'fly_emirates', 'ford', 'fosters', 'fritolay', 'fritos', 'gap', 'generalelectric', 'gildan',
               'gillette', 'goodyear', 'google', 'gucci', 'guinness', 'hanes', 'head', 'head_text', 'heineken',
               'heineken_text', 'heraldsun', 'hermes', 'hersheys', 'hh', 'hisense', 'hm', 'homedepot', 'homedepot_text',
               'honda', 'honda_text', 'hp', 'hsbc', 'hsbc_text', 'huawei', 'huawei_text', 'hyundai', 'hyundai_text',
               'ibm', 'ikea', 'infiniti', 'infiniti_text', 'intel', 'internetexplorer', 'jackinthebox', 'jacobscreek',
               'jagermeister', 'jcrew', 'jello', 'johnnywalker', 'jurlique', 'kelloggs', 'kfc', 'kia', 'kitkat',
               'kodak', 'kraft', 'lacoste', 'lacoste_text', 'lamborghini', 'lays', 'lego', 'levis', 'lexus',
               'lexus_text', 'lg', 'londonunderground', 'loreal', 'lotto', 'luxottica', 'lv', 'marlboro',
               'marlboro_fig', 'marlboro_text', 'maserati', 'mastercard', 'maxwellhouse', 'maxxis', 'mccafe',
               'mcdonalds', 'mcdonalds_text', 'medibank', 'mercedesbenz', 'mercedesbenz_text', 'michelin', 'microsoft',
               'milka', 'millerhighlife', 'mini', 'miraclewhip', 'mitsubishi', 'mk', 'mobil', 'motorola', 'mtv', 'nasa',
               'nb', 'nbc', 'nescafe', 'netflix', 'nike', 'nike_text', 'nintendo', 'nissan', 'nissan_text', 'nivea',
               'northface', 'nvidia', 'obey', 'olympics', 'opel', 'optus', 'optus_yes', 'oracle', 'pampers',
               'panasonic', 'paulaner', 'pepsi', 'pepsi_text', 'pepsi_text1', 'philadelphia', 'philips', 'pizzahut',
               'pizzahut_hut', 'planters', 'playstation', 'poloralphlauren', 'porsche', 'porsche_text', 'prada', 'puma',
               'puma_text', 'quick', 'rbc', 'recycling', 'redbull', 'redbull_text', 'reebok', 'reebok1', 'reebok_text',
               'reeses', 'renault', 'republican', 'rittersport', 'rolex', 'rolex_text', 'ruffles', 'samsung',
               'santander', 'santander_text', 'sap', 'schwinn', 'scion_text', 'sega', 'select', 'shell', 'shell_text',
               'shell_text1', 'siemens', 'singha', 'skechers', 'sony', 'soundcloud', 'soundrop', 'spar', 'spar_text',
               'spiderman', 'sprite', 'standard_liege', 'starbucks', 'stellaartois', 'subaru', 'subway', 'sunchips',
               'superman', 'supreme', 'suzuki', 't-mobile', 'tacobell', 'target', 'target_text', 'teslamotors',
               'texaco', 'thomsonreuters', 'tigerwash', 'timberland', 'tissot', 'tnt', 'tommyhilfiger', 'tostitos',
               'total', 'toyota', 'toyota_text', 'tsingtao', 'twitter', 'umbro', 'underarmour', 'unicef', 'uniqlo',
               'uniqlo1', 'unitednations', 'ups', 'us_president', 'vaio', 'velveeta', 'venus', 'verizon',
               'verizon_text', 'visa', 'vodafone', 'volkswagen', 'volkswagen_text', 'volvo', 'walmart', 'walmart_text',
               'warnerbros', 'wellsfargo', 'wellsfargo_text', 'wii', 'williamhill', 'windows', 'wordpress', 'xbox',
               'yahoo', 'yamaha', 'yonex', 'yonex_text', 'youtube', 'zara']
# fmt: on


def load_openlogo_instances(dirname: str, supervision: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "supervision_type", supervision, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # cls_set = set()
    dicts = []
    for fileid in tqdm.tqdm(fileids, desc='Reading annotations'):
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # if cls not in cls_set:
            #     cls_set.add(cls)
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0

            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_openlogo(name, dirname, split, supervision):
    DatasetCatalog.register(name, lambda: load_openlogo_instances(dirname, supervision, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, supervision=supervision, split=split
    )

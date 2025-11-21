import json
import os
from copy import deepcopy
import pickle
from typing import Dict, List, Set, Tuple
import numpy as np
from functools import cache

from utils import ask_question, key_points_to_bounding_box, bounding_box_iou
from thefuzz import fuzz

DATASET_PATH = os.getenv("DATASET_PATH", "./final_labeling")

FACE_ATTR_NAMES = ['5 oClock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes', 'Bald', 'Bangs', 'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby', 'Double Chin', 'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones', 'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin', 'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair', 'Wearing Earrings', 'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace', 'Wearing Necktie', 'Young']
FACE_ATTR_NAMES_EXTEND = ['5 oClock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes', 'Bald', 'Bangs', 'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby', 'Double Chin', 'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones', 'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin', 'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair', 'Wearing Earrings', 'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace', 'Wearing Necktie', 'Young', 'Mature', "Female", "Face Up", "Face Down", "Face Right Side of Image", "Face Left Side of Image"]
FACE_ATTR_ADMIT_THRESHOLD = np.array([0.80, 0.70, 0.80, 0.50, 0.20, 0.90,                                   0.70,       0.95,       0.50,        0.50,         0.70,     0.70,         0.80,             0.95,      0.80,         0.92,         0.50,      0.50,        0.75,           0.80,              0.98,   0.995,                0.30,        0.75,         0.98,       0.70,        0.40,        0.40,           0.60,                0.14,         0.60,        0.80,      0.70,            0.70,        0.70,                0.75,         0.60,               0.50,               0.70,              0.98])
FACE_ATTR_DENY_THRESHOLD =  np.array([0.10, 0.05, 0.02, 0.05, 0.01, 0.05,                                   0.05,       0.05,       0.005,       0.01,         0.05,     0.01,         0.30,             0.40,      0.10,         0.005,        0.02,      0.005,       0.003,          0.01,              0.002,  0.02,                 0.005,       0.08,         0.40,       0.08,        0.004,       0.02,           0.01,                0.001,        0.02,        0.04,      0.05,            0.05,        0.02,                0.001,        0.01,               0.01,               0.001,             0.50])
FACE_ATTR_DESCRIPTIONS = {
    "5 oClock Shadow": ("Has 5 o'clock shadow", "No 5 o'clock shadow"),
    "Arched Eyebrows": ("Eyebrows are arched", "Eyebrows not arched"),
    "Attractive": ("Attractive face", "Not attractive"),
    "Bags Under Eyes": ("Bags under eyes", "No bags under eyes"),
    "Bald": ("Bald head", "Not bald"),
    "Bangs": ("Has bangs", "No bangs"),
    "Big Lips": ("Big lips", "Small lips"),
    "Big Nose": ("Big nose", "Small nose"),
    "Black Hair": ("Black hair", "Not black hair"),
    "Blond Hair": ("Blond hair", "Not blond hair"),
    "Blurry": ("Image is blurry", "Image is clear"),
    "Brown Hair": ("Brown hair", "Not brown hair"),
    "Bushy Eyebrows": ("Bushy eyebrows", "Not bushy eyebrows"),
    "Chubby": ("Chubby face", "Not chubby"),
    "Double Chin": ("Double chin", "No double chin"),
    "Eyeglasses": ("Wearing eyeglasses", "Not wearing eyeglasses"),
    "Goatee": ("Has goatee", "No goatee"),
    "Gray Hair": ("Gray hair", "Not gray hair"),
    "Heavy Makeup": ("Wearing heavy makeup", "No heavy makeup"),
    "High Cheekbones": ("High cheekbones", "Not high cheekbones"),
    "Male": ("Male", "Female"),
    "Female": ("Female", "Male"),
    "Mouth Slightly Open": ("Mouth slightly open", "Mouth closed"),
    "Mustache": ("Has mustache", "No mustache"),
    "Narrow Eyes": ("Narrow eyes", "Not narrow eyes"),
    "No Beard": ("No beard", "Has beard"),
    "Oval Face": ("Oval face", "Not oval face"),
    "Pale Skin": ("Pale skin", "Not pale skin"),
    "Pointy Nose": ("Pointy nose", "Not pointy nose"),
    "Receding Hairline": ("Receding hairline", "Not receding hairline"),
    "Rosy Cheeks": ("Rosy cheeks", "Not rosy cheeks"),
    "Sideburns": ("Has sideburns", "No sideburns"),
    "Smiling": ("Smiling", "Not smiling"),
    "Straight Hair": ("Straight hair", "Not straight hair"),
    "Wavy Hair": ("Wavy hair", "Not wavy hair"),
    "Wearing Earrings": ("Wearing earrings", "Not wearing earrings"),
    "Wearing Hat": ("Wearing hat", "Not wearing hat"),
    "Wearing Lipstick": ("Wearing lipstick", "Not wearing lipstick"),
    "Wearing Necklace": ("Wearing necklace", "Not wearing necklace"),
    "Wearing Necktie": ("Wearing necktie", "Not wearing necktie"),
    "Young": ("Young face", "Not young"),
    "Mature": ("Mature face", "Not mature"),
    "Face Up": ("Face turned up", "Face not turned up"),
    "Face Down": ("Face turned down", "Face not turned down"),
    "Face Right Side of Image": ("Face turned to right side of image", "Face not turned to right side of image"),
    "Face Left Side of Image": ("Face turned to left side of image", "Face not turned to left side of image"),
}
POSITION_SIMPLIFIER = {
    'headscarf': "head",
    'shoulder': "body", 
    'ears': "head",
    'thighs': "thigh", 
    'hands': "hand",
    'right eye': "face",
    'right ear': "head",
    'right half of the face': "face", 
    'mirror': "hand", 
    'nose': "face", 
    'forehead': "head", 
    'eyes': "face", 
    'wrists': "hand", 
    'arm': "hand", 
    'tongue': "face", 
    'lip': "face", 
    'eyebrows': "face", 
    'himself': "body", 
    'back':"body", 
    'left arm': "hand", 
    'reins': "hand",
    'hair': "head", 
    'lap': "thigh", 
    'mouth': "face", 
    'left shoulder': "body", 
    'pen': "hand", 
    'mask': "face", 
    'left chest': "body"
}

ALL_POSITIONS = ["hand", "both hands", "body", "face", "neck", "legs", "thigh", "head", "left hand", "right hand", "foot"]
POSITION_INCLUDE_MAP = {
    "hand": ["left hand", "right hand", "both hands"],
    "body": ["legs", "thigh"],
    "head": ["face", "neck"],
    "face": ["head"],
    "neck": ["head"],
    "legs": ["body"],
    "thigh": ["body"]
}
POSITION_EXCLUDE_MAP = {
    "hand": ["left hand", "right hand", "both hands", "body", "thigh"],
    "body": ["legs", "thigh"],
    "head": ["face", "neck"],
    "left hand": ["hand", "both hands", "body", "thigh"],
    "right hand": ["hand", "both hands", "body", "thigh"],
    "face": ["head"],
    "neck": ["head"]
}


_full_data = None

class HoiObject:
    def __init__(self, data):
        self.raw_data = data
        self.box = data.get("box", None)
    def get_name(self):
        return self.raw_data.get("name", "")

class Hoi:
    def __init__(self, data, obj: HoiObject):
        self.raw_data = data
        self.obj: HoiObject = obj
    def get_actions(self):
        return set([i[1] for i in self.raw_data.get("action", [])])
    def get_positions(self):
        org_pose = set([i[0] for i in self.raw_data.get("action", [])])
        return set([POSITION_SIMPLIFIER.get(p, p) for p in org_pose])
    def get_object_box(self):
        return self.obj.box
    def get_object_names(self):
        return self.obj.raw_data.get("possible_names", [])
    def get_object_name(self):
        return self.obj.raw_data.get("name", "")
    def get_negative_actions(self):
        return self.raw_data.get("negative_action", [])
    def get_position_action_pairs(self):
        return set((POSITION_SIMPLIFIER.get(i[0], i[0]), i[1]) for i in self.raw_data.get("action", []))

class Person:
    def __init__(self, data, detect_results):
        self.raw_data = data
        self.hois: List[Hoi] = []
        if data.get("without_face") is not True and data.get("face_box") is not None:
            self.face_box:List[float] = detect_results["face_boxes"][data.get("face_box")]
        else:
            self.face_box = None

        if data.get("body_box") is not None:
            self.body_box:List[float] = detect_results["body_boxes"][data.get("body_box")]
        else:
            self.body_box = None

        if data.get("skeleton") is not None:
            self.skeleton:List[List[float]] = detect_results["skeletons"][data.get("skeleton")]
        else:
            self.skeleton = None

    def init_hoi_objects(self, objs: list[HoiObject]):
        for hoi in self.raw_data.get("hoi", []):
            if "no interaction" in [i[1] for i in hoi["relationship"]["action"]]:
                continue

            if hoi.get("deleted") is not True and objs[hoi.get("object")] is not None:
                self.hois.append(Hoi(hoi["relationship"], objs[hoi.get("object")]))

    def get_face_box(self):
        return self.raw_data.get("face_box", None)
    
    def detailing_property(self, key, default=None):
        return self.raw_data.get("qwen_detailing", {}).get(key, default)

    @cache
    def face_area(self):
        """..............，.......1"""
        if self.face_box is not None:
            return (self.face_box[3] - self.face_box[1]) * (self.face_box[2] - self.face_box[0])
        return 0
    
    @cache
    def body_area(self):
        """..............，.......1"""
        if self.body_box is not None:
            return (self.body_box[3] - self.body_box[1]) * (self.body_box[2] - self.body_box[0])
        return 0

    def get_face_attr_vec(self, attr_names = None):
        if self.raw_data.get("facex_detailing"):
            if attr_names is not None:
                return np.array([self.raw_data["facex_detailing"]["attributes"].get(name, 0) for name in attr_names])
            return np.array([i for i in self.raw_data["facex_detailing"]["attributes"].values()])
        return None

    @cache
    def get_face_attr_admit_list(self, additional = True):
        if self.raw_data.get("facex_detailing"):
            ans = []
            feat_vec = self.get_face_attr_vec()
            admit_vec = feat_vec >= FACE_ATTR_ADMIT_THRESHOLD
            ans = [name for name, admitted in zip(FACE_ATTR_NAMES, admit_vec) if admitted]
            if additional:
                ref = self.get_face_attr_deny_list(additional=False)
                if "Young" in ref:
                    ans.append("Mature")
                if "Male" in ref:
                    ans.append("Female")
                if self.raw_data["facex_detailing"]["headpose"]["pitch"] < -15 and self.raw_data["facex_detailing"]["headpose"]["pitch"] > -60:
                    ans.append("Face Down")
                elif self.raw_data["facex_detailing"]["headpose"]["pitch"] > 15 and self.raw_data["facex_detailing"]["headpose"]["pitch"] < 60:
                    ans.append("Face Up")
                if self.raw_data["facex_detailing"]["headpose"]["yaw"] < -15 and self.raw_data["facex_detailing"]["headpose"]["yaw"] > -120:
                    ans.append("Face Right Side of Image")
                elif self.raw_data["facex_detailing"]["headpose"]["yaw"] > 15 and self.raw_data["facex_detailing"]["headpose"]["yaw"] < 120:
                    ans.append("Face Left Side of Image")
            return frozenset(ans)
        return frozenset()
    
    @cache
    def get_face_attr_deny_list(self, additional = True):
        if self.raw_data.get("facex_detailing"):
            ans = []
            feat_vec = self.get_face_attr_vec()
            deny_vec = feat_vec < FACE_ATTR_ADMIT_THRESHOLD
            ans = [name for name, denied in zip(FACE_ATTR_NAMES, deny_vec) if denied]
            if additional:
                ref = self.get_face_attr_admit_list(additional=False)
                if "Young" in ref:
                    ans.append("Mature")
                if "Male" in ref:
                    ans.append("Female")
                if self.raw_data["facex_detailing"]["headpose"]["pitch"] < -15 and self.raw_data["facex_detailing"]["headpose"]["pitch"] > -60:
                    ans.append("Face Down")
                elif self.raw_data["facex_detailing"]["headpose"]["pitch"] > 15 and self.raw_data["facex_detailing"]["headpose"]["pitch"] < 60:
                    ans.append("Face Up")
                if self.raw_data["facex_detailing"]["headpose"]["yaw"] < -15 and self.raw_data["facex_detailing"]["headpose"]["yaw"] > -120:
                    ans.append("Face Right Side of Image")
                elif self.raw_data["facex_detailing"]["headpose"]["yaw"] > 15 and self.raw_data["facex_detailing"]["headpose"]["yaw"] < 120:
                    ans.append("Face Left Side of Image")
            return frozenset(ans)
        return frozenset()

    def get_face_attr_assert_belief(self, admit_set, deny_set):
        if self.raw_data.get("facex_detailing"):
            result = 1.0
            for admit in admit_set:
                result *= self.raw_data["facex_detailing"]["attributes"].get(admit, 0)
            for deny in deny_set:
                result *= (1 - self.raw_data["facex_detailing"]["attributes"].get(deny, 0))
            return result
        return 0

    def get_clothing_list(self, only_confident = False):
        clothings = self.raw_data.get("qwen_detailing", {}).get("clothing", [])
        if isinstance(clothings, list):
            clothings = clothings
        elif isinstance(clothings, dict):
            if only_confident and clothings["vague"]:
                clothings = []
            else:
                clothings = clothings["clothing"]
        if only_confident:
            clothings = [c for c in clothings if c.get("belonging_confident", True) and c.get("existence_confident", True)]
        return clothings
    
    def full_feature_set(self, body_boxes = False) -> List[Tuple[Dict]]:
        """........"""
        feature_set = []
        # ....
        if self.face_box is not None and self.raw_data.get("facex_detailing") and self.detailing_property("face_seen", False):
            if self.face_area() > 0.05:
                for attr_name, attr_value, accept_thresh, deny_thresh in zip(FACE_ATTR_NAMES, self.get_face_attr_vec(), FACE_ATTR_ADMIT_THRESHOLD, FACE_ATTR_DENY_THRESHOLD):
                    # ........，...
                    if attr_name not in ['5 oClock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes', 'Bald', 'Bangs', 'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby', 'Double Chin', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin', 'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair']:
                        continue
                    if attr_value >= accept_thresh:
                        feature_set.append( {"attr_type":"facial", "attr_name": attr_name, "attr_value": True} )
                    elif attr_value < deny_thresh:
                        feature_set.append( {"attr_type":"facial", "attr_name": attr_name, "attr_value": False} )
                    else:
                        feature_set.append( {"attr_type":"facial", "attr_name": attr_name, "attr_value": None} )
                # ..landmark
                if self.skeleton is not None:
                    facex_point_set = np.array(self.raw_data["facex_detailing"]["landmarks"])
                    wpose_point_set = np.array(self.skeleton["dw_face"])
                    facex_nose = key_points_to_bounding_box(facex_point_set[[27,28,29,30,31,32,33,34,35]])  # facex.....
                    wpose_nose = key_points_to_bounding_box(wpose_point_set[[27,28,29,30,31,32,33,34,35]])  # wpose.....
                    if bounding_box_iou(facex_nose, wpose_nose) > 0:
                        feature_set.append( {"attr_type":"bbox", "attr_name": "nose", "attr_value": facex_nose} )
                    facex_mouth = key_points_to_bounding_box(facex_point_set[[48,49,50,51,52,53,54,55,56,57,58,59]])  # facex.....
                    wpose_mouth = key_points_to_bounding_box(wpose_point_set[[48,49,50,51,52,53,54,55,56,57,58,59]])  # wpose.....
                    if bounding_box_iou(facex_mouth, wpose_mouth) > 0:
                        feature_set.append( {"attr_type":"bbox", "attr_name": "mouth", "attr_value": facex_mouth} )
                    facex_leye = key_points_to_bounding_box(facex_point_set[[42,43,44,45,46,47]])  # facex.....
                    wpose_leye = key_points_to_bounding_box(wpose_point_set[[42,43,44,45,46,47]])  # wpose.....
                    if bounding_box_iou(facex_leye, wpose_leye) > 0:
                        feature_set.append( {"attr_type":"bbox", "attr_name": "left_eye", "attr_value": facex_leye} )
                    facex_reye = key_points_to_bounding_box(facex_point_set[[36,37,38,39,40,41]])  # facex.....
                    wpose_reye = key_points_to_bounding_box(wpose_point_set[[36,37,38,39,40,41]])  # wpose.....
                    if bounding_box_iou(facex_reye, wpose_reye) > 0:
                        feature_set.append( {"attr_type":"bbox", "attr_name": "right_eye", "attr_value": facex_reye} )
                    facex_leyebrow = key_points_to_bounding_box(facex_point_set[[22,23,24,25,26]])  # facex.....
                    wpose_leyebrow = key_points_to_bounding_box(wpose_point_set[[22,23,24,25,26]])  # wpose.....
                    if bounding_box_iou(facex_leyebrow, wpose_leyebrow) > 0:
                        feature_set.append( {"attr_type":"bbox", "attr_name": "left_eyebrow", "attr_value": facex_leyebrow} )
                    facex_reyebrow = key_points_to_bounding_box(facex_point_set[[17,18,19,20,21]])  # facex.....
                    wpose_reyebrow = key_points_to_bounding_box(wpose_point_set[[17,18,19,20,21]])  # wpose.....
                    if bounding_box_iou(facex_reyebrow, wpose_reyebrow) > 0:
                        feature_set.append( {"attr_type":"bbox", "attr_name": "right_eyebrow", "attr_value": facex_reyebrow} )
            # ....
            if self.raw_data["facex_detailing"]["headpose"]["pitch"] < -15 and self.raw_data["facex_detailing"]["headpose"]["pitch"] > -60:
                feature_set.append( {"attr_type":"facial", "attr_name": "pitch", "attr_value": "down", "real_value": self.raw_data["facex_detailing"]["headpose"]["pitch"]} )
            elif self.raw_data["facex_detailing"]["headpose"]["pitch"] > 15 and self.raw_data["facex_detailing"]["headpose"]["pitch"] < 60:
                feature_set.append( {"attr_type":"facial", "attr_name": "pitch", "attr_value": "up", "real_value": self.raw_data["facex_detailing"]["headpose"]["pitch"]} )
            else:
                feature_set.append( {"attr_type":"facial", "attr_name": "pitch", "attr_value": None, "real_value": self.raw_data["facex_detailing"]["headpose"]["pitch"]} )

            if self.raw_data["facex_detailing"]["headpose"]["yaw"] < -15 and self.raw_data["facex_detailing"]["headpose"]["yaw"] > -120:
                feature_set.append( {"attr_type":"facial", "attr_name": "yaw", "attr_value": "left", "real_value": self.raw_data["facex_detailing"]["headpose"]["yaw"]} )
            elif self.raw_data["facex_detailing"]["headpose"]["yaw"] > 15 and self.raw_data["facex_detailing"]["headpose"]["yaw"] < 120:
                feature_set.append( {"attr_type":"facial", "attr_name": "yaw", "attr_value": "right", "real_value": self.raw_data["facex_detailing"]["headpose"]["yaw"]} )
            else:
                feature_set.append( {"attr_type":"facial", "attr_name": "yaw", "attr_value": None, "real_value": self.raw_data["facex_detailing"]["headpose"]["yaw"]} )
            # ....
            feature_set.append( {"attr_type":"bbox", "attr_name": "face", "attr_value": self.face_box} )

        # qwen ....
        if self.raw_data.get("qwen_detailing"):
            for key in ["age", "gender", "emotion", "race"]:
                feature_set.append( {"attr_type":"overall", "attr_name": key, "attr_value": None if self.raw_data["qwen_detailing"][key] in ["unknown", "complex"] else self.raw_data["qwen_detailing"][key]} )
            if self.raw_data["qwen_detailing"].get("text") != "no_text":
                feature_set.append( {"attr_type":"overall", "attr_name": "text", "attr_value": self.raw_data["qwen_detailing"]["text"]} )

        # ....
        for clothing in self.get_clothing_list(only_confident=True):
            feature_set.append( {"attr_type":"clothing", "attr_name": "clothing", "attr_value": {"name": clothing["name"], "color": clothing["color"], "type": clothing["type"]}} )

        # ....
        if self.body_box is not None and self.body_area() > 0.1 and len([(x,y)  for x,y in self.skeleton["dw_body"] if x> 0 and y>0]) >= 10:
            feature_set.append( {"attr_type":"bbox", "attr_name": "body", "attr_value": self.body_box} )

        if body_boxes and self.body_box is not None and self.skeleton is not None:
            left_hand_points = np.array(self.skeleton["dw_hand_1"])
            right_hand_points = np.array(self.skeleton["dw_hand_2"])
            left_foot_points = np.array(self.skeleton["dw_foot_1"])
            right_foot_points = np.array(self.skeleton["dw_foot_2"])
            bx1, by1, bx2, by2 = self.body_box
            # .........body box.，.......
            if np.any((left_hand_points[:,0] >= bx1) & (left_hand_points[:,0] <= bx2) & (left_hand_points[:,1] >= by1) & (left_hand_points[:,1] <= by2)):
                feature_set.append( {"attr_type":"bbox", "attr_name": "left_hand", "attr_value": key_points_to_bounding_box(left_hand_points)} )
            if np.any((right_hand_points[:,0] >= bx1) & (right_hand_points[:,0] <= bx2) & (right_hand_points[:,1] >= by1) & (right_hand_points[:,1] <= by2)):
                feature_set.append( {"attr_type":"bbox", "attr_name": "right_hand", "attr_value": key_points_to_bounding_box(right_hand_points)} )
            if np.any((left_foot_points[:,0] >= bx1) & (left_foot_points[:,0] <= bx2) & (left_foot_points[:,1] >= by1) & (left_foot_points[:,1] <= by2)):
                feature_set.append( {"attr_type":"bbox", "attr_name": "left_foot", "attr_value": key_points_to_bounding_box(left_foot_points)} )
            if np.any((right_foot_points[:,0] >= bx1) & (right_foot_points[:,0] <= bx2) & (right_foot_points[:,1] >= by1) & (right_foot_points[:,1] <= by2)):
                feature_set.append( {"attr_type":"bbox", "attr_name": "right_foot", "attr_value": key_points_to_bounding_box(right_foot_points)} )

        # .-.....
        for hoi in self.hois:
            if len(hoi.get_position_action_pairs()) == 0:
                continue
            feature_set.append( {"attr_type":"hoi", "attr_name": "hoi", "attr_value": {"relation": hoi.get_position_action_pairs(), "object": hoi.get_object_name(), "bbox": hoi.get_object_box()}} )


        return feature_set
    
    def hand_cant_swap(self):
        """........，......."""
        left_hand_items = set()
        right_hand_items = set()
        for hoi in self.hois:
            for pos, action in hoi.get_position_action_pairs():
                if pos in ["left hand"]:
                    left_hand_items.add(hoi.get_object_name())
                if pos in ["right hand"]:
                    right_hand_items.add(hoi.get_object_name())
        return len(left_hand_items & right_hand_items) > 0
    def has_multi_hand_hoi(self):
        appeared_pos = set()
        for hoi in self.hois:
            for pos, action in hoi.get_position_action_pairs():
                appeared_pos.add(pos)
                if pos in ["both hands", "hand"]:
                    return True
        if "left hand" in appeared_pos and "right hand" in appeared_pos:
            return True
        return False

class Picture:
    def __init__(self, data):
        self.raw_data = data
        self.persons:List[Person] = [Person(p, data["detect_results"]) for p in data.get("persons", []) if p.get("deleted") is not True]
        self.hoi_objects: List[HoiObject] = []
        for obj in data.get("objects", []):
            if obj.get("deleted") is not True:
                self.hoi_objects.append(HoiObject(obj))
            else:
                self.hoi_objects.append(None)
        for person in self.persons:
            person.init_hoi_objects(self.hoi_objects)

    def image_path(self):
        return os.path.join(DATASET_PATH, self.raw_data.get("image_path").split("/")[-1])

    def full_hoi(self):
        result = []
        for person in self.persons:
            result.extend(person.hois)
        return result

    def object_names(self):
        result = []
        for obj in self.hoi_objects:
            if obj is not None:
                result.append(obj.get_name())
        return result

def get_full_data():
    global _full_data
    if _full_data is not None:
        return _full_data
    if os.path.exists(os.path.join(DATASET_PATH, "full_data.pkl")):
        with open(os.path.join(DATASET_PATH, "full_data.pkl"), "rb") as f:
            _full_data = pickle.load(f)
            return deepcopy(_full_data)
    data = []
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".json"):
            with open(os.path.join(DATASET_PATH, filename), "r") as f:
                file_data = json.load(f)
                data.append(file_data)
    _full_data = data
    if not os.path.exists(os.path.join(DATASET_PATH, "full_data.pkl")):
        with open(os.path.join(DATASET_PATH, "full_data.pkl"), "wb") as f:
            pickle.dump(_full_data, f)
    return deepcopy(_full_data)

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    print("Warning: unknown type in json dump:", type(obj))
    raise TypeError

cloth_desc_cache = {}

def get_cloth_description(name, colors):
    global cloth_desc_cache
    if (len(cloth_desc_cache) == 0) and os.path.exists("clothing_descriptions.json"):
        with open("clothing_descriptions.json", "r") as f:
            cloth_desc_cache = json.load(f)
    indexer = f"{name}||{','.join(sorted(colors))}"
    if indexer in cloth_desc_cache:
        return cloth_desc_cache[indexer]
    desc = ask_question(f"Please make a short description of a clothing item named '{name}' with colors '{', '.join(colors)}'. Provide your answer directly include all the words of its name and colors unless it is meaningless word like 'unknown'. Your final answer should be as short as possible.")
    cloth_desc_cache[indexer] = desc
    return desc

hoi_desc_cache = {}

import random
def get_hoi_description(name, relations, no_obj_name = False, no_pos = False):
    pos, act = random.choice(list(relations))
    if no_obj_name:
        if pos == "standalone":
            return f"body part: no direct contact, action: {act}"
        else:
            return f"body part: {pos}, action: {act}"
    else:
        if no_pos:
            return f"action: {act}, object: {name}"
        else:
            if pos == "standalone":
                return f"body part: no direct contact, action: {act}, object: {name}"
            else:
                return f"body part: {pos}, action: {act}, object: {name}"
    # global hoi_desc_cache
    # if (len(hoi_desc_cache) == 0) and os.path.exists("hoi_descriptions.json"):
    #     with open("hoi_descriptions.json", "r") as f:
    #         hoi_desc_cache = json.load(f)
    # indexer = f"{name}||{','.join(sorted([f'<{relation[0]}, {relation[1]}>' for relation in relations]))}||{no_obj_name}"
    # if indexer in hoi_desc_cache:
    #     return hoi_desc_cache[indexer]
    # if no_obj_name:
    #     desc = ask_question(f"Please make a short description of a human-object interaction with object named '{name}'. Some pairs of contacting body parts and verbs of interaction are {', '.join(sorted([f'<{relation[0] if relation[0] != }, {relation[1]}>' for relation in relations]))} (might not accurate, please find the most appropriate expression). Provide your answer directly include necessary words of its name and relations unless it is meaningless word like 'unknown'. You need to express all of the infomation about the hoi and object in your description. Please use the present continuous tense and phrases with a verbal expression supplemented by modifying elements to describe, where the hidden subject is 'The person' (don't include 'the person' in your answer). Your final answer should be as short as possible, avoiding repetition, mention a body part only once, using only the word provided with minimal conjunctions to construct the shortest term. Do not give any extra imaginary description that is not informed above, your description should only based on provided facts. Do not provide any information that can directly infer that the object is \"{name}\" in your expression.")
    # else:
    #     desc = ask_question(f"Please make a short description of a human-object interaction with object named '{name}'. Some pairs of contacting body parts and verbs of interaction are {', '.join(sorted([f'<{relation[0]}, {relation[1]}>' for relation in relations]))} (might not accurate, please find the most appropriate expression). Provide your answer directly include necessary words of its name and relations unless it is meaningless word like 'unknown'. You need to express all of the infomation about the hoi and object in your description. Please use the present continuous tense and phrases with verb-object structures supplemented by modifying elements to describe, where the hidden subject is 'The person' (don't include 'the person' in your answer). Your final answer should be as short as possible, avoiding repetition, mention a body part only once, using only the word provided with minimal conjunctions to construct the shortest term. Do not give any extra imaginary description that is not informed above, your description should only based on provided facts.")
    # hoi_desc_cache[indexer] = desc
    # return desc

# def at_exit():
#     with open("clothing_descriptions.json", "w") as f:
#         json.dump(cloth_desc_cache, f, indent=4)
#     with open("hoi_descriptions.json", "w") as f:
#         json.dump(hoi_desc_cache, f, indent=4)

# import atexit
# atexit.register(at_exit)
# ================== ....... ==================

class QuestionGenerator:
    """......."""
    
    def __init__(self, dataset_pictures):
        self.dataset_pictures: List[Picture] = dataset_pictures
        self.picture_occurrence: Dict[Picture, int] = {}
    
    def filter_pictures(self):
        """....，........."""
        raise NotImplementedError("Subclasses must implement filter_pictures method")
    
    def generate_questions(self):
        """....，........."""
        raise NotImplementedError("Subclasses must implement generate_questions method")
    
    def save_questions(self, questions, filename):
        """......."""
        with open(filename, "w") as f:
            json.dump(questions, f, indent=4, default=set_default)
        print(f"Generated {len(questions)} questions and saved to {filename}")

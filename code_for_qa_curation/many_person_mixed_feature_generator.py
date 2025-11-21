import json
import os
from typing import Dict, List
import itertools
import concurrent.futures
import threading
from test_framework import Person, QuestionGenerator, POSITION_INCLUDE_MAP, POSITION_EXCLUDE_MAP, POSITION_SIMPLIFIER
from utils import ask_question, bounding_box_iou
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz
import random

CLOTHING_SYNONYMS = None
HOI_SYNONYMS = None


def load_synonym_dicts() -> Dict[str, Dict[str, List[str]]]:
    global CLOTHING_SYNONYMS, HOI_SYNONYMS
    if CLOTHING_SYNONYMS is None or HOI_SYNONYMS is None:
        with open("clothing_synonym_dict.json", "r", encoding="utf-8") as f:
            CLOTHING_SYNONYMS = json.load(f)["synonyms"]
        with open("hoi_synonym_dict.json", "r", encoding="utf-8") as f:
            HOI_SYNONYMS = json.load(f)["synonyms"]
    return {
        "clothing_synonyms": CLOTHING_SYNONYMS,
        "hoi_synonyms": HOI_SYNONYMS
    }

class ManyPersonMixedFeatureQuestionGenerator(QuestionGenerator):
    """............."""
    def __init__(self, dataset_pictures):
        super().__init__(dataset_pictures)
        
    def feat_to_distinct_str(self, feat):
        if feat["attr_type"] == "facial":
            return f"facial-{feat['attr_name']}-{feat['attr_value']}"
        elif feat["attr_type"] == "overall":
            return f"overall-{feat['attr_name']}-{feat['attr_value']}"
        elif feat["attr_type"] == "text":
            return f"text"
        elif feat["attr_type"] == "hoi":
            print(feat)
            return f"hoi-{list(feat['attr_value']['relation'])[0][0]}"
        elif feat["attr_type"] == "clothing":
            return f"clothing-{feat['attr_value']['type']}"
        else:
            return f"{feat['attr_type']}-{feat['attr_name']}"

    def generate_questions(self):
        results = []
        num = 0
        for picture in self.dataset_pictures:
            # .........
            unique_cond_feat_map = {}
            unique_ans_feat_map = {}
            unity_feat_list = None
            
            for i, person in enumerate(picture.persons):

                can_have_body_box = True
                for other in picture.persons:
                    if other.skeleton is None:
                        can_have_body_box = False
                    if other != person:
                        if other.body_box is not None:
                            iou = bounding_box_iou(person.body_box, other.body_box)
                            if iou > 0.1:
                                can_have_body_box = False
                                break
                        else:
                            can_have_body_box = False
                            break

                feat = person.full_feature_set(body_boxes=can_have_body_box)
                for f in feat:
                    f["person_no"] = i

                if unity_feat_list is None:
                    unity_feat_list = [f for f in feat if (f["attr_type"] != "bbox")]
                else:
                    unity_feat_list = self.feature_set_intersect(unity_feat_list, feat)

                for other in picture.persons:
                    if other != person:
                        feat = self.feature_set_substract(feat, other.full_feature_set())
                        
                unique_ans_feat_map[person], unique_cond_feat_map[person] = self.purify_features(feat)
                
                  
            unity_feat_list, _ = self.purify_features(unity_feat_list, exclude_facial=False)
            # ............：....grounding，......，......，..............（...grounding），..............（...grounding），....grounding
            for person in picture.persons:
                true_cond_feats = unique_cond_feat_map[person]
                ans_feats = unique_ans_feat_map[person]
                false_cond_feats = []
                for other in picture.persons:
                    if other != person:
                        false_cond_feats.extend(unique_cond_feat_map[other])
                
                # ...bbox.feats
                bbox_ans_feats = [feat for feat in ans_feats if feat["attr_type"] == "bbox"]

                # ..grounding..
                if bbox_ans_feats:
                    selected_ans = random.choice(bbox_ans_feats)
                    selected_cond = None
                    # ....selected_feat...unique_cond
                    different_cond_feats = [feat for feat in true_cond_feats if feat != selected_ans]
                    if different_cond_feats:
                        selected_cond = random.choice(different_cond_feats)
                    if selected_cond:
                        results.append({
                            "type": "identify-grounding",
                            "condition": selected_cond,
                            "question": selected_ans,
                            "image": picture.image_path(),
                            "distinct": [f"identify-grounding-{self.feat_to_distinct_str(selected_cond)}-{self.feat_to_distinct_str(selected_ans)}"],
                        })

                # ......
                # .......：
                # "attr_type":"facial", "attr_name": "pitch"
                # "attr_type":"facial", "attr_name": "yaw"
                # "attr_type":"overall", "attr_name": "gender"
                # "attr_type":"overall", "attr_name": "age"
                # "attr_type":"overall", "attr_name": "race"
                # "attr_type":"overall", "attr_name": "emotion"
                # "attr_type":"clothing"
                # "attr_type":"hoi"
                # ...............feat
                suitable_fill_mask_feats = []
                for feat in ans_feats:
                    if feat["attr_type"] in ["facial", "overall", "clothing", "hoi"]:
                        if feat["attr_name"] not in ["pitch", "yaw", "gender", "age", "race", "emotion", "clothing", "hoi"]:
                            continue
                        suitable_fill_mask_feats.append(feat)
                if suitable_fill_mask_feats:
                    selected_blank = random.choice(suitable_fill_mask_feats)
                    selected_cond = None
                    different_cond_feats = self.remove_same_place_features(true_cond_feats, [selected_blank]) # [feat for feat in true_cond_feats if feat != selected_blank]
                    if different_cond_feats:
                        selected_cond = random.choice(different_cond_feats)
                    if selected_cond and selected_blank:
                        results.append({
                            "type": "identify-blank",
                            "condition": selected_cond,
                            "question": selected_blank,
                            "image": picture.image_path(),
                            "can_mutate_hand_to_false": not person.hand_cant_swap(),
                            "distinct": [f"identify-blank-{self.feat_to_distinct_str(selected_cond)}-{self.feat_to_distinct_str(selected_blank)}"],
                        })

                # .....
                # ....true_cond_feats......，...true_cond_feats......，..false_cond_feats......
                try:
                    selected_cond = random.choice(true_cond_feats)
                    possible_ans = [feat for feat in true_cond_feats if (feat != selected_cond and (feat["attr_type"] != "bbox"))]
                    selected_ans = random.choice(possible_ans)
                    false_ans = random.sample(self.remove_same_place_features(false_cond_feats, [selected_cond]), 3)
                    results.append({
                        "type": "identify-choice",
                        "condition": selected_cond,
                        "image": picture.image_path(),
                        "true_answer": selected_ans,
                        "false_answers": false_ans,
                        "distinct": [f"identify-choice-{self.feat_to_distinct_str(selected_cond)}-{self.feat_to_distinct_str(selected_ans)}"],
                    })
                except Exception as e:
                    # print(e)
                    pass

                # ..............（...grounding）
                try:
                    cond_1 = random.choice([feat for feat in true_cond_feats if feat not in unity_feat_list and (feat["attr_type"] != "bbox")])
                    if len([feat for feat in unity_feat_list if (feat != cond_1 and (feat["attr_type"] != "bbox"))]) > 0:
                        cond_2 = random.choice([feat for feat in unity_feat_list if (feat != cond_1 and (feat["attr_type"] != "bbox"))])
                    else:
                        cond_2 = random.choice([feat for feat in true_cond_feats if (feat != cond_1 and (feat["attr_type"] != "bbox"))])
                    ans = random.choice(self.remove_same_place_features(bbox_ans_feats+suitable_fill_mask_feats, [cond_1, cond_2]))
                    results.append({
                        "type": "identify-tf_grounding" if ans in bbox_ans_feats else "identify-tf_blank",
                        "condition_1": cond_1,
                        "condition_2": cond_2,
                        "answer": ans,
                        "image": picture.image_path(),
                        "can_mutate_hand_to_false": not person.hand_cant_swap(),
                        "confuse_cond_from": "unity" if cond_2 in unity_feat_list else "self",
                        "distinct": [f"identify-t_{'grounding' if ans in bbox_ans_feats else 'blank'}-{self.feat_to_distinct_str(cond_1)}-{self.feat_to_distinct_str(ans)}"],
                    })
                except Exception as e:
                    # print(e)
                    pass

                # ..............（...grounding）
                try:
                    cond_1 = random.choice([feat for feat in true_cond_feats if feat not in unity_feat_list and (feat["attr_type"] != "bbox" or feat["attr_name"] in ["face", "body"])])
                    cond_2 = random.choice([feat for feat in self.remove_same_place_features(false_cond_feats, [cond_1]) if feat != cond_1 and (feat["attr_type"] != "bbox" or feat["attr_name"] in ["face", "body"])])
                    ans = random.choice(self.remove_same_place_features(bbox_ans_feats+suitable_fill_mask_feats, [cond_1, cond_2]))
                    results.append({
                        "type": "identify-tf_grounding" if ans in bbox_ans_feats else "identify-tf_blank",
                        "condition_1": cond_1,
                        "condition_2": cond_2,
                        "fake_answer": ans, # ...............
                        "image": picture.image_path(),
                        "distinct": [f"identify-f_{'grounding' if ans in bbox_ans_feats else 'blank'}-{self.feat_to_distinct_str(cond_1)}-{self.feat_to_distinct_str(ans)}"]
                    })
                except Exception as e:
                    # print(e)
                    pass

                # ..HOI...grounding
                # ...true_cond_feats..HOI
                try:
                    if any(feat for feat in true_cond_feats if feat["attr_type"] == "hoi"):
                        # .....HOI....
                        ans = random.choice([feat for feat in true_cond_feats if feat["attr_type"] == "hoi"])
                        # ......HOI.true_cond_feats....
                        cond = random.choice([feat for feat in true_cond_feats if feat != ans and (feat["attr_type"] != "bbox" or feat["attr_name"] in ["face", "body"])])
                        
                        results.append({
                            "type": "identify-open_grounding",
                            "condition": cond,
                            "answer": ans,
                            "image": picture.image_path(),
                            "can_mutate_hand_to_false": not person.has_multi_hand_hoi(),
                            "distinct": [f"identify-open_grounding-{self.feat_to_distinct_str(cond)}-{self.feat_to_distinct_str(ans)}"]
                        })
                except Exception as e:
                    pass

            # .......，..........（...）
            try:
                answer = random.choice(unity_feat_list)

                false_ans = random.sample([feat for feat in itertools.chain(*unique_cond_feat_map.values()) if feat != answer and feat["attr_type"]!="bbox"], k=3)

                results.append({
                    "type": "common_choice",
                    "true_answer": answer,
                    "false_answers": false_ans,
                    "image": picture.image_path(),
                    "distinct": [f"common_choice-{self.feat_to_distinct_str(answer)}"]
                })
            except Exception as e:
                pass

        return results

    def filter_pictures(self):
        """........."""
        filtered_pictures = []
        for picture in self.dataset_pictures:
            # ........，.........
            if len(picture.persons) > 1 and all(person.body_box is not None for person in picture.persons):
                filtered_pictures.append(picture)
        print(f"Filtered down to {len(filtered_pictures)} records for multi-person cross feature questions.")
        self.dataset_pictures = filtered_pictures
        self._construct_synonym_dict()
        return filtered_pictures
    
    def _construct_synonym_dict(self):
        """......."""
        load_synonym_dicts()
        self.clothing_synonyms = CLOTHING_SYNONYMS
        self.hoi_synonyms = HOI_SYNONYMS

    def feature_set_substract(self, a, b):
        # ..............
        c = []
        for feat_a in a:
            # ..feat_b.....attr_type.attr_name..
            sub_b = []
            for feat_b in b:
                if feat_a["attr_type"] == feat_b["attr_type"] and feat_a["attr_name"] == feat_b["attr_name"]:
                    sub_b.append(feat_b)
            # .........
            if feat_a["attr_type"] in ["facial", "overall"]:
                # .......，....
                assert len(sub_b) <= 1
                if len(sub_b) == 1:
                    if feat_a["attr_value"] != sub_b[0]["attr_value"] and sub_b[0]["attr_value"] is not None:
                        c.append(feat_a)
                else:
                    c.append(feat_a)
            # ...bounding box.iou..0.5....
            if feat_a["attr_type"] == "bbox":
                assert len(sub_b) <= 1
                if len(sub_b) == 1:
                    iou = bounding_box_iou(feat_a["attr_value"], sub_b[0]["attr_value"])
                    if iou < 0.5:
                        c.append(feat_a)
                else:
                    c.append(feat_a)
            # clothing..b.........................
            if feat_a["attr_type"] == "clothing":
                found = False
                for feat_b in sub_b:
                    type_match = feat_a["attr_value"]["name"] in self.clothing_synonyms[feat_b["attr_value"]["name"]] or feat_a["attr_value"]["name"] == feat_b["attr_value"]["name"]
                    color_match = False
                    for a_color in feat_a["attr_value"]["color"]:
                        for b_color in feat_b["attr_value"]["color"]:
                            if a_color in self.clothing_synonyms[b_color] or a_color == b_color:
                                color_match = True
                                break
                        if color_match:
                            break
                    if type_match and color_match:
                        found = True
                        break
                if not found:
                    c.append(feat_a)
            # hoi..b................a......exclude....obj.....
            if feat_a["attr_type"] == "hoi":
                found = False
                for feat_b in sub_b:
                    action_position_match = False
                    for a_position, a_action in feat_a["attr_value"]["relation"]:
                        for b_position, b_action in feat_b["attr_value"]["relation"]:
                            action_synonym = a_action in (self.hoi_synonyms[b_action] + [b_action])
                            position_exclude = a_position in (POSITION_EXCLUDE_MAP.get(b_position, []) + [b_position])
                            if action_synonym and position_exclude:
                                action_position_match = True
                                break
                        if action_position_match:
                            break
                    name_match = feat_a["attr_value"]["object"] in self.hoi_synonyms.get(feat_b["attr_value"]["object"], []) or feat_b["attr_value"]["object"] == feat_a["attr_value"]["object"]
                    if action_position_match and name_match:
                        found = True
                        break
                if not found:
                     c.append(feat_a)
            # .........0.8
            if feat_a["attr_type"] == "text":
                found = False
                for feat_b in sub_b:
                    if fuzz.token_sort_ratio(feat_a["attr_value"], feat_b["attr_value"]) > 80:
                        found = True
                if not found:
                    c.append(feat_a)
        return c
    
    def feature_set_intersect(self, a, b):
        # ........
        c = []
        for feat_a in a:
            # ..feat_b.....attr_type.attr_name..
            sub_b = []
            for feat_b in b:
                if feat_a["attr_type"] == feat_b["attr_type"] and feat_a["attr_name"] == feat_b["attr_name"]:
                    sub_b.append(feat_b)
            # .........
            if feat_a["attr_type"] in ["facial", "overall"]:
                # .......，....
                assert len(sub_b) <= 1
                if len(sub_b) == 1:
                    if feat_a["attr_value"] == sub_b[0]["attr_value"]:
                        c.append(feat_a)
            # ..bounding box.........
            
            # clothing..b.............................
            if feat_a["attr_type"] == "clothing":
                found = False
                for feat_b in sub_b:
                    type_match = feat_a["attr_value"]["name"] in self.clothing_synonyms[feat_b["attr_value"]["name"]] or feat_a["attr_value"]["name"] == feat_b["attr_value"]["name"]
                    color_match = False
                    a_color_match = False
                    b_color_match = False
                    for a_color in feat_a["attr_value"]["color"]:
                        a_color_match = False
                        for b_color in feat_b["attr_value"]["color"]:
                            if a_color in self.clothing_synonyms[b_color] or a_color == b_color:
                                a_color_match = True
                                break
                        if not a_color_match:
                            break
                    for b_color in feat_b["attr_value"]["color"]:
                        b_color_match = False
                        for a_color in feat_a["attr_value"]["color"]:
                            if b_color in self.clothing_synonyms[a_color] or b_color == a_color:
                                b_color_match = True
                                break
                        if not b_color_match:
                            break
                    color_match = (a_color_match and b_color_match)
                    if type_match and color_match:
                        found = True
                        break
                if found:
                    c.append(feat_a)
            # hoi..b................a......include....obj.....，bbox......，bbox....，....，..bbox
            if feat_a["attr_type"] == "hoi":
                for feat_b in sub_b:
                    action_match = False
                    position_match = False
                    for a_position, a_action in feat_a["attr_value"]["relation"]:
                        for b_position, b_action in feat_b["attr_value"]["relation"]:
                            action_synonym = a_action in self.hoi_synonyms[b_action]
                            position_include = a_position in (POSITION_INCLUDE_MAP.get(b_position, []) + [b_position])
                            if action_synonym:
                                action_match = True
                            if position_include:
                                position_match = True
                    name_match = feat_a["attr_value"]["object"] in self.hoi_synonyms.get(feat_b["attr_value"]["object"], []) or feat_b["attr_value"]["object"] == feat_a["attr_value"]["object"]
                    if action_match and position_match and name_match:
                        if bounding_box_iou(feat_a["attr_value"]["bbox"], feat_b["attr_value"]["bbox"]) > 0.99:
                            c.append(feat_a)
                        else:
                            feat_a_copy = feat_a.copy()
                            feat_a_copy["attr_value"]["bbox"] = None
                            c.append(feat_a_copy)
                            break
        return c
    def person_ignore_face(self, person:Person):
        return person.detailing_property("face_seen", True)
    def purify_features(self, features, exclude_facial=False):
        """........"""
        whole = [feat for feat in features if feat["attr_value"] is not None]
        if exclude_facial:
            whole = [feat for feat in whole if (feat["attr_type"] != "facial")]
            whole = [feat for feat in whole if (feat["attr_type"] != "bbox" or (feat["attr_type"] == "bbox" and feat["attr_value"] in ["body", "face"]))]
        # bbox.........input
        can_input = [feat for feat in whole if not (feat["attr_type"] == "bbox" and feat["attr_name"] not in ["face", "body"])]
        return whole, can_input
    def remove_same_place_features(self, features, provided):
        """............"""
        seen_positions = set()
        seen_bbox = set()
        overall_attr = set()
        for f in provided:
            if f["attr_type"] == "clothing":
                seen_positions.add(f["attr_value"]["type"])
            if f["attr_type"] == "hoi":
                for pos, act in f["attr_value"]["relation"]:
                    seen_positions.add(pos)
            if f["attr_type"] == "bbox":
                seen_bbox.add(f["attr_name"])
            if f["attr_type"] == "overall":
                overall_attr.add(f["attr_name"])
        r = []
        for f in features:
            if f["attr_type"] == "clothing":
                if f["attr_value"]["type"] in seen_positions:
                    continue
            if f["attr_type"] == "hoi":
                valid = True
                for pos, act in f["attr_value"]["relation"]:
                    if pos in seen_positions:
                        valid = False
                if not valid:
                    continue
            if f["attr_type"] == "bbox":
                if f["attr_name"] in seen_bbox:
                    continue
            if f in provided:
                continue
            if f["attr_type"] == "overall":
                if f["attr_name"] in overall_attr:
                    continue
            r.append(f)
        return r
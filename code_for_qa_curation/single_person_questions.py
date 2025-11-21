import json
import os
from typing import Dict, List
import itertools
import concurrent.futures
import threading
from test_framework import Person, QuestionGenerator, POSITION_INCLUDE_MAP, POSITION_EXCLUDE_MAP, POSITION_SIMPLIFIER, ALL_POSITIONS, FACE_ATTR_DESCRIPTIONS, get_cloth_description
from utils import ask_question, bounding_box_iou
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz
import random
from copy import deepcopy

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

class SinglePersonQuestions(QuestionGenerator):
    """..........."""
    def __init__(self, dataset_pictures):
        super().__init__(dataset_pictures)
        

    

    def generate_questions(self):
        results = []
        
        for picture in self.dataset_pictures:
            person = picture.persons[0]  # .......
            feats = person.full_feature_set(body_boxes=True)
            facial_admit_set = person.get_face_attr_admit_list()
            facial_deny_set = person.get_face_attr_deny_list()
            clothes = person.get_clothing_list(only_confident=True)

            # ..（..）...

            ## ....
            try:
                # count = random.choice([1,2,3][:min(len(facial_admit_set), 3)])
                count = 1
                true_answers = random.sample(facial_admit_set, count)
                false_answers = random.sample(facial_deny_set, 4 - count)
                results.append({
                    "type": "face_choice",
                    "true_answers": true_answers,
                    "false_answers": false_answers,
                    "image": picture.image_path(),
                    "distinct": [f"face_choice-{true_answers[0]}", f"face_choice-false-{false_answers[0]}", f"face_choice-false-{false_answers[1]}", f"face_choice-false-{false_answers[2]}"],
                })
            except Exception as e:
                pass

            ## ....
            if len(clothes) > 0:
                count = random.choice([1,2,3][:min(len(clothes), 3)])
                true_answers = random.sample(clothes, count)
                false_answers = []
                for true_cloth in true_answers:
                    cloth_type = true_cloth['type']
                    same_type_names = self.type_clothes_dict.get(cloth_type, set())
                    syn_names = set(self.clothing_synonyms.get(true_cloth['name'], []))
                    if len(list(same_type_names - ({true_cloth['name']} | syn_names))) > 0:
                        fake_name = random.choice(list(same_type_names - ({true_cloth['name']} | syn_names)))
                    else:
                        fake_name = None
                    
                    if fake_name is not None:
                        false_cloth = deepcopy(true_cloth)
                        false_cloth['name'] = fake_name
                        false_answers.append(false_cloth)

                    cloth_colors = set(true_cloth.get('color', []))
                    
                    for color in deepcopy(cloth_colors):
                        cloth_colors.update(self.clothing_synonyms.get(color, []))
                    false_cloth = deepcopy(true_cloth)
                    false_cloth['color'] = [random.choice(list(self.all_colors - cloth_colors))]
                    false_answers.append(false_cloth)

                    if count == 1 and fake_name is not None:
                        false_cloth = deepcopy(true_cloth)
                        false_cloth['color'] = [random.choice(list(self.all_colors - cloth_colors))]
                        false_cloth['name'] = fake_name
                        false_answers.append(false_cloth)
                if len(false_answers) >= 3:
                    true_answers = random.sample(true_answers, 1)
                    false_answers = random.sample(false_answers, 3)
                    distinct = [f"cloth_choice-{true_answers[0]['type']}", f"cloth_choice-false-{false_answers[0]['type']}", f"cloth_choice-false-{false_answers[1]['type']}", f"cloth_choice-false-{false_answers[2]['type']}",]
                    
                    results.append({
                        "type": "cloth_choice",
                        "true_answers": true_answers,
                        "false_answers": false_answers,
                        "image": picture.image_path(),
                        "distinct": list(set(distinct)),
                    })
        

            ## HOI..
            if len(person.hois) > 0:
                # count = random.choice([1,2,3][:min(len(person.hois), 3)])
                count = 1
                true_answers = [{"relation": hoi.get_position_action_pairs(), "object": hoi.get_object_name(), "bbox": hoi.get_object_box()} for hoi in random.sample(person.hois, count)]
                false_answers = []
                for true_hoi in true_answers:
                    all_pos = set()
                    for hoi in person.hois:
                        if hoi.get_object_name() not in (self.hoi_synonyms[true_hoi["object"]] + [true_hoi["object"]]):
                            continue
                        for pos, act in hoi.get_position_action_pairs():
                            all_pos.add(pos)
                    false_pos_set = set(ALL_POSITIONS) - all_pos
                    for pos in all_pos:
                        false_pos_set = false_pos_set - set(POSITION_EXCLUDE_MAP.get(pos, []))
                    if "left hand" in false_pos_set and not person.hand_cant_swap():
                        false_pos = "left hand"
                    elif "right hand" in false_pos_set and not person.hand_cant_swap():
                        false_pos = "right hand"
                    else:
                        false_pos = random.choice(list(false_pos_set))
                    false_answer = deepcopy(true_hoi)
                    appeared_act = set([i[1] for i in true_hoi["relation"]])
                    false_pos_act =  [(false_pos,i) for i in appeared_act]
                    false_answers.append({
                        "relation": false_pos_act,
                        "object": false_answer["object"]
                    })

                    possible_objs = set()
                    for pos in all_pos:
                        possible_objs.update(self.position_object_dict.get(pos, []))
                    all_objs = set()
                    for hoi in person.hois:
                        all_objs.update(self.hoi_synonyms.get(hoi.get_object_name(), []))
                    if len(possible_objs - all_objs) > 0:
                        false_obj = random.choice(list(possible_objs - all_objs))
                    else:
                        false_obj = None
                    if false_obj is not None:
                        false_answers.append({
                            "relation": false_answer["relation"],
                            "object": false_obj
                        })

                        if len(true_answers) == 1:
                            false_answers.append(
                                {
                                    "relation": false_pos_act,
                                    "object": false_obj
                                }
                            )
                distinct = []
                for ta in true_answers:
                    for pos, act in ta["relation"]:
                        distinct.append(f"hoi_choice-{pos}")
                for fa in false_answers:
                    for pos, act in fa["relation"]:
                        distinct.append(f"hoi_choice-false-{pos}")
                results.append({
                    "type": "hoi_choice",
                    "true_answers": true_answers,
                    "false_answers": false_answers,
                    "image": picture.image_path(),
                    "distinct": list(set(distinct))
                })

            # bounding box..
            ## ..bounding box
            try:
                face_boxes = [f for f in feats if f["attr_type"] == "bbox" and f["attr_name"] in ["nose", "mouth", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "face"]]
                if face_boxes:
                    face_box = random.choice(face_boxes)
                    results.append({
                        "type": "face_grounding",
                        "question": face_box,
                        "image": picture.image_path(),
                        "distinct": [f"face_grounding-{face_box['attr_name']}"]
                    })
            except Exception as e:
                pass
            ## ..bounding box
            try:
                body_boxes = [f for f in feats if f["attr_type"] == "bbox" and f["attr_name"] in ["body", "left_hand", "right_hand", "left_foot", "right_foot"]]
                
                if body_boxes:
                    body_box = random.choice(body_boxes)
                    results.append({
                        "type": "body_grounding",
                        "question": body_box,
                        "image": picture.image_path(),
                        "distinct": [f"body_grounding-{body_box['attr_name']}"]
                    })
            except Exception as e:
                pass

            ## HOI bounding box
            try:
                hoi_boxes = [f for f in feats if f["attr_type"] == "hoi"]
                if hoi_boxes:
                    hoi_box = random.choice(hoi_boxes)
                    distinct = []
                    for pos, act in hoi_box['attr_value']["relation"]:
                        distinct.append(f"hoi_grounding-{pos}")
                    results.append({
                        "type": "hoi_grounding",
                        "question": hoi_box,
                        "image": picture.image_path(),
                        "distinct": distinct,
                    })
            except Exception as e:
                pass

            # ......
            ## .....
            try:
                clothing_item = random.choice([f for f in feats if f["attr_type"] == "clothing"])
                results.append({
                    "type": "open_clothing",
                    "question": clothing_item,
                    "image": picture.image_path(),
                    "distinct": [f"open_clothing-{clothing_item['attr_value']['type']}"]
                })
            except Exception as e:
                pass
            ## ..hoi.
            if len(person.hois) > 0:
                hoi_item = random.choice([f for f in feats if f["attr_type"] == "hoi"])
                distinct = []
                for pos, act in hoi_item['attr_value']["relation"]:
                    distinct.append(f"open_hoi-{pos}")
                results.append({
                    "type": "open_hoi",
                    "question": hoi_item,
                    "image": picture.image_path(),
                    "distinct": list(set(distinct)),
                    "can_mutate_hand_to_false": not person.hand_cant_swap(),
                })
            

        return results

    def filter_pictures(self):
        """........."""
        filtered_pictures = []
        for picture in self.dataset_pictures:
            # .......
            if len(picture.persons) == 1:
                filtered_pictures.append(picture)
        print(f"Filtered down to {len(filtered_pictures)} records for single-person questions.")
        self.dataset_pictures = filtered_pictures
        self._construct_synonym_dict()
        return filtered_pictures
    
    def _construct_synonym_dict(self):
        """......."""
        load_synonym_dicts()
        self.clothing_synonyms = CLOTHING_SYNONYMS
        self.hoi_synonyms = HOI_SYNONYMS
        type_clothes_dict = {}
        position_object_dict = {}
        all_colors = set()
        for image in self.dataset_pictures:
            for person in image.persons:
                for clothing in person.get_clothing_list():
                    ctype = clothing['type']
                    cname = clothing['name']
                    if ctype not in type_clothes_dict:
                        type_clothes_dict[ctype] = set()
                    type_clothes_dict[ctype].add(cname)
                    for color in clothing.get('color', []):
                        all_colors.add(color)
            for hoi in person.hois:
                for pos, act in hoi.get_position_action_pairs():
                    obj = hoi.get_object_name()
                    if pos not in position_object_dict:
                        position_object_dict[pos] = set()
                    position_object_dict[pos].add(obj)
        self.type_clothes_dict = type_clothes_dict
        self.position_object_dict = position_object_dict
        self.all_colors = all_colors

    def generate_qa(self, question_list):
        """....."""
        qa_list = []
        for q in question_list:
            question_msgs = []
            answer_msg = {}
            question_msgs.append(
                {
                    "type": "image",
                    "image": q["image"]
                }
            )
            answers = []
            if q["type"] == "face_choice":
                for true_answer in q["true_answers"]:
                    answers.append((True,  FACE_ATTR_DESCRIPTIONS[true_answer][0]))
                for false_answer in q["false_answers"]:
                    answers.append((False, FACE_ATTR_DESCRIPTIONS[false_answer][0]))
                random.shuffle(answers)
                true_selection = []
                selection_strs = []
                selection_str = ""
                for serial, (is_true, text) in zip(["A", "B", "C", "D"], answers):
                    if is_true:
                        true_selection.append(serial)
                    selection_strs.append(f"{serial}. {text}")
                selection_str = "\n".join(selection_strs)
                question_msgs.append(
                    {
                        "type": "text",
                        "text": f"Please select the facial features of the person in the image from the following options (multiple selections are allowed):\n\n{selection_str}\n\nPlease provide the option letters of all correct choices, separated by commas if multiple."
                    }
                )
                answer_msg = {
                    "type": "multiple_choice",
                    "data": true_selection
                }
                qa_list.append({
                    "type": "face_choice",
                    "question": question_msgs,
                    "answer": answer_msg,
                    "data": q
                })
            elif q["type"] == "cloth_choice":
                for true_answer in q["true_answers"]:
                    answers.append((True,  get_cloth_description(true_answer["name"], true_answer["color"])))
                for false_answer in q["false_answers"]:
                    answers.append((False, get_cloth_description(false_answer["name"], false_answer["color"])))
                random.shuffle(answers)
                true_selection = []
                selection_strs = []
                selection_str = ""
                for serial, (is_true, text) in zip(["A", "B", "C", "D"], answers):
                    if is_true:
                        true_selection.append(serial)
                    selection_strs.append(f"{serial}. {text}")
                selection_str = "\n".join(selection_strs)
                question_msgs.append(
                    {
                        "type": "text",
                        "text": f"Please select the clothing items of the person in the image from the following options (multiple selections are allowed):\n\n{selection_str}\n\nPlease provide the option letters of all correct choices, separated by commas if multiple."
                    }
                )
                answer_msg = {
                    "type": "multiple_choice",
                    "data": true_selection
                }
                qa_list.append({
                    "type": "cloth_choice",
                    "question": question_msgs,
                    "answer": answer_msg,
                    "data": q
                })
            else:
                print(f"Unsupported question type: {q['type']}")
        return qa_list
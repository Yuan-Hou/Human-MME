from dotenv import load_dotenv
from abstract_single_image_choice import AbstractSingleImageChoiceQuestionGenerator
from multi_hoi_generator import MultiImageHoiFeatureQuestionGenerator
from test_framework import FACE_ATTR_DESCRIPTIONS, get_cloth_description, get_full_data, Picture, get_hoi_description
from multi_face_feature_generator import MultiFaceFeatureQuestionGenerator
from multi_clothing_feature_generator import MultiPersonClothingFeatureQuestionGenerator
from many_person_mixed_feature_generator import ManyPersonMixedFeatureQuestionGenerator
from single_person_questions import SinglePersonQuestions
import json
import os
import shutil
import random
from PIL import Image
import asyncio
import threading
from copy import deepcopy

load_dotenv()

COLORS = ["red", "green", "blue", "yellow", "white"]

def history_to_str(history):
    full_str = ""
    images = []
    for h in history:
        if h["type"] == "image":
            images.append(h["image"])
            full_str += f"[{h['image']}]"
        elif h["type"] == "multi_image":
            images.append(h["image_A"])
            images.append(h["image_B"])
            images.append(h["image_C"])
            images.append(h["image_D"])
            full_str += f"[A:{h['image_A']}] [B:{h['image_B']}] [C:{h['image_C']}] [D:{h['image_D']}]\n"
        elif h["type"] == "image_bbox":
            images.append(h["image"])
            full_str += f"[{h['image']} with bboxes {h.get('bboxes', [])}]"
        else:
            full_str += f"{h['text']}\n"
    return full_str, images

def process_cond(cond, img_msg):
    if cond["attr_type"] == "facial":
        if cond["attr_name"] == "pitch":
            return f"Looking {cond['attr_value']}ward", (None, None)
        elif cond["attr_name"] == "yaw":
            if cond["attr_value"] == "left":
                return "Face turned to right side of image", (None, None)
            else:
                return "Face turned to left side of image", (None, None)
        else:
            return FACE_ATTR_DESCRIPTIONS[cond["attr_name"]][0 if cond["attr_value"] else 1], (None, None)
    elif cond["attr_type"] == "overall":
        if cond["attr_name"] == "text":
            return f"Have text \"{cond['attr_value']}\" on body, wearings or related objects", (None, None)
        else:
            return f"{cond['attr_name']}: {cond['attr_value']}", (None, None)
    elif cond["attr_type"] == "clothing":
        return f"Wearing {get_cloth_description(cond['attr_value']['name'], cond['attr_value']['color'])}", (None, None)
    elif cond["attr_type"] == "hoi":
        return f"Have interaction with object: \"{get_hoi_description(cond['attr_value']['object'], cond['attr_value']['relation'])}\"", (None, None)
    elif cond["attr_type"] == "bbox":
        new_img = deepcopy(img_msg)
        new_img["type"] = "image_bbox"
        new_img["bboxes"] = new_img.get("bboxes", [])
        color = COLORS[len(new_img["bboxes"])]
        new_img["bboxes"].append((color, cond["attr_value"]))
        return f"{cond['attr_name'].replace('_', ' ').replace('body', 'whole visible body')} in the bounding box [{int(cond['attr_value'][0]*img_msg['width'])},{int(cond['attr_value'][1]*img_msg['height'])},{int(cond['attr_value'][2]*img_msg['width'])},{int(cond['attr_value'][3]*img_msg['height'])}] (xyxy format)", (f"{cond['attr_name'].replace('_', ' ').replace('body', 'whole visible body')} in the {color} bounding box", new_img)
    else:
        raise RuntimeError(f"Cant handle attr_type {cond['attr_type']}")

def ask_blank_about_condition(cond):
    if cond["attr_name"] == "pitch":
        return f"The face is looking up or down?", cond["attr_value"]
    if cond["attr_name"] == "yaw":
        return f"The face is turned to left or right side of image?", "right side" if cond["attr_value"] == "left" else "left side"
    if cond["attr_name"] == "hoi": 
        return f"What is the object that have interation can be described as \"{get_hoi_description(cond['attr_value']['object'], cond['attr_value']['relation'], no_obj_name=True)}\" with the person?", cond["attr_value"]["object"]
    if cond["attr_name"] == "clothing":
        return f"What is name and color of the clothing item of type \"{cond['attr_value']['type']}\" that the person is wearing?", get_cloth_description(cond['attr_value']['name'], cond['attr_value']['color'])
    if cond["attr_type"] == "overall":
        return f"What is the {cond['attr_name']} of the person?", cond["attr_value"]
    raise RuntimeError(f"Cant handle attr_name {cond['attr_name']} for blank question")
    
        

def multi_mixed_gen_qa(q, q_msgs):
    results = []
    if q["type"] == "identify-grounding":
        cond_str, (bbox_cond_str, bbox_img_msg) = process_cond(q["condition"], q_msgs[0])
        q_msgs.append(
            {
                "type": "text",
                "text": f"Resolution of the image provided is {q_msgs[0]['width']}x{q_msgs[0]['height']}. There is one person in the image that meets the following condition:\n\n- {cond_str}\n\nPlease provide the bounding box of the person's {q['question']['attr_name'].replace('_', ' ').replace('body', 'whole visible body')} in xyxy format."
            }
        )
        answer = {
            "type": "bbox",
            "data":q['question']["attr_value"]
        }
        results.append({
            "type": "identify-grounding",
            "question": q_msgs,
            "answer": answer,
            "data": q
        })
        if bbox_cond_str is not None and bbox_img_msg is not None:
            new_q_msgs = deepcopy(q_msgs)
            new_q_msgs[0] = bbox_img_msg
            new_q_msgs[1] = {
                "type": "text",
                "text": f"Resolution of the image provided is {bbox_img_msg['width']}x{bbox_img_msg['height']}. There is one person in the image that meets the following condition:\n\n- {bbox_cond_str}\n\nPlease provide the bounding box of the person's {q['question']['attr_name'].replace('_', ' ').replace('body', 'whole visible body')} in xyxy format."
            }
        
            results.append({
                "type": "identify-grounding",
                "question": new_q_msgs,
                "answer": answer,
                "data": q
            })
    elif q["type"] == "identify-blank":
        cond_str, (bbox_cond_str, bbox_img_msg) = process_cond(q["condition"], q_msgs[0])
        q_str, a = ask_blank_about_condition(q["question"])
        q_msgs.append(
            {
                "type": "text",
                "text": f"Resolution of the image provided is {q_msgs[0]['width']}x{q_msgs[0]['height']}. There is one person in the image that meets the following condition:\n\n- {cond_str}\n\n{q_str}"
            }
        )
        answer = {
            "type": "blank",
            "data": a
        }
        results.append({
            "type": "identify-blank",
            "question": q_msgs,
            "answer": answer,
            "data": q
        })
        if bbox_cond_str is not None and bbox_img_msg is not None:
            new_q_msgs = deepcopy(q_msgs)
            new_q_msgs[0] = bbox_img_msg
            new_q_msgs[1] = {
                "type": "text",
                "text": f"Resolution of the image provided is {bbox_img_msg['width']}x{bbox_img_msg['height']}. There is one person in the image that meets the following condition:\n\n- {bbox_cond_str}\n\nPlease provide the missing information of the person's {q['question']['attr_name'].replace('_', ' ').replace('body', 'whole visible body')}."
            }
        
            results.append({
                "type": "identify-blank",
                "question": new_q_msgs,
                "answer": answer,
                "data": q
            })
    elif q["type"] == "identify-choice":
        cond_str, (bbox_cond_str, bbox_img_msg) = process_cond(q["condition"], q_msgs[0])
        choices = []
        bbox_choices = []
        bbox_used = bbox_cond_str is not None
        for choice_cond in [q["true_answer"]] + q["false_answers"]:
            choice_str, (bbox_choice_str, bbox_choice_img_msg) = process_cond(choice_cond, q_msgs[0] if bbox_img_msg is None else bbox_img_msg)
            choices.append((choice_cond == q["true_answer"], choice_str))
            if bbox_choice_str is not None:
                bbox_choices.append((choice_cond == q["true_answer"], bbox_choice_str))
                bbox_used = True
                bbox_img_msg = bbox_choice_img_msg
            else:
                bbox_choices.append((choice_cond == q["true_answer"], choice_str))

        random.shuffle(choices)
        true_selection = []
        selections = []
        for seq, (tf, selection) in zip(["A", "B", "C", "D"], choices):
            if tf:
                true_selection.append(seq)
            selections.append(f"{seq}. {selection}")
        selection_str = "\n".join(selections)
        q_msgs.append(
            {
                "type": "text",
                "text": f"Resolution of the image provided is {q_msgs[0]['width']}x{q_msgs[0]['height']}. There is one person in the image that meets the following condition:\n\n- {cond_str}\n\nIgnoring other persons, please select the option that best describes the referred person.\n\n{selection_str}\n\nPlease provide the option letter of the most possible answer."
            }
        )
        answer = {
            "type": "choice",
            "data":true_selection
        }
        results.append({
            "type": "identify-choice",
            "question": q_msgs,
            "answer": answer,
            "data": q
        })
        if bbox_used:
            random.shuffle(bbox_choices)
            true_selection = []
            selections = []
            for seq, (tf, cond_str) in zip(["A", "B", "C", "D"], bbox_choices):
                if tf:
                    true_selection.append(seq)
                selections.append(f"{seq}. {cond_str}")
            selection_str = "\n".join(selections)
            new_q_msgs = deepcopy(q_msgs)
            new_q_msgs[0] = bbox_img_msg
            new_q_msgs[1] = {
                "type": "text",
                "text": f"Resolution of the image provided is {bbox_img_msg['width']}x{bbox_img_msg['height']}. There is one person in the image that meets the following condition:\n\n- {bbox_cond_str if bbox_cond_str is not None else cond_str}\n\nIgnoring other persons, please select the option that best describes the referred person.\n\n{selection_str}\n\nPlease provide the option letter of the most possible answer."
            }
            answer = {
                "type": "choice",
                "data":true_selection
            }
            results.append({
                "type": "identify-choice",
                "question": new_q_msgs,
                "answer": answer,
                "data": q
            })
        
    elif q["type"] == "identify-tf_grounding":
        cond_1_str, (bbox_cond_1_str, bbox_img_msg) = process_cond(q["condition_1"], q_msgs[0])
        cond_2_str, (bbox_cond_2_str, bbox_img_msg_2) = process_cond(q["condition_2"], q_msgs[0] if bbox_img_msg is None else bbox_img_msg)
        if bbox_img_msg_2 is not None:
            bbox_img_msg = bbox_img_msg_2
        q_msgs.append(
            {
                "type": "text",
                "text": f"Resolution of the image provided is {q_msgs[0]['width']}x{q_msgs[0]['height']}. There might be one person in the image that meets the following two conditions:\n\n- {cond_1_str}\n- {cond_2_str}\n\nPlease provide the bounding box of the person's {(q['answer']['attr_name'] if 'answer' in q else q['fake_answer']['attr_name']).replace('_', ' ').replace('body', 'whole visible body')} in xyxy format if there is such a person. Or else, please provide [-1,-1,-1,-1] as answer."
            }
        )
        answer = {
            "type": "tf_bbox",
            "data":q['answer']["attr_value"] if "answer" in q else [-1,-1,-1,-1]
        }
        results.append({
            "type": "identify-tf_grounding",
            "question": q_msgs,
            "answer": answer,
            "data": q
        })
        if (bbox_cond_1_str is not None or bbox_cond_2_str is not None) and bbox_img_msg is not None:
            new_q_msgs = deepcopy(q_msgs)
            new_q_msgs[0] = bbox_img_msg
            new_q_msgs[1] = {
                "type": "text",
                "text": f"Resolution of the image provided is {bbox_img_msg['width']}x{bbox_img_msg['height']}. There might be one person in the image that meets the following two conditions:\n\n- {bbox_cond_1_str if bbox_cond_1_str is not None else cond_1_str}\n- {bbox_cond_2_str if bbox_cond_2_str is not None else cond_2_str}\n\nPlease provide the bounding box of the person's {(q['answer']['attr_name'] if 'answer' in q else q['fake_answer']['attr_name']).replace('_', ' ').replace('body', 'whole visible body')} in xyxy format if there is such a person. Or else, please provide [-1,-1,-1,-1] as answer."
            }
        
            results.append({
                "type": "identify-tf_grounding",
                "question": new_q_msgs,
                "answer": answer,
                "data": q
            })
        if "fake_answer" not in q and q["can_mutate_hand_to_false"] and q["condition_1"]["attr_type"] == "hoi" and len(set([r[0] for r in q["condition_1"]["attr_value"]["relation"]]) & set(["left hand", "right hand"])) > 0:
            new_q = deepcopy(q)
            new_q["condition_1"]["attr_value"]["relation"] = [("left hand" if r[0] == "right hand" else "right hand", r[1]) for r in new_q["condition_1"]["attr_value"]["relation"] if r[0] in ["left hand", "right hand"]]
            new_q["fake_answer"] = new_q["answer"]
            new_q.pop("answer")
            new_q["can_mutate_hand_to_false"] = "MUTATED"
            results += multi_mixed_gen_qa(new_q, deepcopy(q_msgs[:-1]))

    elif q["type"] == "identify-tf_blank":
        cond_1_str, (bbox_cond_1_str, bbox_img_msg) = process_cond(q["condition_1"], q_msgs[0])
        cond_2_str, (bbox_cond_2_str, bbox_img_msg_2) = process_cond(q["condition_2"], q_msgs[0] if bbox_img_msg is None else bbox_img_msg)
        if bbox_img_msg_2 is not None:
            bbox_img_msg = bbox_img_msg_2
        q_str, a = ask_blank_about_condition(q["answer"] if "answer" in q else q["fake_answer"])
        q_msgs.append(
            {
                "type": "text",
                "text": f"Resolution of the image provided is {q_msgs[0]['width']}x{q_msgs[0]['height']}. There might be one person in the image that meets the following two conditions:\n\n- {cond_1_str}\n- {cond_2_str}\n\nPlease answer the following question if there is such a person. Or else, please provide \"unknown\" as answer:\n\n{q_str}"
            }
        )
        answer = {
            "type": "tf_blank",
            "data": a if "answer" in q else "unknown"
        }
        results.append({
            "type": "identify-tf_blank",
            "question": q_msgs,
            "answer": answer,
            "data": q
        })
        if ((bbox_cond_1_str is not None) or (bbox_cond_2_str is not None)) and (bbox_img_msg is not None):
            new_q_msgs = deepcopy(q_msgs)
            new_q_msgs[0] = bbox_img_msg
            new_q_msgs[1] = {
                "type": "text",
                "text": f"Resolution of the image provided is {bbox_img_msg['width']}x{bbox_img_msg['height']}. There might be one person in the image that meets the following two conditions:\n\n- {bbox_cond_1_str if bbox_cond_1_str is not None else cond_1_str}\n- {bbox_cond_2_str if bbox_cond_2_str is not None else cond_2_str}\n\nPlease answer the following question if there is such a person. Or else, please provide \"unknown\" as answer:\n\n{q_str}"
            }
            results.append({
                "type": "identify-tf_blank",
                "question": new_q_msgs,
                "answer": answer,
                "data": q
            })
        if "fake_answer" not in q and q["can_mutate_hand_to_false"] and q["condition_1"]["attr_type"] == "hoi" and len(set([r[0] for r in q["condition_1"]["attr_value"]["relation"]]) & set(["left hand", "right hand"])) > 0:
            new_q = deepcopy(q)
            new_q["condition_1"]["attr_value"]["relation"] = [("left hand" if r[0] == "right hand" else "right hand", r[1]) for r in new_q["condition_1"]["attr_value"]["relation"] if r[0] in ["left hand", "right hand"]]
            new_q["fake_answer"] = new_q["answer"]
            new_q.pop("answer")
            new_q["can_mutate_hand_to_false"] = "MUTATED"
            results += multi_mixed_gen_qa(new_q, deepcopy(q_msgs[:-1]))

    elif q["type"] == "identify-open_grounding":
        cond_str, (bbox_cond_str, bbox_img_msg) = process_cond(q["condition"], q_msgs[0])
        q_msgs.append(
            {
                "type": "text",
                "text": f"Resolution of the image provided is {q_msgs[0]['width']}x{q_msgs[0]['height']}. There is one person in the image that meets the following condition:\n\n- {cond_str}\n\nPlease provide the name and bounding box in xyxy format of the object that have interation \"{get_hoi_description(q['answer']['attr_value']['object'], q['answer']['attr_value']['relation'], no_obj_name=True)}\" with the person."
            }
        )
        answer = {
            "type": "open_hoi",
            "data":{
                "name": q['answer']['attr_value']['object'],
                "bbox": q['answer']['attr_value']['bbox']
            }
        }
        results.append({
            "type": "identify-open_grounding",
            "question": q_msgs,
            "answer": answer,
            "data": q
        })
        if bbox_cond_str is not None and bbox_img_msg is not None:
            new_q_msgs = deepcopy(q_msgs)
            new_q_msgs[0] = bbox_img_msg
            new_q_msgs[1] = {
                "type": "text",
                "text": f"Resolution of the image provided is {q_msgs[0]['width']}x{q_msgs[0]['height']}. There is one person in the image that meets the following condition:\n\n- {bbox_cond_str}\n\nPlease provide the name and bounding box in xyxy format of the object that have interation \"{get_hoi_description(q['answer']['attr_value']['object'], q['answer']['attr_value']['relation'], no_obj_name=True)}\" with the person."
            }
        
            results.append({
                "type": "identify-open_grounding",
                "question": new_q_msgs,
                "answer": answer,
                "data": q
            })
    elif q["type"] == "common_choice":
        choices = []
        bbox_choices = []
        bbox_used = False
        bbox_img_msg = None
        for choice_cond in [q["true_answer"]] + q["false_answers"]:
            choice_str, (bbox_choice_str, bbox_choice_img_msg) = process_cond(choice_cond, q_msgs[0] if bbox_img_msg is None else bbox_img_msg)
            choices.append((choice_cond == q["true_answer"], choice_str))
            if bbox_choice_str is not None:
                bbox_choices.append((choice_cond == q["true_answer"], bbox_choice_str))
                bbox_used = True
                bbox_img_msg = bbox_choice_img_msg
            else:
                bbox_choices.append((choice_cond == q["true_answer"], choice_str))

        random.shuffle(choices)
        true_selection = []
        selections = []
        for seq, (tf, cond_str) in zip(["A", "B", "C", "D"], choices):
            if tf:
                true_selection.append(seq)
            selections.append(f"{seq}. {cond_str}")
        selection_str = "\n".join(selections)
        q_msgs.append(
            {
                "type": "text",
                "text": f"Resolution of the image provided is {q_msgs[0]['width']}x{q_msgs[0]['height']}. Please select the option that fits most or all of persons in the image:\n\n{selection_str}\n\nPlease provide the option letter of the most possible answer."
            }
        )
        answer = {
            "type": "choice",
            "data":true_selection
        }
        results.append({
            "type": "common_choice",
            "question": q_msgs,
            "answer": answer,
            "data": q
        })
        if bbox_used:
            random.shuffle(bbox_choices)
            true_selection = []
            selections = []
            for seq, (tf, cond_str) in zip(["A", "B", "C", "D"], bbox_choices):
                if tf:
                    true_selection.append(seq)
                selections.append(f"{seq}. {cond_str}")
            selection_str = "\n".join(selections)
            new_q_msgs = deepcopy(q_msgs)
            new_q_msgs[0] = bbox_img_msg
            new_q_msgs[1] = {
                "type": "text",
                "text": f"Resolution of the image provided is {bbox_img_msg['width']}x{bbox_img_msg['height']}. Please select the option that fits most or all of persons in the image:\n\n{selection_str}\n\nPlease provide the option letter of the most possible answer."
            }
            answer = {
                "type": "choice",
                "data":true_selection
            }
            results.append({
                "type": "common_choice",
                "question": new_q_msgs,
                "answer": answer,
                "data": q
            })
    else:
        raise RuntimeError(f"Cant handle multi mixed question type {q['type']}")
    return results



async def generate_qa(question_list):
    """....."""
    async def process_question(q):
        question_msgs = []
        answer_msg = {}
        if q["type"] in ["multiface", "multihoi", "multiclothing"]:
            q_msg = {
                "type": "multi_image",
                "image_A": None,
                "image_B": None,
                "image_C": None,
                "image_D": None
            }
            if q["type"] == "multihoi":
                img_pairs = []
                img_pairs.append((True, q["full"]))
                img_pairs.append((False, q["diff_position"]))
                img_pairs.append((False, q["diff_object"]))
                img_pairs.append((False, q["diff_extra"]))
                random.shuffle(img_pairs)
                for seq, (is_true, img) in zip(["A", "B", "C", "D"], img_pairs):
                    q_msg[f"image_{seq}"] = img
                    if is_true:
                        answer_msg = {
                            "type": "choice",
                            "answer": seq
                        }
            else:
                ans = [None, None, None, None]
                img_pairs = []
                img_pairs.append((0, q["none"]))
                img_pairs.append((1, q["solo"]))
                img_pairs.append((2, q["duo"]))
                img_pairs.append((3, q["fullfit"]))
                random.shuffle(img_pairs)
                for seq, (prop, img) in zip(["A", "B", "C", "D"], img_pairs):
                    q_msg[f"image_{seq}"] = img
                    ans[3-prop] = seq
                answer_msg = {
                    "type": "sequence",
                    "answer": ans
                }
            question_msgs.append(
                q_msg
            )
        else:
            W, H = Image.open(q["image"]).size
            question_msgs.append(
                {
                    "type": "image",
                    "image": q["image"],
                    "width": W,
                    "height": H
                }
            )
            
        answers = []
        result = None
        
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
                    "text": f"Please select the facial features of the person in the image from the following options (only one selection is allowed):\n\n{selection_str}\n\nPlease provide the option letters of the most possible answer."
                }
            )
            answer_msg = {
                "type": "choice",
                "data": true_selection
            }
            result = {
                "type": "face_choice",
                "question": question_msgs,
                "answer": answer_msg,
                "data": q
            }
        elif q["type"] == "cloth_choice":
            for true_answer in q["true_answers"]:
                answers.append((True,  await asyncio.to_thread(get_cloth_description, true_answer["name"], true_answer["color"])))
            for false_answer in q["false_answers"]:
                answers.append((False, await asyncio.to_thread(get_cloth_description, false_answer["name"], false_answer["color"])))
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
                    "text": f"Please select the wearing of the person in the image from the following options (only one selection is allowed):\n\n{selection_str}\n\nPlease provide the option letters of the most possible answer."
                }
            )
            answer_msg = {
                "type": "choice",
                "data": true_selection
            }
            result = {
                "type": "cloth_choice",
                "question": question_msgs,
                "answer": answer_msg,
                "data": q
            }
        elif q["type"] == "hoi_choice":
            for true_answer in q["true_answers"]:
                answers.append((True,  await asyncio.to_thread(get_hoi_description, true_answer["object"], true_answer["relation"])))
            for false_answer in q["false_answers"]:
                answers.append((False, await asyncio.to_thread(get_hoi_description, false_answer["object"], false_answer["relation"])))
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
                    "text": f"Please select the option that best describes the interaction between the person and object (only one selection is allowed):\n\n{selection_str}\n\nPlease provide the option letters of all correct choices, separated by commas if multiple."
                }
            )
            answer_msg = {
                "type": "choice",
                "data": true_selection
            }
            result = {
                "type": "hoi_choice",
                "question": question_msgs,
                "answer": answer_msg,
                "data": q
            }
        elif q["type"] == "face_grounding":
            question_msgs.append(
                {
                    "type": "text",
                    "text": f"Resolution of the image provided is {W}x{H}. Please provide the bounding box of the facial part \"{q['question']['attr_name']}\" of the main person in the image."
                }
            )
            answer = {
                "type": "bbox",
                "data":q['question']["attr_value"]
            }
            result = {
                "type": "face_grounding",
                "question": question_msgs,
                "answer": answer,
                "data": q
            }
        elif q["type"] == "hoi_grounding":
            question_msgs.append(
                {
                    "type": "text",
                    "text": f"Resolution of the image provided is {W}x{H}. Please provide the bounding box (in xyxy format) of the object that have interation \"{get_hoi_description(q['question']['attr_value']['object'], q['question']['attr_value']['relation'], no_obj_name=True)}\" with the main person in the image."
                }
            )
            answer = {
                "type": "bbox",
                "data":q['question']["attr_value"]["bbox"]
            }
            result = {
                "type": "hoi_grounding",
                "question": question_msgs,
                "answer": answer,
                "data": q
            }
        elif q["type"] == "body_grounding":
            question_msgs.append(
                {
                    "type": "text",
                    "text": f"Resolution of the image provided is {W}x{H}. Please provide the bounding box (in xyxy format) of the \"{q['question']['attr_name'].replace('_', ' ').replace('body', 'whole visible body')}\" of the main person in the image."
                }
            )
            answer = {
                "type": "bbox",
                "data":q['question']["attr_value"]
            }
            result = {
                "type": "body_grounding",
                "question": question_msgs,
                "answer": answer,
                "data": q
            }
        elif q["type"] == "open_hoi":
            question_msgs.append(
                {
                    "type": "text",
                    "text": f"Please name the object that have interation \"{get_hoi_description(q['question']['attr_value']['object'], q['question']['attr_value']['relation'], no_obj_name=True)}\" with the main person in the image."
                }
            )
            
            answer = {
                "type": "blank",
                "data": q['question']["attr_value"]["object"]
            }
            result = {
                "type": "open_hoi",
                "question": question_msgs,
                "answer": answer,
                "data": q
            }
        elif q["type"] == "open_clothing":
            question_msgs.append(
                {
                    "type": "text",
                    "text": f"Please give name and color of the clothing item of type \"{q['question']['attr_value']['type']}\" that the main person is wearing in the image."
                }
            )
            answer = {
                "type": "blank",
                "data": get_cloth_description(q['question']['attr_value']['name'], q['question']['attr_value']['color'])
            }
            result = {
                "type": "open_clothing",
                "question": question_msgs,
                "answer": answer,
                "answer_detail": answer,
                "data": q
            }
        elif q["type"] == "emotion":
            answers.append((True,  q["true_answer"]))
            for false_answer in q["false_answers"]:
                answers.append((False, false_answer))
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
                    "text": f"Please select the best analysis of emotion for someone appearing in the image:\n\n{selection_str}\n\nPlease provide the option letter of the most possible answer."
                }
            )
            answer_msg = {
                "type": "choice",
                "data": true_selection
            }
            result = {
                "type": "emotion",
                "question": question_msgs,
                "answer": answer_msg,
                "data": q
            }
        elif q["type"] == "intention":
            answers.append((True,  q["true_answer"]))
            for false_answer in q["false_answers"]:
                answers.append((False, false_answer))
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
                    "text": f"Please select the best analysis of intention for someone appearing in the image:\n\n{selection_str}\n\nPlease provide the option letter of the most possible answer."
                }
            )
            answer_msg = {
                "type": "choice",
                "data": true_selection
            }
            result = {
                "type": "intention",
                "question": question_msgs,
                "answer": answer_msg,
                "data": q
            }
        elif q["type"] == "causal":
            answers.append(("past",  q["true_answer"][0]))
            answers.append(("future",  q["true_answer"][1]))
            for false_answer in q["false_answers"]:
                answers.append((None, false_answer))
            random.shuffle(answers)
            true_selection = {}
            selection_strs = []
            selection_str = ""
            for serial, (is_true, text) in zip(["A", "B", "C", "D"], answers):
                if is_true == "past":
                    true_selection["past"] = serial
                elif is_true == "future":
                    true_selection["future"] = serial
                selection_strs.append(f"{serial}. {text}")
            selection_str = "\n".join(selection_strs)
            question_msgs.append(
                {
                    "type": "text",
                    "text": f"Please select the best analysis of what happened in the past and what will happen in the future:\n\n{selection_str}\n\nPlease provide the option letters of the most possible answer separately for past and future."
                }
            )
            answer_msg = {
                "type": "double_choice",
                "data": true_selection
            }
            result = {
                "type": "causal",
                "question": question_msgs,
                "answer": answer_msg,
                "data": q
            }
        elif q["type"] == "multiclothing":
            clothing_desc_list = []
            for cloth in q["combine"]:
                clothing_desc_list.append(get_cloth_description(cloth["name"], cloth["color"]))
            clothing_desc = "\n".join([f"- {desc}" for desc in clothing_desc_list])
            question_msgs.append(
                {
                    "type": "text",
                    "text": f"{clothing_desc}\n\nListed are some clothing items that appeared in the four images above. Please give the sequence of four images by the maximum count of clothing listed that appears in one single person. If someone in a specific image is wearing all three clothing items, it should be the first image in your answer, and if none of three clothing items are present, it should be the last image. Please provide a explicit sequence of four images by their letters."
                }
            )
            
            result = {
                "type": "multiclothing",
                "question": question_msgs,
                "answer": answer_msg,
                "data": q
            }
        elif q["type"] == "multiface":
            face_desc_list = []
            for face in q["combine"]:
                face_desc_list.append(FACE_ATTR_DESCRIPTIONS[face][0])
            face_desc = "\n".join([f"- {desc}" for desc in face_desc_list])
            question_msgs.append(
                {
                    "type": "text",
                    "text": f"{face_desc}\n\nListed are some facial attributes that appeared in the four images above. Please give the sequence of four images by the maximum count of facial attributes that appears in one single person. If someone in a specific image is showing all three facial attributes, it should be the first image in your answer, and if none of three facial attributes are present, it should be the last image. Please provide a explicit sequence of four images by their letters."
                }
            )

            result = {
                "type": "multiface",
                "question": question_msgs,
                "answer": answer_msg,
                "data": q
            }
        elif q["type"] == "multihoi":
            hoi_desc = get_hoi_description(q["object"], q["hoi"])
            question_msgs.append(
                {
                    "type": "text",
                    "text": f"This is a description of a human-object interaction: \"{hoi_desc}\"\n\nWhich one of four images listed above best represents this interaction? Provide your answer with the corresponding image letter."
                }
            )

            result = {
                "type": "multihoi",
                "question": question_msgs,
                "answer": answer_msg,
                "data": q
            }
        elif q["type"] in ["identify-grounding", "identify-blank", "identify-choice", "identify-tf_grounding", "identify-tf_blank", "identify-open_grounding", "common_choice"]:
            result = multi_mixed_gen_qa(q, question_msgs)
        else:
            print(f"Unsupported question type: {q['type']}")
        
        return result
    
    # ........，......64
    from asyncio import Semaphore
    from tqdm import tqdm
    
    sem = Semaphore(64)
    
    async def process_with_semaphore(q):
        async with sem:
            return await process_question(q)
    
    # ..tqdm....
    with tqdm(total=len(question_list), desc="....") as pbar:
        async def process_and_update(q):
            result = await process_with_semaphore(q)
            pbar.update(1)
            return result
            
        tasks = [process_and_update(q) for q in question_list]
        results = await asyncio.gather(*tasks)
    
    ret = []
    for r in results:
        if isinstance(r, list):
            ret.extend(r)
        elif r is not None:
            ret.append(r)
    # ... None .....
    return ret

async def main():
    questions = []
    # for fn in ["multi_mixed_feature_questions.json", "multi_hoi_feature_questions.json", "multi_clothing_feature_questions.json", "multi_face_feature_questions.json", "single_feature_questions.json", "abstract_feature_questions.json"]:
    #     with open(fn, "r") as f:
    #         questions.extend(json.load(f))
    # qs = sample_question(questions)
    # single_feature_qa = await generate_qa(qs)
    for dcn in os.listdir("final_qa/cloth_choice"):
        dcp = os.path.join("final_qa/cloth_choice", dcn)
        if not os.path.isdir(dcp):
            continue
        with open(os.path.join(dcp, "qa.json"), "r") as qa_f:
            questions.append(json.load(qa_f)["data"])
    for dcn in os.listdir("final_qa/hoi_choice"):
        dcp = os.path.join("final_qa/hoi_choice", dcn)
        if not os.path.isdir(dcp):
            continue
        with open(os.path.join(dcp, "qa.json"), "r") as qa_f:
            questions.append(json.load(qa_f)["data"])
    
    final_qa = await generate_qa(questions)
    print(f"Generated {len(final_qa)} QA pairs.")
    for i, qa in enumerate(final_qa):
        # if qa["question"][0]["type"] != "image_bbox":
        #     continue
        qa_type = qa["type"]
        qa_dir = os.path.join("final_qa", qa_type, str(i+25104))
        os.makedirs(qa_dir, exist_ok=True)
        with open(os.path.join(qa_dir, "qa.json"), "w") as qa_f:
            json.dump(qa, qa_f, indent=4)
        q, imgs = history_to_str(qa["question"])
        for img in imgs:
            dst = os.path.join(qa_dir, os.path.basename(img))
            # ........256
            with Image.open(img) as im:
                im.thumbnail((256, 256))
                im.save(dst)
        with open(os.path.join(qa_dir, "qa.txt"), "w") as q_f:
            q_f.write(f"{q}\n\n----------\n\n{str(qa['answer'])}")
            print(f"Saved QA to {os.path.join(qa_dir, 'qa.txt')}")

def sample_question(questions):
    random.seed(42)
    type_count_map = {
        "causal": 500,
        "intention": 500,
        "emotion": 300,
        "body_grounding": 200,
        "common_choice": 20,
        "face_grounding": 150,
        "hoi_choice": 100,
        "hoi_grounding": 100,
        "identify-blank": 3,
        "identify-choice": 2,
        "identify-grounding": 8,
        "identify-open_grounding": 10,
        "identify-tf_blank": 3,
        "identify-tf_grounding": 2,
        "multiclothing": 50,
        "multiface": 25,
        "multihoi": 15,
        "open_hoi": 50,
        "open_clothing": 50,
    }
    distinct_count = {}
    distinct_question_map = {}
    filename_appearance_count = {}
    all_filenames = {}
    result = []
    distinct_sequence = []
    for q in questions:
        if id(q) not in all_filenames:
            all_filenames[id(q)] = []
            if "image" in q:
                all_filenames[id(q)].append(q["image"])
            if "fullfit" in q:
                all_filenames[id(q)].append(q["fullfit"])
            if "solo" in q:
                all_filenames[id(q)].append(q["solo"])
            if "duo" in q:
                all_filenames[id(q)].append(q["duo"])
            if "none" in q:
                all_filenames[id(q)].append(q["none"])
            if "full" in q:
                all_filenames[id(q)].append(q["full"])
            if "diff_position" in q:
                all_filenames[id(q)].append(q["diff_position"])
            if "diff_object" in q:
                all_filenames[id(q)].append(q["diff_object"])
            if "diff_extra" in q:
                all_filenames[id(q)].append(q["diff_extra"])
            for fn in all_filenames[id(q)]:
                filename_appearance_count[fn] = filename_appearance_count.get(fn, 0) + 1
        for d in q.get("distinct", []):
            if d not in distinct_sequence:
                distinct_sequence.append(d)
            if d not in distinct_count:
                distinct_count[d] = type_count_map.get(q["type"], 50)
            if d not in distinct_question_map:
                distinct_question_map[d] = [q]
            else:
                distinct_question_map[d].append(q)
    def filename_score(fn):
        return min([filename_appearance_count.get(f, 0) for f in all_filenames[id(fn)]])
    for d in distinct_sequence:
        qs = distinct_question_map.get(d, [])
        if distinct_count[d] > 0:
            # sqs = random.sample(qs, min(distinct_count[d], len(qs)))
            sqs = sorted(qs, key=lambda x: filename_score(x))[:min(distinct_count[d], len(qs))]
            result.extend(sqs)
            for q in sqs:
                for qd in q.get("distinct", []):
                    distinct_count[qd] -= 1
                for fn in all_filenames[id(q)]:
                    filename_appearance_count[fn] += 100000
    print(distinct_count)
    return result




if __name__ == "__main__":
    # import asyncio
    
    
    dataset_pictures = [Picture(i) for i in get_full_data()]
    # .........
    # single_feature_generator = SinglePersonQuestions(dataset_pictures)
    # single_feature_generator.filter_pictures()
    # single_feature_questions = single_feature_generator.generate_questions()
    # if single_feature_questions:
    #     single_feature_questions = sample_question(single_feature_questions)
    #     single_feature_generator.save_questions(single_feature_questions, "single_feature_questions.json")
    

    # ..........
    # multi_face_generator = MultiFaceFeatureQuestionGenerator(dataset_pictures)
    # multi_face_generator.filter_pictures()
    # face_questions = multi_face_generator.generate_questions()
    # if face_questions:
    #     multi_face_generator.save_questions(face_questions, "multi_face_feature_questions.json")
    

    # ............
    # multi_clothing_generator = MultiPersonClothingFeatureQuestionGenerator(dataset_pictures)
    # multi_clothing_generator.filter_pictures()
    # clothing_questions = multi_clothing_generator.generate_questions()
    
    # if clothing_questions:
    #     multi_clothing_generator.save_questions(clothing_questions, "multi_clothing_feature_questions.json")

    # .....-.......
    # multi_hoi_generator = MultiImageHoiFeatureQuestionGenerator(dataset_pictures)
    # multi_hoi_generator.filter_pictures()
    # hoi_questions = multi_hoi_generator.generate_questions()
    # if hoi_questions:
    #     multi_hoi_generator.save_questions(hoi_questions, "multi_hoi_feature_questions.json")

    # ..............
    # multi_mixed_generator = ManyPersonMixedFeatureQuestionGenerator(dataset_pictures)
    # multi_mixed_generator.filter_pictures()
    # mixed_questions = multi_mixed_generator.generate_questions()
    # if mixed_questions:
    #     multi_mixed_generator.save_questions(mixed_questions, "multi_mixed_feature_questions.json")

    # .........
    # abstract_generator = AbstractSingleImageChoiceQuestionGenerator(dataset_pictures)
    # abstract_generator.filter_pictures()
    # abstract_questions = abstract_generator.generate_questions()
    # if abstract_questions:
    #     abstract_generator.save_questions(abstract_questions, "abstract_feature_questions.json")

    asyncio.run(main())

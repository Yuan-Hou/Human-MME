import json
import os
from typing import Dict, List
import itertools
import concurrent.futures
import threading
from test_framework import QuestionGenerator, POSITION_INCLUDE_MAP, POSITION_EXCLUDE_MAP, POSITION_SIMPLIFIER
from utils import ask_question
from sentence_transformers import SentenceTransformer, util



class MultiImageHoiFeatureQuestionGenerator(QuestionGenerator):
    """...-.........."""
    def __init__(self, dataset_pictures):
        super().__init__(dataset_pictures)
        self.synonym_dict = {}
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.word_embs = {}
        self.position_include_map = POSITION_INCLUDE_MAP
        self.position_exclude_map = POSITION_EXCLUDE_MAP


    

    def generate_questions(self):
        questions = []
        def synonym_expand(word_list):
            result = set(word_list)
            for word in word_list:
                result.update(self.synonym_dict.get(word, []))
            return result

        def find_hoi_match(objs=None, actions=None, positions=None, exclude_objs=None, exclude_actions=None, exclude_positions=None, exclude_picture=None):
            results = []
            if objs is not None:
                objs = synonym_expand(objs)
            if actions is not None:
                actions = synonym_expand(actions)
            if exclude_objs is not None:
                exclude_objs = synonym_expand(exclude_objs)
            if exclude_actions is not None:
                exclude_actions = synonym_expand(exclude_actions)
            positions = set(positions) if positions is not None else None
            exclude_positions = set(exclude_positions) if exclude_positions is not None else None

            if actions is not None and ("holding" in actions or "hold" in actions):
                if exclude_positions is None:
                    exclude_positions = set(["hand", "both hands", "left hand", "right hand"])
                else:
                    exclude_positions = set(exclude_positions) | set(["hand", "both hands", "left hand", "right hand"])

            if positions is not None and ("hand" in positions or "both hands" in positions or "left hand" in positions or "right hand" in positions):
                if exclude_actions is None:
                    exclude_actions = set(["holding", "hold", "holds"])
                else:
                    exclude_actions = set(exclude_actions) | set(["holding", "hold", "holds"])
                if exclude_actions is not None:
                    exclude_actions = synonym_expand(exclude_actions)

            for picture in self.dataset_pictures:
                has_match = False
                for hoi in picture.full_hoi():
                    if exclude_picture is not None and picture == exclude_picture:
                        continue
                    if objs is not None and len(objs & set(hoi.get_object_names())) == 0:
                        continue
                    if actions is not None and len(actions & set(hoi.get_actions())) == 0:
                        continue
                    if positions is not None and len(positions & set(hoi.get_positions())) == 0:
                        continue
                    if exclude_objs is not None and len(exclude_objs & set(hoi.get_object_names())) > 0:
                        continue
                    if exclude_actions is not None and len(exclude_actions & set(hoi.get_actions())) > 0:
                        continue
                    if exclude_positions is not None and len(exclude_positions & set(hoi.get_positions())) > 0:
                        continue
                    has_match = True
                    break
                if not has_match:
                    continue
                for hoi in picture.full_hoi():
                    if exclude_objs is not None and len(exclude_objs & set(picture.object_names())) > 0:
                        has_match = False
                        break
                    if exclude_actions is not None and len(exclude_actions & set(hoi.get_actions())) > 0:
                        has_match = False
                        break
                    if exclude_positions is not None and len(exclude_positions & set(hoi.get_positions())) > 0:
                        has_match = False
                        break
                if has_match:
                    results.append(picture)
            return results

        for idx, picture in enumerate(self.dataset_pictures):
            for person in picture.persons:
                for hoi in person.hois:
                    pos = hoi.get_positions()
                    act = hoi.get_actions()
                    obj_name = hoi.get_object_name()
                    include_positions = []
                    exclude_positions = []
                    exclude_acts = []
                    for pos_name in pos:
                        if pos_name in self.position_include_map:
                            include_positions.extend(self.position_include_map[pos_name] + [pos_name])
                        else:
                            include_positions.extend([pos_name])

                    for person in picture.persons:
                        for h in person.hois:
                            if h.get_object_name() != obj_name:
                                continue
                            for p in h.get_positions():
                                if p in self.position_exclude_map:
                                    exclude_positions.extend(self.position_exclude_map[p] + [p])
                                else:
                                    exclude_positions.extend([p])
                            
                            exclude_acts.extend(h.get_actions())

                    diff_pos = find_hoi_match(objs=hoi.get_object_names(), actions=act, exclude_positions=exclude_positions, exclude_picture=picture)
                    diff_act = find_hoi_match(objs=hoi.get_object_names(), positions=include_positions, exclude_actions=exclude_acts, exclude_picture=picture)
                    diff_obj = find_hoi_match(actions=act, positions=include_positions, exclude_objs=hoi.get_object_names(), exclude_picture=picture)
                    if len(diff_pos)  + len(diff_obj) > 2 and len(diff_pos) > 0 and len(diff_obj) > 0:
                        # ...........
                        diff_pos.sort(key=lambda x: self.picture_occurrence.get(x, 0))
                        diff_obj.sort(key=lambda x: self.picture_occurrence.get(x, 0))
                        
                        position_diff = []
                        happen_to_obj = set(synonym_expand(hoi.get_object_names())) | set(hoi.get_object_names())
                        
                        for p in diff_pos[0].full_hoi():
                            if len(set(p.get_object_names()) & happen_to_obj) > 0:
                                position_diff.extend(p.get_positions())

                        extra_pos_diff = []
                        diff_ext = None
                        name_cate = ""
                        for name, syns in self.synonym_dict.items():
                            if obj_name == name or obj_name in syns:
                                name_cate = name
                                break
                        q = {
                            "type": "multihoi",
                            "object": hoi.get_object_name(),
                            "hoi": list(hoi.get_position_action_pairs()),
                            "full": picture.image_path(),
                            "diff_object": diff_obj[0].image_path(),
                            "object_diff": [h.get_object_name() for h in diff_obj[0].full_hoi()],
                            "diff_position": diff_pos[0].image_path(),
                            "position_diff": position_diff,
                            "distinct": [f"multihoi-{name_cate.lower()}-{list(hoi.get_position_action_pairs())[0][0]}"]
                        }
                        if len(diff_pos) > 1:
                            diff_ext = diff_pos[1]
                            for p in diff_pos[1].full_hoi():
                                if len(set(p.get_object_names()) & happen_to_obj) > 0:
                                    extra_pos_diff.extend(p.get_positions())
                            q["extra_type"] = "position"
                            q["extra_diff"] = extra_pos_diff
                            q["diff_extra"] = diff_ext.image_path()
                        else:
                            diff_ext = diff_obj[1]
                            q["extra_type"] = "object"
                            q["extra_diff"] = [h.get_object_name() for h in diff_ext.full_hoi()]
                            q["diff_extra"] = diff_ext.image_path()

                        # ...........
                        questions.append(q)
                        print(f"Found {len(questions)} multi-image HOI feature questions so far. {idx}/{len(self.dataset_pictures)}")
                        self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
                        self.picture_occurrence[diff_obj[0]] = self.picture_occurrence.get(diff_obj[0], 0) + 1
                        # self.picture_occurrence[diff_act[0]] = self.picture_occurrence.get(diff_act[0], 0) + 1
                        self.picture_occurrence[diff_pos[0]] = self.picture_occurrence.get(diff_pos[0], 0) + 1
                        self.picture_occurrence[diff_ext] = self.picture_occurrence.get(diff_ext, 0) + 1

        return questions

    def _construct_infos(self):
        """......."""
        self.synonym_dict = {}
        action_set = set()
        object_set = set()
        position_set = set()
        for picture in self.dataset_pictures:
            for person in picture.persons:
                for hoi in person.hois:
                    position_set.update(hoi.get_positions())
                    action_set.update(hoi.get_actions())
                    action_set.update(hoi.get_negative_actions())
                    object_set.update(hoi.get_object_names())

        for obj in picture.hoi_objects:
            object_set.update(obj.raw_data.get("possible_names", []))

        print(f"actions: {action_set}")
        print("-"*20)
        print(f"objects: {object_set}")
        print("-"*20)
        print(f"positions: {position_set}")
        print("-"*20)
        for action in action_set:
            self.word_embs[action] = self.sentence_model.encode(action, convert_to_tensor=True)
        for obj in object_set:
            self.word_embs[obj] = self.sentence_model.encode(obj, convert_to_tensor=True)
        self._construct_synonym_dict(object_set, action_set)

    def filter_pictures(self):
        """........."""
        filtered_pictures = []
        for picture in self.dataset_pictures:
            # ..........HOI
            if any(len(person.hois) > 0 for person in picture.persons):
                filtered_pictures.append(picture)
        print(f"Filtered down to {len(filtered_pictures)} records for multi-person clothing feature questions.")
        self.dataset_pictures = filtered_pictures
        self._construct_infos()
        return filtered_pictures
    
    def _construct_synonym_dict(self, name_list, action_list):
        """.......，..16.....，......"""
        # ...........
        lock = threading.Lock()
        cnt = 0
        
        # ............
        existing_synonyms = {}
        if os.path.exists("hoi_synonym_dict.json"):
            try:
                with open("hoi_synonym_dict.json", "r") as f:
                    existing_data = json.load(f)
                    existing_synonyms = existing_data.get("synonyms", {})
                print(f"Loaded existing synonym dictionary with {len(existing_synonyms)} entries.")
            except (json.JSONDecodeError, FileNotFoundError):
                print("Could not load existing synonym dictionary, starting fresh.")
        
        # ...........
        self.synonym_dict.update(existing_synonyms)
        
        def process_name_combination(combo):
            name1, name2 = combo
            if util.cos_sim(self.word_embs[name1], self.word_embs[name2]).item() < 0.2:
                ans = "no"
            else:
                ans = ask_question(f"'{name1}' and '{name2}' are words discribing two objects. Please analyze their meanings and decide if they are looking alike, of same meaning, or one of them can be a part of the other visually. At the end of your answer, please put a single line of 'yes' if they are some kind of synonymous or might have some visual belonging relationship as said, put 'no' if they are not.")
            yes_idx = ans[::-1].lower().find("yes"[::-1])
            no_idx = ans[::-1].lower().find("no"[::-1])
            if yes_idx == -1:
                yes_idx = float('inf')
            if no_idx == -1:
                no_idx = float('inf')
            if yes_idx < no_idx:
                print(ans)
                print(yes_idx, no_idx)
            return (name1, name2, yes_idx < no_idx)

        def process_action_combination(combo):
            action1, action2 = combo
            if util.cos_sim(self.word_embs[action1], self.word_embs[action2]).item() < 0.2:
                ans = "no"
            else:
                ans = ask_question(f"'{action1}' and '{action2}' are words discribing two actions for human to interact with objects. Please analyze their meanings and decide if they are possoible look alike in static images, of same meaning, or one of them belong to the other. At the end of your answer, please put a single line of 'yes' if they might look alike as said or 'no' if they are not.")
            yes_idx = ans[::-1].lower().find("yes"[::-1])
            no_idx = ans[::-1].lower().find("no"[::-1])
            if yes_idx == -1:
                yes_idx = float('inf')
            if no_idx == -1:
                no_idx = float('inf')
            if yes_idx < no_idx:
                print(ans)
                print(yes_idx, no_idx)
            return (action1, action2, yes_idx < no_idx)
        
        def combination_already_processed(item1, item2):
            """..........."""
            # ...........
            if item1 in existing_synonyms and item2 in existing_synonyms:
                return True
            return False
        
        # .....
        for name in name_list:
            if name not in self.synonym_dict:
                self.synonym_dict[name] = []

        for action in action_list:
            if action not in self.synonym_dict:
                self.synonym_dict[action] = []

        # ......，........
        name_combinations = [combo for combo in itertools.combinations(name_list, 2) 
                           if not combination_already_processed(combo[0], combo[1])]
        total_name_combinations = len(name_combinations)
        
        print(f"Found {total_name_combinations} new name combinations to process.")
        
        if total_name_combinations > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                future_to_combo = {executor.submit(process_name_combination, combo): combo for combo in name_combinations}
                
                for future in concurrent.futures.as_completed(future_to_combo):
                    name1, name2, is_synonym = future.result()
                    
                    with lock:
                        if is_synonym:
                            self.synonym_dict[name1].append(name2)
                            self.synonym_dict[name2].append(name1)
                        
                        cnt += 1
                        print(f"Processed {cnt}/{total_name_combinations} name combinations so far.")
                        if cnt % 2000 == 0:
                            
                            with open("hoi_synonym_dict.json", "w") as f:
                                json.dump({
                                    "synonyms": self.synonym_dict,
                                }, f)
        
        # ......，........
        action_combinations = [combo for combo in itertools.combinations(action_list, 2) 
                            if not combination_already_processed(combo[0], combo[1])]
        total_action_combinations = len(action_combinations)
        processed_actions = 0

        print(f"Found {total_action_combinations} new action combinations to process.")

        if total_action_combinations > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                future_to_combo = {executor.submit(process_action_combination, combo): combo for combo in action_combinations}
                
                for future in concurrent.futures.as_completed(future_to_combo):
                    action1, action2, is_synonym = future.result()

                    with lock:
                        if is_synonym:
                            self.synonym_dict[action1].append(action2)
                            self.synonym_dict[action2].append(action1)

                        processed_actions += 1
                        if processed_actions % 100 == 0:  # ......，......
                            print(f"Processed {processed_actions}/{total_action_combinations} action combinations so far.")
                            with open("hoi_synonym_dict.json", "w") as f:
                                json.dump({
                                    "synonyms": self.synonym_dict
                                }, f)
        
        # ....
        with open("hoi_synonym_dict.json", "w") as f:
            json.dump({
                "synonyms": self.synonym_dict,
            }, f)

        print(f"Completed processing {total_name_combinations} new name combinations and {total_action_combinations} new color combinations.")
        print(f"Total entries in synonym dictionary: {len(self.synonym_dict)}")

    
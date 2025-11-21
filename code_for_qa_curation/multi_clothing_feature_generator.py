import json
import os
from typing import Dict, List
import itertools
import concurrent.futures
import threading
from test_framework import QuestionGenerator
from utils import ask_question

class MultiPersonClothingFeatureQuestionGenerator(QuestionGenerator):
    """多图人体服装特征题型生成器"""
    def __init__(self, dataset_pictures):
        super().__init__(dataset_pictures)
        self.clothing_color_name_2_picture_dict: Dict[str, Dict[str, List]] = {}
        self.clothing_name_color_2_picture_dict: Dict[str, Dict[str, List]] = {}
        self.synonym_dict: Dict[str, List[str]] = {}
        self.distinguishable_dict: Dict[str, List[str]] = {}
        self.clothing_freq_dict: Dict[str, int] = {}

    def _construct_clothing_dict(self, filtered_pictures):
        for pic in filtered_pictures:
            for person in pic.persons:
                clothings = person.get_clothing_list()
                for clothing in clothings:
                    self.clothing_freq_dict[clothing["name"]] = self.clothing_freq_dict.get(clothing["name"], 0) + 1
                    name = clothing.get("name")
                    colors = clothing.get("color", [])
                    for color in colors:
                        if color not in self.clothing_color_name_2_picture_dict:
                            self.clothing_color_name_2_picture_dict[color] = {}
                        if name not in self.clothing_color_name_2_picture_dict[color]:
                            self.clothing_color_name_2_picture_dict[color][name] = []
                        self.clothing_color_name_2_picture_dict[color][name].append(pic)

                        if name not in self.clothing_name_color_2_picture_dict:
                            self.clothing_name_color_2_picture_dict[name] = {}
                        if color not in self.clothing_name_color_2_picture_dict[name]:
                            self.clothing_name_color_2_picture_dict[name][color] = []
                        self.clothing_name_color_2_picture_dict[name][color].append(pic)
        print(f"All {len(self.clothing_name_color_2_picture_dict)} clothing name-color mappings constructed: {', '.join(list(self.clothing_name_color_2_picture_dict.keys()))}")
        print(f"All {len(self.clothing_color_name_2_picture_dict)} clothing color-name mappings constructed: {', '.join(list(self.clothing_color_name_2_picture_dict.keys()))}")
        self._construct_synonym_dict(list(self.clothing_name_color_2_picture_dict.keys()), list(self.clothing_color_name_2_picture_dict.keys()))

    def _construct_synonym_dict(self, name_list, color_list):
        """构建同义词词典，使用16个并发线程，支持增量更新"""
        # 创建线程锁保护共享资源
        lock = threading.Lock()
        cnt = 0
        
        # 读取已有的同义词字典文件
        existing_synonyms = {}
        existing_distinguishable = {}
        if os.path.exists("clothing_synonym_dict.json"):
            try:
                with open("clothing_synonym_dict.json", "r") as f:
                    existing_data = json.load(f)
                    existing_synonyms = existing_data.get("synonyms", {})
                    existing_distinguishable = existing_data.get("distinguishable", {})
                print(f"Loaded existing synonym dictionary with {len(existing_synonyms)} entries.")
            except (json.JSONDecodeError, FileNotFoundError):
                print("Could not load existing synonym dictionary, starting fresh.")
        
        # 合并已有数据到当前实例
        self.synonym_dict.update(existing_synonyms)
        self.distinguishable_dict.update(existing_distinguishable)
        
        def process_name_combination(combo):
            name1, name2 = combo
            wording_overlap = False
            for word in name1.split():
                if word in name2:
                    wording_overlap = True
                    break
            for word in name2.split():
                if word in name1:
                    wording_overlap = True
                    break
            if not wording_overlap:
                return (name1, name2, False)
            ans = ask_question(f"'{name1}' and '{name2}' are words discribing two wearable items. Please analyze their meanings and decide if they are looking alike, of same meaning, or one of them belong to the other. At the end of your answer, please put 'yes' if they are some kind of synonymous as said or 'no' if they are not.")
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
        
        def process_color_combination(combo):
            color1, color2 = combo
            wording_overlap = False
            for word in color1.split():
                if word in color2:
                    wording_overlap = True
                    break
            for word in color2.split():
                if word in color1:
                    wording_overlap = True
                    break
            if not wording_overlap:
                return (color1, color2, False)
            ans = ask_question(f"'{color1}' and '{color2}' are words discribing two color types of some wearings. Please analyze their meanings and decide if they are looking alike, of same meaning, possibly hard to distinguish, or one of them belong to the other. At the end of your answer, please put 'yes' if they are this kind of similar color pattern or 'no' if they are not.")
            print(ans)
            yes_idx = ans[::-1].lower().find("yes"[::-1])
            no_idx = ans[::-1].lower().find("no"[::-1])
            if yes_idx == -1:
                yes_idx = float('inf')
            if no_idx == -1:
                no_idx = float('inf')
            if yes_idx < no_idx:
                print(ans)
                print(yes_idx, no_idx)
            return (color1, color2, yes_idx < no_idx)
        
        def combination_already_processed(item1, item2):
            """检查组合是否已经处理过"""
            # 检查是否在同义词字典中
            if item1 in existing_synonyms and item2 in existing_synonyms:
                return True
            if item1 in existing_distinguishable and item2 in existing_distinguishable:
                return True
            return False
        
        # 初始化字典
        for name in name_list:
            if name not in self.synonym_dict:
                self.synonym_dict[name] = []
            if name not in self.distinguishable_dict:
                self.distinguishable_dict[name] = []
        
        for color in color_list:
            if color not in self.synonym_dict:
                self.synonym_dict[color] = []
            if color not in self.distinguishable_dict:
                self.distinguishable_dict[color] = []
        
        # 处理名称组合，跳过已处理的组合
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
                        else:
                            self.distinguishable_dict[name1].append(name2)
                            self.distinguishable_dict[name2].append(name1)
                        
                        cnt += 1
                        if cnt % 2000 == 0:
                            print(f"Processed {cnt}/{total_name_combinations} name combinations so far.")
                            with open("clothing_synonym_dict.json", "w") as f:
                                json.dump({
                                    "synonyms": self.synonym_dict,
                                    "distinguishable": self.distinguishable_dict
                                }, f)
        
        # 处理颜色组合，跳过已处理的组合
        color_combinations = [combo for combo in itertools.combinations(color_list, 2) 
                            if not combination_already_processed(combo[0], combo[1])]
        total_color_combinations = len(color_combinations)
        processed_colors = 0
        
        print(f"Found {total_color_combinations} new color combinations to process.")
        
        if total_color_combinations > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                future_to_combo = {executor.submit(process_color_combination, combo): combo for combo in color_combinations}
                
                for future in concurrent.futures.as_completed(future_to_combo):
                    color1, color2, is_synonym = future.result()
                    
                    with lock:
                        if is_synonym:
                            self.synonym_dict[color1].append(color2)
                            self.synonym_dict[color2].append(color1)
                        else:
                            self.distinguishable_dict[color1].append(color2)
                            self.distinguishable_dict[color2].append(color1)
                        
                        processed_colors += 1
                        if processed_colors % 100 == 0:  # 更频繁地保存，避免丢失进度
                            print(f"Processed {processed_colors}/{total_color_combinations} color combinations so far.")
                            with open("clothing_synonym_dict.json", "w") as f:
                                json.dump({
                                    "synonyms": self.synonym_dict,
                                    "distinguishable": self.distinguishable_dict
                                }, f)
        
        # 最终保存
        with open("clothing_synonym_dict.json", "w") as f:
            json.dump({
                "synonyms": self.synonym_dict,
                "distinguishable": self.distinguishable_dict
            }, f)
        
        print(f"Completed processing {total_name_combinations} new name combinations and {total_color_combinations} new color combinations.")
        print(f"Total entries in synonym dictionary: {len(self.synonym_dict)}")
        print(f"Total entries in distinguishable dictionary: {len(self.distinguishable_dict)}")


    

    def generate_questions(self):
        """生成多图片多人物服饰特征相关的问题"""
        questions = []

        def find_image_partial_clothing(clothing_list, fit_count):
            """找出恰好满足clothing_list中fit_count个服饰的图片-服饰对"""
            clothing_word_buckets = [set(self.synonym_dict.get(clothing['name'], []) + [clothing['name']]) for clothing in clothing_list]
            color_word_buckets = []
            for clothing in clothing_list:
                color_bucket = set(clothing.get('color', []))
                for color in clothing.get('color', []):
                    color_bucket.update(self.synonym_dict.get(color, []) + [color])
                color_word_buckets.append(color_bucket)
            image_clothing_list = []
            for picture in self.dataset_pictures:
                matched_clothing_results = []
                for person in picture.persons:
                    matched_clothing = []
                    person_clothings = person.get_clothing_list(only_confident=False)
                    for clothing_bucket, color_bucket in zip(clothing_word_buckets, color_word_buckets):
                        for clothing in person_clothings:
                            if clothing['name'] in clothing_bucket and any(color in clothing.get('color', []) for color in color_bucket):
                                matched_clothing.append(clothing)
                    if len(matched_clothing) >= fit_count:
                        matched_clothing_results.append(matched_clothing)
                
                # 只有一人刚好达到fit_count
                if len(matched_clothing_results) == 1 and len(matched_clothing_results[0]) == fit_count:
                    image_clothing_list.append((picture, matched_clothing_results[0]))
                    self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
            return image_clothing_list
        
        def clothing_color_match_score(picture, colors: set[str]):
            color_count = 0
            for person in picture.persons:
                person_colors = set()
                for clothing in person.get_clothing_list(only_confident=False):
                    person_colors.update(clothing.get('color', []))
                color_count += len(person_colors & colors)
            return color_count

        first_image_clothing_list = []
        second_image_clothing_list_list = []
        third_image_clothing_list_list = []
        fourth_image_clothing_list_list = []
        
        # 先确定第一张图片：有占比超过20%的人物，且有超过三件服饰
        total_pictures = len(self.dataset_pictures)
        processed_pictures = 0
        print(f"开始处理 {total_pictures} 张图片，寻找符合条件的服装组合...")
        
        for picture in self.dataset_pictures:
            for person in picture.persons:
                # 对于每一个穿着超过三件服饰的人物，记录下他们穿着的最独特的三件服饰，并配合图片
                if person.body_area() > 0.2 and len(person.get_clothing_list(only_confident=True)) > 3:
                    top_clothings = sorted(person.get_clothing_list(only_confident=True), key=lambda x: self.clothing_freq_dict[x['name']])[:3]
                    first_image_clothing_list.append((picture, top_clothings))
                    color_appeared = set()
                    for clothing in top_clothings:
                        for color in clothing.get('color', []):
                            color_appeared.update(self.synonym_dict.get(color, []) + [color])
                    self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
                    # 再找出部分符合的图片
                    partial_clothing = find_image_partial_clothing(top_clothings, fit_count=2)
                    if len(partial_clothing) > 10:
                        # 太多的话，先找颜色最符合的图片
                        partial_clothing = sorted(partial_clothing, key=lambda x: -clothing_color_match_score(x[0], color_appeared))[:10]
                    # 以及更欠符合的图片
                    less_fitting_clothing = find_image_partial_clothing(top_clothings, fit_count=1)
                    if len(less_fitting_clothing) > 10:
                        less_fitting_clothing = sorted(less_fitting_clothing, key=lambda x: -clothing_color_match_score(x[0], color_appeared))[:10]
                    # 以及最不符合的图片
                    least_fitting_clothing = find_image_partial_clothing(top_clothings, fit_count=0)
                    if len(least_fitting_clothing) > 10:
                        least_fitting_clothing = sorted(least_fitting_clothing, key=lambda x: -clothing_color_match_score(x[0], color_appeared))[:10]
                    second_image_clothing_list_list.append(partial_clothing)
                    third_image_clothing_list_list.append(less_fitting_clothing)
                    fourth_image_clothing_list_list.append(least_fitting_clothing)
            
            processed_pictures += 1
            if processed_pictures % 100 == 0 or processed_pictures == total_pictures:
                print(f"已处理 {processed_pictures}/{total_pictures} 张图片，找到 {len(first_image_clothing_list)} 个符合条件的服装组合")

        print(f"图片处理完成！共找到 {len(first_image_clothing_list)} 个符合条件的服装组合")
        print("开始生成问题...")
        
        questions = []
        total_combinations = len(first_image_clothing_list)
        for idx, (first_image, second_image_list, third_image_list, fourth_image_list) in enumerate(zip(first_image_clothing_list, second_image_clothing_list_list, third_image_clothing_list_list, fourth_image_clothing_list_list)):
            # 找self.clothing_freq_dict出现频率最少的图片
            second_image = min(second_image_list, key=lambda x: self.clothing_freq_dict.get(x[0], 0), default=None)
            third_image = min(third_image_list, key=lambda x: self.clothing_freq_dict.get(x[0], 0), default=None)
            fourth_image = min(fourth_image_list, key=lambda x: self.clothing_freq_dict.get(x[0], 0), default=None)
            if second_image is None or third_image is None or fourth_image is None:
                continue
            types = []
            for clothing in first_image[1]:
                types.append(clothing['type'])
            sorted_types = sorted(types)
            questions.append(
                {
                    "type": "multiclothing",
                    "combine": first_image[1],
                    "fullfit": first_image[0].image_path(),
                    "duo": second_image[0].image_path(),
                    "duo_admit": second_image[1],
                    "solo": third_image[0].image_path(),
                    "solo_admit": third_image[1],
                    "none": fourth_image[0].image_path(),
                    "distinct": [f"multiclothing-{sorted_types[0]}-{sorted_types[1]}-{sorted_types[2]}"],
                }
            )
            
            if (idx + 1) % 50 == 0 or (idx + 1) == total_combinations:
                print(f"已生成 {idx + 1}/{total_combinations} 个问题")

        print(f"问题生成完成！共生成 {len(questions)} 个多图服装特征问题")
        return questions

    def filter_pictures(self):
        """过滤符合条件的图片"""
        filtered_pictures = []
        for picture in self.dataset_pictures:
            # 需要人体占比超过20%
            body_area_sum = sum(person.body_area() for person in picture.persons)
            # 需要至少有一人有一件服饰
            has_clothing = any(len(person.get_clothing_list()) > 0 for person in picture.persons)
            if body_area_sum > 0.2 and has_clothing:
                filtered_pictures.append(picture)
            else:
                filtered_pictures.append(picture)
        print(f"Filtered down to {len(filtered_pictures)} records for multi-person clothing feature questions.")
        self.dataset_pictures = filtered_pictures
        self._construct_clothing_dict(filtered_pictures)
        return filtered_pictures
        
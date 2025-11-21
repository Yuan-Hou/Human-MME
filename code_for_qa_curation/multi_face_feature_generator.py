import json
import itertools
from rich.progress import track
from test_framework import QuestionGenerator, FACE_ATTR_NAMES_EXTEND

class MultiFaceFeatureQuestionGenerator(QuestionGenerator):
    """多图人脸特征题型生成器"""
    
    def filter_pictures(self):
        """过滤符合条件的图片"""
        # 找出其中单张面部框占屏幕比例超过3%的、没有其他foreground的人脸、face_seen为true、人没有被删除、不是no_face
        filtered_pictures = []
        for picture in self.dataset_pictures:
            # 需要有占比超过3%的面部框
            if not any(person.face_area() > 0.03 for person in picture.persons):
                continue
            # 占比低于3%的面部框里面，不能有face_seen并且不是background的
            if any(person.face_area() < 0.03 and (person.detailing_property("face_seen", True) and not person.detailing_property("background", False)) for person in picture.persons):
                continue
            filtered_pictures.append(picture)
        
        print(f"Filtered down to {len(filtered_pictures)} records after applying face feature criteria.")
        return filtered_pictures
    
    def _process_attribute_combinations(self, filtered_pictures):
        """处理人脸属性组合"""
        combine_domains = {}
        cnt = 0
        
        # 遍历FACE_ATTR_NAMES中任意三个特征组成的三元组，添加rich进度条
        for combo in track(itertools.combinations(FACE_ATTR_NAMES_EXTEND, 3), description="Processing face attribute combinations..."):
            # 过滤符合条件的图片:图中存在一人符合三元组
            fullfit_filtered = self._find_fullfit_pictures(filtered_pictures, combo)
            
            if len(fullfit_filtered) == 0:
                continue

            # 过滤符合条件的图片:图中存在一人（a）仅符合三元组中任意两个条件
            duo_filtered = self._find_duo_pictures(filtered_pictures, combo)
            
            # 过滤符合条件的图片：图中存在一人（a）仅符合三元组中任意一个条件
            solo_filtered = self._find_solo_pictures(filtered_pictures, combo)
            
            # 过滤符合条件的图片：图中所有人都否定了所有的属性
            none_filtered = self._find_none_pictures(filtered_pictures, combo)

            if len(fullfit_filtered) + len(duo_filtered) + len(solo_filtered) + len(none_filtered) == 0:
                continue

            combine_domains[frozenset(combo)] = {
                "type": "multiface",
                "fullfit": fullfit_filtered,
                "duo": duo_filtered,
                "solo": solo_filtered,
                "none": none_filtered,
                "distinct": [f"multiface-{attr}" for attr in combo]
            }
            cnt += 1
            if cnt % 1000 == 0:
                print(f"Processed {cnt} combinations so far.")
        
        return combine_domains
    
    def _find_fullfit_pictures(self, filtered_pictures, combo):
        """找出完全符合三元组属性的图片"""
        fullfit_filtered = []
        for picture in filtered_pictures:
            found = False
            for person in picture.persons:
                # 只考虑有面部框且面部区域占比超过3%的人
                if person.face_box is not None and person.face_area() > 0.03:
                    # 检查该人是否同时具备三元组中的所有属性
                    has_all_attrs = True
                    for attr in combo:
                        if attr not in person.get_face_attr_admit_list():
                            has_all_attrs = False
                            break
                    if has_all_attrs:
                        found = True
                        break
            if found:
                fullfit_filtered.append(picture)
                self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
        return fullfit_filtered
    
    def _find_duo_pictures(self, filtered_pictures, combo):
        """找出符合两个属性、否定一个属性的图片"""
        duo_filtered = []
        for picture in filtered_pictures:
            found = False
            for person in picture.persons:
                if person.face_box is not None and person.face_area() > 0.03:
                    # 检查该人是否同时具备三元组中的任意两个属性，并确定要否定的属性
                    admit_count = 0
                    admit_subset = set()
                    deny_attr = None
                    for attr in combo:
                        if attr in person.get_face_attr_admit_list():
                            admit_count += 1
                            admit_subset.add(attr)
                        elif attr in person.get_face_attr_deny_list():
                            deny_attr = attr
                    if admit_count == 2 and deny_attr is not None:
                        # 检查其他人是否都否定了deny_attr
                        if all(deny_attr not in other_person.get_face_attr_admit_list() 
                               for other_person in picture.persons 
                               if (other_person != person and other_person.face_box is not None and other_person.face_area() > 0.03)):
                            found = True
                            break
            if found:
                duo_filtered.append((picture, admit_subset, set([deny_attr])))
                self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
        return duo_filtered
    
    def _find_solo_pictures(self, filtered_pictures, combo):
        """找出符合一个属性、否定两个属性的图片"""
        solo_filtered = []
        for picture in filtered_pictures:
            found = False
            for person in picture.persons:
                if person.face_box is not None and person.face_area() > 0.03:
                    # 检查该人是否同时具备三元组中的任意一个属性，并确定要否定的属性
                    admit_count = 0
                    admit_attr = None
                    deny_attrs = set()
                    for attr in combo:
                        if attr in person.get_face_attr_admit_list():
                            admit_count += 1
                            admit_attr = attr
                        elif attr in person.get_face_attr_deny_list():
                            deny_attrs.add(attr)
                    if admit_count == 1 and (len(deny_attrs) == 2):
                        # 检查其他人是否都否定了deny_attr
                        if all(other_person.get_face_attr_deny_list().issuperset(deny_attrs)
                               for other_person in picture.persons 
                               if (other_person != person and other_person.face_box is not None and other_person.face_area() > 0.03)):
                            found = True
                            break
            if found:
                solo_filtered.append((picture, set([admit_attr]), deny_attrs))
                self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
        return solo_filtered
    
    def _find_none_pictures(self, filtered_pictures, combo):
        """找出所有人都否定所有属性的图片"""
        none_filtered = []
        deny_attrs = set(combo)
        for picture in filtered_pictures:
            if all(other_person.get_face_attr_deny_list().issuperset(deny_attrs)
                   for other_person in picture.persons
                   if (other_person.face_box is not None and other_person.face_area() > 0.03)):
                none_filtered.append(picture)
                self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
        return none_filtered
    
    def _calculate_penalty(self, **kwargs):
        """计算图片的惩罚值"""
        confidence = 0
        picture = kwargs["picture"]
        admit_attrs = kwargs.get("admit_attrs", set())
        deny_attrs = kwargs.get("deny_attrs", set())
        for person in picture.persons:
            person_confidence = person.get_face_attr_assert_belief(admit_attrs, deny_attrs)
            if person_confidence > confidence:
                confidence = person_confidence
        occurrence = self.picture_occurrence.get(picture, 0)
        return occurrence * (1 - confidence)
    
    def generate_questions(self):
        """生成多图人脸特征题目"""
        filtered_pictures = self.filter_pictures()
        combine_domains = self._process_attribute_combinations(filtered_pictures)
        
        # 取得出题用的数据，准备往模板里填充
        questions = []
        cnt = 0
        for combine, domain in combine_domains.items():
            try:
                fullfit_pictures = domain["fullfit"]
                duo_pictures = domain["duo"]
                solo_pictures = domain["solo"]
                none_pictures = domain["none"]
                
                if len(fullfit_pictures) > 10:
                    fullfit_pictures = sorted(fullfit_pictures, key=lambda pic: self._calculate_penalty(picture=pic, admit_attrs=combine), reverse=False)[:10]
                if len(duo_pictures) > len(fullfit_pictures):
                    duo_pictures = sorted(duo_pictures, key=lambda item: self._calculate_penalty(picture=item[0], admit_attrs=item[1], deny_attrs=item[2]), reverse=False)[:len(fullfit_pictures)]
                else:
                    duo_pictures = duo_pictures * (len(fullfit_pictures) // len(duo_pictures) + 1)
                    duo_pictures = duo_pictures[:len(fullfit_pictures)]
                if len(solo_pictures) > 10:
                    solo_pictures = sorted(solo_pictures, key=lambda item: self._calculate_penalty(picture=item[0], admit_attrs=item[1], deny_attrs=item[2]), reverse=False)[:10]
                else:
                    solo_pictures = solo_pictures * (len(fullfit_pictures) // len(solo_pictures) + 1)
                    solo_pictures = solo_pictures[:len(fullfit_pictures)]
                if len(none_pictures) > 10:
                    none_pictures = sorted(none_pictures, key=lambda pic: self._calculate_penalty(picture=pic, deny_attrs=combine), reverse=False)[:10]
                else:
                    none_pictures = none_pictures * (len(fullfit_pictures) // len(none_pictures) + 1)
                    none_pictures = none_pictures[:len(fullfit_pictures)]

                for fullfit_pic, duo_pic, solo_pic, none_pic in zip(fullfit_pictures, duo_pictures, solo_pictures, none_pictures):
                    question = {
                        "type": "multiface",
                        "combine": list(combine),
                        "fullfit": fullfit_pic.image_path(),
                        "duo": duo_pic[0].image_path(),
                        "duo_admit": list(duo_pic[1]),
                        "solo": solo_pic[0].image_path(),
                        "solo_admit": list(solo_pic[1]),
                        "none": none_pic.image_path(),
                        "distinct": [f"multiface-{attr}" for attr in combine]
                    }
                    questions.append(question)
            except Exception as e:
                print(f"Error generating question for combination {combine}: {e}")
            cnt += 1
            if cnt % 2000 == 0:
                with open("questions_partial.json", "w") as f:
                    json.dump(questions, f)
                print(f"Generated {len(questions)} questions so far.")
        
        return questions

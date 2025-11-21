import json
import itertools
from rich.progress import track
from test_framework import QuestionGenerator, FACE_ATTR_NAMES_EXTEND

class MultiFaceFeatureQuestionGenerator(QuestionGenerator):
    """..........."""
    
    def filter_pictures(self):
        """........."""
        # ................3%.、....foreground...、face_seen.true、......、..no_face
        filtered_pictures = []
        for picture in self.dataset_pictures:
            # .......3%....
            if not any(person.face_area() > 0.03 for person in picture.persons):
                continue
            # ....3%......，...face_seen....background.
            if any(person.face_area() < 0.03 and (person.detailing_property("face_seen", True) and not person.detailing_property("background", False)) for person in picture.persons):
                continue
            filtered_pictures.append(picture)
        
        print(f"Filtered down to {len(filtered_pictures)} records after applying face feature criteria.")
        return filtered_pictures
    
    def _process_attribute_combinations(self, filtered_pictures):
        """........"""
        combine_domains = {}
        cnt = 0
        
        # ..FACE_ATTR_NAMES.............，..rich...
        for combo in track(itertools.combinations(FACE_ATTR_NAMES_EXTEND, 3), description="Processing face attribute combinations..."):
            # .........:...........
            fullfit_filtered = self._find_fullfit_pictures(filtered_pictures, combo)
            
            if len(fullfit_filtered) == 0:
                continue

            # .........:......（a）.............
            duo_filtered = self._find_duo_pictures(filtered_pictures, combo)
            
            # .........：......（a）.............
            solo_filtered = self._find_solo_pictures(filtered_pictures, combo)
            
            # .........：..............
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
        """.............."""
        fullfit_filtered = []
        for picture in filtered_pictures:
            found = False
            for person in picture.persons:
                # ................3%..
                if person.face_box is not None and person.face_area() > 0.03:
                    # ...................
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
        """........、........."""
        duo_filtered = []
        for picture in filtered_pictures:
            found = False
            for person in picture.persons:
                if person.face_box is not None and person.face_area() > 0.03:
                    # .....................，.........
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
                        # ...........deny_attr
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
        """........、........."""
        solo_filtered = []
        for picture in filtered_pictures:
            found = False
            for person in picture.persons:
                if person.face_box is not None and person.face_area() > 0.03:
                    # .....................，.........
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
                        # ...........deny_attr
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
        """..............."""
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
        """........"""
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
        """.........."""
        filtered_pictures = self.filter_pictures()
        combine_domains = self._process_attribute_combinations(filtered_pictures)
        
        # ........，........
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

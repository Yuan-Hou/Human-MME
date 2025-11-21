import json
import itertools
import os
import pickle
from typing import List, Dict, Any
from rich.progress import track
import torch
import clip
import faiss
from PIL import Image
import numpy as np
from test_framework import Picture, QuestionGenerator, FACE_ATTR_NAMES
import random

class AbstractSingleImageChoiceQuestionGenerator(QuestionGenerator):
    """..........."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ...CLIP..
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # ...FAISS.....
        self.vector_dimension = 512  # CLIP ViT-B/32.....
        self.index = None
        self.features = []  # ......（.....）
        self.picture_map = []  # .......picture..
        self.feature_type = []  # ......（'image' . 'text'）
        
    def _get_db_path(self) -> str:
        """..........."""
        return os.path.join(os.path.dirname(self.dataset_pictures[0].image_path()), "vector_db.pkl")
        
    def _save_vector_database(self):
        """.........."""
        db_path = self._get_db_path()
        
        # ......... Picture ..
        picture_paths = [pic.image_path() for pic in self.picture_map]
        
        # ........
        db_data = {
            'features': np.array(self.features),
            'picture_paths': picture_paths,
            'feature_type': self.feature_type,
        }
        
        # ....
        with open(db_path, 'wb') as f:
            pickle.dump(db_data, f)
            
        # .... FAISS ..
        faiss.write_index(self.index, db_path + '.faiss')
        print(f".........: {db_path}")
        
    def _load_vector_database(self) -> bool:
        """..........
        Returns:
            bool: ......
        """
        db_path = self._get_db_path()
        if not (os.path.exists(db_path) and os.path.exists(db_path + '.faiss')):
            return False
            
        try:
            # .....
            with open(db_path, 'rb') as f:
                db_data = pickle.load(f)
                
            # ........ Picture ..
            path_to_picture = {pic.image_path(): pic for pic in self.filtered_pictures}
            self.picture_map = [path_to_picture[path] for path in db_data['picture_paths']]
            
            # ......
            self.features = db_data['features'].tolist()
            self.feature_type = db_data['feature_type']
            
            # .. FAISS ..
            self.index = faiss.read_index(db_path + '.faiss')
            
            print(f".........: {db_path}")
            return True
        except Exception as e:
            print(f".........: {e}")
            return False
    
    def _truncate_text(self, text, max_words=50):
        """..........，....CLIP........"""
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text

    def build_vector_database(self):
        """.......，..........."""
        for picture in track(self.filtered_pictures, description="............"):
            # ....
            image = Image.open(picture.image_path())
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_feature = self.model.encode_image(image_input)
            image_feature = image_feature.cpu().numpy()
            
            # ......
            scene_text = self._truncate_text(picture.raw_data["scene"])
            try:
                text_input = clip.tokenize([scene_text]).to(self.device)
                with torch.no_grad():
                    text_feature = self.model.encode_text(text_input)
                text_feature = text_feature.cpu().numpy()
                
                # .........
                self.features.extend([image_feature[0], text_feature[0]])
                self.picture_map.extend([picture, picture])
                self.feature_type.extend(['image', 'text'])
            except RuntimeError as e:
                # ........，.......
                print(f"..：.. {picture.image_path()} .....，.......：{scene_text}")
                self.features.append(image_feature[0])
                self.picture_map.append(picture)
                self.feature_type.append('image')
            
        # ........FAISS..
        features_array = np.array(self.features, dtype='float32')
        self.index.add(features_array)
    
    def filter_pictures(self):
        """........."""
        filtered_pictures: List[Picture] = []
        for picture in self.dataset_pictures:
            # ...scene
            if "scene" not in picture.raw_data:
                continue 
            filtered_pictures.append(picture)
        self.filtered_pictures = filtered_pictures
        print(f"...........: {len(filtered_pictures)}")
        
        # ............
        if not self._load_vector_database():
            print("...........，............")
            self.index = faiss.IndexFlatL2(self.vector_dimension)
            self.build_vector_database()
            self._save_vector_database()
            
        return filtered_pictures
    
    def search_similar(self, query_feature, k=5, filter_type=None):
        """......
        Args:
            query_feature: ......
            k: ..........
            filter_type: ..，......... ('image' . 'text')
        Returns:
            pictures: ...picture....
            types: .........
            distances: .......
        """
        # ......k...
        distances, indices = self.index.search(query_feature.reshape(1, -1).astype('float32'), k)
        
        # .....picture.......
        pictures = [self.picture_map[i] for i in indices[0]]
        types = [self.feature_type[i] for i in indices[0]]
        
        # .........，....
        if filter_type:
            filtered_results = [(p, t, d) for p, t, d in zip(pictures, types, distances[0]) if t == filter_type]
            if filtered_results:
                pictures, types, distances = zip(*filtered_results)
                return list(pictures), list(types), list(distances)
            return [], [], []
        
        return pictures, types, distances[0].tolist()

    def search_by_image(self, image_path, k=5):
        """.........."""
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image_input)
        return self.search_similar(image_feature.cpu().numpy(), k)
    
    def search_by_text(self, text, k=5):
        """.........."""
        truncated_text = self._truncate_text(text)
        try:
            text_input = clip.tokenize([truncated_text]).to(self.device)
            with torch.no_grad():
                text_feature = self.model.encode_text(text_input)
            return self.search_similar(text_feature.cpu().numpy(), k)
        except RuntimeError as e:
            print(f"..：......：{truncated_text}")
            return [], [], []

    def most_similar_by_text(self, text, batch_size=50):
        """............（.........）
        Args:
            text: .......
            batch_size: .........（...，...........）
        Yields:
            tuple: (picture, distance) .............
        """
        # ....
        truncated_text = self._truncate_text(text)
        try:
            text_input = clip.tokenize([truncated_text]).to(self.device)
            with torch.no_grad():
                text_feature = self.model.encode_text(text_input)
            text_feature = text_feature.cpu().numpy()
        except RuntimeError as e:
            print(f"..：......：{truncated_text}")
            return

        # .........
        distances, indices = self.index.search(text_feature.reshape(1, -1).astype('float32'), len(self.features))
        
        # ..........
        for idx, distance in zip(indices[0], distances[0]):
            if self.feature_type[idx] == 'image':
                yield self.picture_map[idx], distance

    def most_similar_by_image(self, image_path, batch_size=50):
        """............（.........）
        Args:
            image_path: .........
            batch_size: .........
        Yields:
            tuple: (picture, distance) .............
        """
        # ....
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image_input)
        image_feature = image_feature.cpu().numpy()

        # .........
        total_features = len(self.features)
        for start_idx in range(0, total_features, batch_size):
            end_idx = min(start_idx + batch_size, total_features)
            distances, indices = self.index.search(
                image_feature, end_idx - start_idx
            )
            
            # ..........
            for idx, distance in zip(indices[0], distances[0]):
                if self.feature_type[idx] == 'image':
                    yield self.picture_map[idx], distance

    def get_image_similarity(self, picture1, picture2):
        """....picture..........
        Args:
            picture1: ...Picture..
            picture2: ...Picture..
        Returns:
            float: .....。........，...-1.1..
        """
        # ..........
        image1 = Image.open(picture1.image_path())
        image2 = Image.open(picture2.image_path())
        
        image1_input = self.preprocess(image1).unsqueeze(0).to(self.device)
        image2_input = self.preprocess(image2).unsqueeze(0).to(self.device)
        
        # ....
        with torch.no_grad():
            feature1 = self.model.encode_image(image1_input)
            feature2 = self.model.encode_image(image2_input)
            
        # .......
        similarity = torch.nn.functional.cosine_similarity(
            feature1, feature2
        ).item()
        
        return similarity

    def generate_questions(self):        
        # ........，........
        questions = []
        used_pnf = {}  # ........ past_and_future，....
        for picture in track(self.filtered_pictures):
            use_pic = False
            for person in picture.persons:
                if person.detailing_property("meaningful", False):
                    use_pic = True
                    break
            if not use_pic:
                continue

            scene_text = picture.raw_data["scene"]
            similar_scene_pics = []
            complex_emotions = None
            cur_emotion = None  
            past_and_future = None
            other_intentions = None
            for person in picture.persons:
                if (person.detailing_property("emotion") is not None) and (person.detailing_property("emotion") not in ["neutral", "unknown"]) and (cur_emotion != "complex"): # complex ..
                    complex_emotions = []
                    cur_emotion = person.detailing_property("emotion")
                if person.detailing_property("intention_ok", False):
                    other_intentions = []

            # .......，...........，...........0.8，...........，.....，...........，..........0.8，....
            print(f"\n....: {picture.image_path()}")
            print(f"....: {scene_text[:100]}...")
            
            found_count = 0
            for another_pic, distance in self.most_similar_by_text(scene_text):
                if used_pnf.get(another_pic.image_path(), 0) > 20:
                    continue
                found_count += 1
                if another_pic == picture:
                    print(f".......: {another_pic.image_path()}")
                    continue
                
                
                if len(similar_scene_pics) >= 3 and (complex_emotions is not None and len(complex_emotions) < 3) and past_and_future is not None and (other_intentions is None or len(other_intentions) >= 3):
                    skip = True
                    for person in another_pic.persons:
                        if person.detailing_property("emotion") == cur_emotion:
                            skip = False
                            break
                    if skip:
                        continue
                
                sim = self.get_image_similarity(picture, another_pic)
                print(f".... {another_pic.image_path()}, ....: {distance:.4f}, .....: {sim:.4f}")
                
                if sim < 0.8:
                    similar_scene_pics.append(another_pic)
                    print(f".........，....: {len(similar_scene_pics)}")
                    if past_and_future is None and another_pic.raw_data.get("past_scene_ok", None) is not None and another_pic.image_path() not in used_pnf:
                        past_and_future = (another_pic.raw_data["overall_past_clean"], another_pic.raw_data["overall_future_clean"])
                        
                        print(f".. past_and_future: {past_and_future}")
                        
                    for person in another_pic.persons:
                        print(person.detailing_property("emotion"), cur_emotion)
                        if person.detailing_property("emotion") == cur_emotion and complex_emotions is not None:
                            emotion = person.detailing_property("complex_emotion_clean")
                            complex_emotions.append((another_pic.image_path(), emotion))
                            print(f"......: {emotion}, ....: {len(complex_emotions)}")
                        if person.detailing_property("intention_ok", False) and other_intentions is not None:
                            intention = person.detailing_property("intention")
                            other_intentions.append((another_pic.image_path(),intention))
                            print(f"....: {intention}, ....: {len(other_intentions)}")
                        

                if len(similar_scene_pics) >= 3 and (complex_emotions is None or len(complex_emotions) >= 3) and past_and_future is not None and (other_intentions is None or len(other_intentions) >= 3):
                    print(".......，....")
                    break
                    
            if found_count == 0:
                print("..：..........")
            try:
                for person in picture.persons:
                    if person.detailing_property("emotion") == cur_emotion and complex_emotions is not None:
                        complex_emotion = person.detailing_property("complex_emotion_clean")
                        fas = random.sample(list(set(complex_emotions[:min(len(complex_emotions), 10)])), 3)
                        for p,_ in fas:
                            used_pnf[p] = used_pnf.get(p, 0) + 1
                        questions.append(
                            {
                                "type": "emotion",
                                "true_answer": complex_emotion,
                                "false_answers": [fa[1] for fa in fas],
                                "image": picture.image_path(),
                                "distinct": [f"emotion-{cur_emotion}"]
                            }
                        )
                    if person.detailing_property("intention_ok", False):
                        intention = person.detailing_property("intention")
                        fas = random.sample(list(set(other_intentions[:min(len(other_intentions), 10)])), 3)
                        for p,_ in fas:
                            used_pnf[p] = used_pnf.get(p, 0) + 1
                        questions.append(
                            {
                                "type": "intention",
                                "true_answer": intention,
                                "false_answers": [fa[1] for fa in fas],
                                "image": picture.image_path(),
                                "distinct": [f"intention"]
                            }
                        )
                    

                if picture.raw_data.get("past_scene_ok", False) and picture.raw_data.get("future_scene_ok", False):
                    questions.append(
                        {
                            "type": "causal",
                            "true_answer": (picture.raw_data["overall_past_clean"], picture.raw_data["overall_future_clean"]),
                            "false_answers": past_and_future,
                            "image": picture.image_path(),
                            "distinct": [f"causal"]
                        }
                    )
            except:
                pass

            print(f"......: {len(questions)}")

            

        return questions

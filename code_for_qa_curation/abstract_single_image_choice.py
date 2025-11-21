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
    """多图人脸特征题型生成器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # 初始化FAISS索引和存储
        self.vector_dimension = 512  # CLIP ViT-B/32的特征维度
        self.index = None
        self.features = []  # 存储所有特征（图片和文本）
        self.picture_map = []  # 存储特征对应的picture对象
        self.feature_type = []  # 标记特征类型（'image' 或 'text'）
        
    def _get_db_path(self) -> str:
        """获取向量数据库文件路径"""
        return os.path.join(os.path.dirname(self.dataset_pictures[0].image_path()), "vector_db.pkl")
        
    def _save_vector_database(self):
        """保存向量数据库到文件"""
        db_path = self._get_db_path()
        
        # 保存图片路径而不是 Picture 对象
        picture_paths = [pic.image_path() for pic in self.picture_map]
        
        # 构建要保存的数据
        db_data = {
            'features': np.array(self.features),
            'picture_paths': picture_paths,
            'feature_type': self.feature_type,
        }
        
        # 保存数据
        with open(db_path, 'wb') as f:
            pickle.dump(db_data, f)
            
        # 单独保存 FAISS 索引
        faiss.write_index(self.index, db_path + '.faiss')
        print(f"向量数据库已保存到: {db_path}")
        
    def _load_vector_database(self) -> bool:
        """从文件加载向量数据库
        Returns:
            bool: 是否成功加载
        """
        db_path = self._get_db_path()
        if not (os.path.exists(db_path) and os.path.exists(db_path + '.faiss')):
            return False
            
        try:
            # 加载主数据
            with open(db_path, 'rb') as f:
                db_data = pickle.load(f)
                
            # 将图片路径转换回 Picture 对象
            path_to_picture = {pic.image_path(): pic for pic in self.filtered_pictures}
            self.picture_map = [path_to_picture[path] for path in db_data['picture_paths']]
            
            # 加载其他数据
            self.features = db_data['features'].tolist()
            self.feature_type = db_data['feature_type']
            
            # 加载 FAISS 索引
            self.index = faiss.read_index(db_path + '.faiss')
            
            print(f"成功加载向量数据库: {db_path}")
            return True
        except Exception as e:
            print(f"加载向量数据库失败: {e}")
            return False
    
    def _truncate_text(self, text, max_words=50):
        """截断文本至指定单词数，避免超出CLIP的上下文长度限制"""
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text

    def build_vector_database(self):
        """构建向量数据库，同时存储图片和文本特征"""
        for picture in track(self.filtered_pictures, description="正在构建向量数据库..."):
            # 处理图片
            image = Image.open(picture.image_path())
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_feature = self.model.encode_image(image_input)
            image_feature = image_feature.cpu().numpy()
            
            # 处理场景文本
            scene_text = self._truncate_text(picture.raw_data["scene"])
            try:
                text_input = clip.tokenize([scene_text]).to(self.device)
                with torch.no_grad():
                    text_feature = self.model.encode_text(text_input)
                text_feature = text_feature.cpu().numpy()
                
                # 存储特征和对应信息
                self.features.extend([image_feature[0], text_feature[0]])
                self.picture_map.extend([picture, picture])
                self.feature_type.extend(['image', 'text'])
            except RuntimeError as e:
                # 如果文本仍然太长，只存储图片特征
                print(f"警告：图片 {picture.image_path()} 的文本太长，已跳过文本特征：{scene_text}")
                self.features.append(image_feature[0])
                self.picture_map.append(picture)
                self.feature_type.append('image')
            
        # 将所有特征添加到FAISS索引
        features_array = np.array(self.features, dtype='float32')
        self.index.add(features_array)
    
    def filter_pictures(self):
        """过滤符合条件的图片"""
        filtered_pictures: List[Picture] = []
        for picture in self.dataset_pictures:
            # 必须有scene
            if "scene" not in picture.raw_data:
                continue 
            filtered_pictures.append(picture)
        self.filtered_pictures = filtered_pictures
        print(f"找到符合条件的图片数量: {len(filtered_pictures)}")
        
        # 尝试加载现有的向量数据库
        if not self._load_vector_database():
            print("未找到现有的向量数据库，开始构建新的数据库...")
            self.index = faiss.IndexFlatL2(self.vector_dimension)
            self.build_vector_database()
            self._save_vector_database()
            
        return filtered_pictures
    
    def search_similar(self, query_feature, k=5, filter_type=None):
        """搜索相似特征
        Args:
            query_feature: 查询特征向量
            k: 返回的最相似结果数量
            filter_type: 可选，筛选返回结果的类型 ('image' 或 'text')
        Returns:
            pictures: 相似的picture对象列表
            types: 对应的特征类型列表
            distances: 对应的距离列表
        """
        # 搜索最相似的k个特征
        distances, indices = self.index.search(query_feature.reshape(1, -1).astype('float32'), k)
        
        # 获取对应的picture对象和特征类型
        pictures = [self.picture_map[i] for i in indices[0]]
        types = [self.feature_type[i] for i in indices[0]]
        
        # 如果指定了筛选类型，进行过滤
        if filter_type:
            filtered_results = [(p, t, d) for p, t, d in zip(pictures, types, distances[0]) if t == filter_type]
            if filtered_results:
                pictures, types, distances = zip(*filtered_results)
                return list(pictures), list(types), list(distances)
            return [], [], []
        
        return pictures, types, distances[0].tolist()

    def search_by_image(self, image_path, k=5):
        """通过图片搜索相似内容"""
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image_input)
        return self.search_similar(image_feature.cpu().numpy(), k)
    
    def search_by_text(self, text, k=5):
        """通过文本搜索相似内容"""
        truncated_text = self._truncate_text(text)
        try:
            text_input = clip.tokenize([truncated_text]).to(self.device)
            with torch.no_grad():
                text_feature = self.model.encode_text(text_input)
            return self.search_similar(text_feature.cpu().numpy(), k)
        except RuntimeError as e:
            print(f"警告：输入文本太长：{truncated_text}")
            return [], [], []

    def most_similar_by_text(self, text, batch_size=50):
        """按文本相似度迭代所有图片（从最相似到最不相似）
        Args:
            text: 用于比较的文本
            batch_size: 每次获取的批次大小（已弃用，现在一次性搜索所有结果）
        Yields:
            tuple: (picture, distance) 图片对象和对应的相似度距离
        """
        # 编码文本
        truncated_text = self._truncate_text(text)
        try:
            text_input = clip.tokenize([truncated_text]).to(self.device)
            with torch.no_grad():
                text_feature = self.model.encode_text(text_input)
            text_feature = text_feature.cpu().numpy()
        except RuntimeError as e:
            print(f"警告：输入文本太长：{truncated_text}")
            return

        # 一次性搜索所有特征
        distances, indices = self.index.search(text_feature.reshape(1, -1).astype('float32'), len(self.features))
        
        # 只返回图片类型的结果
        for idx, distance in zip(indices[0], distances[0]):
            if self.feature_type[idx] == 'image':
                yield self.picture_map[idx], distance

    def most_similar_by_image(self, image_path, batch_size=50):
        """按图片相似度迭代所有图片（从最相似到最不相似）
        Args:
            image_path: 用于比较的图片路径
            batch_size: 每次获取的批次大小
        Yields:
            tuple: (picture, distance) 图片对象和对应的相似度距离
        """
        # 编码图片
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image_input)
        image_feature = image_feature.cpu().numpy()

        # 分批次搜索所有特征
        total_features = len(self.features)
        for start_idx in range(0, total_features, batch_size):
            end_idx = min(start_idx + batch_size, total_features)
            distances, indices = self.index.search(
                image_feature, end_idx - start_idx
            )
            
            # 只返回图片类型的结果
            for idx, distance in zip(indices[0], distances[0]):
                if self.feature_type[idx] == 'image':
                    yield self.picture_map[idx], distance

    def get_image_similarity(self, picture1, picture2):
        """计算两个picture对象对应图片的相似度
        Args:
            picture1: 第一个Picture对象
            picture2: 第二个Picture对象
        Returns:
            float: 相似度分数。值越大表示越相似，范围在-1到1之间
        """
        # 加载并预处理两张图片
        image1 = Image.open(picture1.image_path())
        image2 = Image.open(picture2.image_path())
        
        image1_input = self.preprocess(image1).unsqueeze(0).to(self.device)
        image2_input = self.preprocess(image2).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            feature1 = self.model.encode_image(image1_input)
            feature2 = self.model.encode_image(image2_input)
            
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            feature1, feature2
        ).item()
        
        return similarity

    def generate_questions(self):        
        # 取得出题用的数据，准备往模板里填充
        questions = []
        used_pnf = {}  # 记录已经使用过的 past_and_future，避免重复
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
                if (person.detailing_property("emotion") is not None) and (person.detailing_property("emotion") not in ["neutral", "unknown"]) and (cur_emotion != "complex"): # complex 优先
                    complex_emotions = []
                    cur_emotion = person.detailing_property("emotion")
                if person.detailing_property("intention_ok", False):
                    other_intentions = []

            # 按照文本相似度，从向量数据库一个一个找，与本张图片相似度要小于0.8，但是文本描述与图片相似，一共找三张，再找具有复杂情感人物的，不要求图片相似度小于0.8，也找三张
            print(f"\n处理图片: {picture.image_path()}")
            print(f"场景文本: {scene_text[:100]}...")
            
            found_count = 0
            for another_pic, distance in self.most_similar_by_text(scene_text):
                if used_pnf.get(another_pic.image_path(), 0) > 20:
                    continue
                found_count += 1
                if another_pic == picture:
                    print(f"跳过同一张图片: {another_pic.image_path()}")
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
                print(f"检查图片 {another_pic.image_path()}, 文本距离: {distance:.4f}, 图片相似度: {sim:.4f}")
                
                if sim < 0.8:
                    similar_scene_pics.append(another_pic)
                    print(f"添加到相似场景列表，当前数量: {len(similar_scene_pics)}")
                    if past_and_future is None and another_pic.raw_data.get("past_scene_ok", None) is not None and another_pic.image_path() not in used_pnf:
                        past_and_future = (another_pic.raw_data["overall_past_clean"], another_pic.raw_data["overall_future_clean"])
                        
                        print(f"设置 past_and_future: {past_and_future}")
                        
                    for person in another_pic.persons:
                        print(person.detailing_property("emotion"), cur_emotion)
                        if person.detailing_property("emotion") == cur_emotion and complex_emotions is not None:
                            emotion = person.detailing_property("complex_emotion_clean")
                            complex_emotions.append((another_pic.image_path(), emotion))
                            print(f"添加复杂情感: {emotion}, 当前数量: {len(complex_emotions)}")
                        if person.detailing_property("intention_ok", False) and other_intentions is not None:
                            intention = person.detailing_property("intention")
                            other_intentions.append((another_pic.image_path(),intention))
                            print(f"添加意图: {intention}, 当前数量: {len(other_intentions)}")
                        

                if len(similar_scene_pics) >= 3 and (complex_emotions is None or len(complex_emotions) >= 3) and past_and_future is not None and (other_intentions is None or len(other_intentions) >= 3):
                    print("已满足所有条件，退出循环")
                    break
                    
            if found_count == 0:
                print("警告：没有找到任何相似图片")
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

            print(f"当前题目总数: {len(questions)}")

            

        return questions

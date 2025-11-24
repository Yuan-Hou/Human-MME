"""
.........
..............
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import os
import json
import asyncio
import sys
import torch
import pickle
import cv2 as cv
import torchvision
# from deepface import DeepFace
import numpy as np
from rich.console import Console
import torchvision.transforms as T
from typing import Dict, Any, Set, List
# from sentence_transformers import SentenceTransformer, util
from torchvision.transforms import InterpolationMode
from utils import ask_about_image, key_points_to_bounding_box, manual_retry, check_retry_status, ask_question, bounding_box_iou, scale_down_image, yes_or_no
from PIL import Image
# Add GroundingDINO to path
# sys.path.append("./repos/GroundingDINO")
# from repos.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

# Add SAM2 to path  
# sys.path.append("./repos/sam2")
# from repos.sam2.sam2.build_sam import build_sam2
# from repos.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

# sys.path.append("./repos/facexformer")
# from inference import FaceXFormer, denorm_points, unnormalize, adjust_bbox, visualize_head_pose

console = Console()

class BaseTask(ABC):
    """....，......"""
    
    def __init__(self, task_name: str, data_dir: str, progress_file_prefix: str):
        self.task_name = task_name
        self.data_dir = data_dir
        self.progress_file_prefix = progress_file_prefix
    
    @abstractmethod
    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............"""
        pass
    
    @abstractmethod
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        """..........."""
        pass
    
    @abstractmethod
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """..JSON......."""
        pass
    
    def get_task_description(self) -> str:
        """......"""
        return f".. {self.task_name} .."
    

class HoiDetectTask(BaseTask):
    """........"""
    def __init__(self):
        super().__init__(task_name="hoi_detect", data_dir="./person_labeling", progress_file_prefix="hoi_detect_progress")
        self.sam2_model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', './repos/sam2/checkpoints/sam2.1_hiera_large.pt')
    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............."""
        if shard_count > 1:
            return f"{self.progress_file_prefix}_shard_{shard_index}_of_{shard_count}.json"
        else:
            return f"{self.progress_file_prefix}.json"
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        """..........hoi.."""
        return data.get('hoi_processed', False)
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # Process each person detected in the image
        image_path = os.path.join(self.data_dir, data['image_path'])
        
        console.print(f"[dim]📷 ....: {data['image_path']}[/dim]")
        image = cv.imread(image_path)
        if image is None:
            raise Exception(f"........: {image_path}")
        H, W, C = image.shape

        sam2_predictor = SAM2ImagePredictor(self.sam2_model, mask_threshold=0.3)
        sam2_predictor.set_image(cv.cvtColor(image.copy(), cv.COLOR_BGR2RGB))
        objs = data['objects']
        body_boxes = data['detect_results']['body_boxes']
        skeletons = data['detect_results']['skeletons']
        obj_masks = []
        pickle_data = {
            "object_masks": obj_masks,
        }
        for obj_info in objs:
            x1, y1, x2, y2 = obj_info['box']
            object_box = (x1*W, y1*H, x2*W, y2*H)
            mask, _, _ = sam2_predictor.predict(
                box=object_box,
                multimask_output=False,
            )
            obj_masks.append(mask)
        mask_file_path = os.path.join(self.data_dir, f"{data['image_path']}_masks.pkl")
        with open(mask_file_path, 'wb') as f:
            pickle.dump(pickle_data, f)
            data['mask_file'] = mask_file_path
        # Analyze person-object relationships
        for idx, obj_info in enumerate(objs):
            candidate_person_objects = []
        
            # Find persons who mentioned this object
            for person in data['persons']:
                if "qwen_detailing" not in person or not person["qwen_detailing"]:
                    continue
                for person_obj in person["qwen_detailing"]["objects"]:
                    if person_obj["name"] in obj_info["possible_names"]:
                        candidate_person_objects.append((person, person_obj))
            
            # Analyze relationship for each candidate
            for candidate_person, person_obj in candidate_person_objects:
                info_img = image.copy()
                # Draw object bounding box in green
                x1, y1, x2, y2 = (obj_info['box'][0] * W, obj_info['box'][1] * H, 
                                obj_info['box'][2] * W, obj_info['box'][3] * H)
                cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw person bounding box in red
                person_body_box = body_boxes[candidate_person['body_box']]
                x1, y1, x2, y2 = (person_body_box[0] * W, person_body_box[1] * H,
                                person_body_box[2] * W, person_body_box[3] * H)
                cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                question = f"""Besides the image, a description of the person in the red bounding box is provided (possibly inaccurate):

{candidate_person['qwen_detailing']['description']}

Please provide information about the relationship between the "{obj_info['name']}" in the green bounding box and the person in the red bounding box in the following JSON format:

```json
{{
    "relevant": true/false, // whether the object is relevant to the person or not, true means relevant, false means not relevant.
    "relationship": {{ // If the "relevant" is false, this field should be null.
        "standalone": true/false, // true if the object has no physical contact with the person
        "position": "the part of the body that holds the object, please only use one of 'one hand', 'both hands', 'head', 'body', 'foot' or 'other'. If the object has no contact with the person, please use 'standalone' instead.",
        "action": ["verb phrase 1", "verb 2", ...] // a list of verbs or verb phrases that describe what the person is doing to the object. Please provide in lower case.
    }}
}}
```
"""
                # json_str = ask_about_image(info_img, question, json_format=True)
                json_str = await asyncio.to_thread(ask_about_image, info_img, question, json_format=True)
                relation_desc = json.loads(json_str)
                
                if not relation_desc["relevant"]:
                    continue

                # Refine hand position if specified as "one hand"
                if relation_desc['relationship']['position'] == "one hand":
                    left_hand = key_points_to_bounding_box(np.asarray(skeletons[candidate_person["skeleton"]]["dw_hand_1"]))
                    right_hand = key_points_to_bounding_box(np.asarray(skeletons[candidate_person["skeleton"]]["dw_hand_2"]))
                    left_hand_x = (left_hand[0] + left_hand[2]) / 2 * W
                    right_hand_x = (right_hand[0] + right_hand[2]) / 2 * W
                    left_hand_y = (left_hand[1] + left_hand[3]) / 2 * H
                    right_hand_y = (right_hand[1] + right_hand[3]) / 2 * H
                    
                    obj_mask = obj_masks[idx]
                    
                    mask_coords = np.argwhere(obj_mask[0] > 0.5)[:, ::-1]  # Get object region coordinates
                    
                    # Calculate distances from hands to object
                    left_hand_distance = np.min(np.sqrt((mask_coords[:, 0] - left_hand_x) ** 2 + 
                                                    (mask_coords[:, 1] - left_hand_y) ** 2))
                    right_hand_distance = np.min(np.sqrt((mask_coords[:, 0] - right_hand_x) ** 2 + 
                                                        (mask_coords[:, 1] - right_hand_y) ** 2))
                    
                    if left_hand_distance < right_hand_distance:
                        relation_desc['relationship']['position'] = "left hand"
                    else:
                        relation_desc['relationship']['position'] = "right hand"
                relation_desc["object"] = idx
                if "hoi" not in candidate_person:
                    candidate_person["hoi"] = []
                candidate_person["hoi"].append(relation_desc)
        data["hoi_processed"] = True
        return data

class ObjectDetectTask(BaseTask):
    """......"""
    def __init__(self):
        super().__init__(task_name="object_detect", data_dir="./person_labeling", progress_file_prefix="object_detect_progress")
        self.dino_model = load_model("repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "repos/GroundingDINO/weights/groundingdino_swint_ogc.pth")
        self.sentence_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2', 
            trust_remote_code=True, 
            local_files_only=True
        )
    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............."""
        if shard_count > 1:
            return f"{self.progress_file_prefix}_shard_{shard_index}_of_{shard_count}.json"
        else:
            return f"{self.progress_file_prefix}.json"
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        """..........object_detect.."""
        objects = data.get('objects', [])
        if not objects:
            return False
        return True
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # Process each person detected in the image
        image_path = os.path.join(self.data_dir, data['image_path'])
        
        console.print(f"[dim]📷 ....: {data['image_path']}[/dim]")
        image = cv.imread(image_path)
        if image is None:
            raise Exception(f"........: {image_path}")
        H, W, C = image.shape

        obj_names = []
        for person in data.get('persons', []):
            if "qwen_detailing" not in person:
                continue
            result = person["qwen_detailing"]
            
            # Collect object names mentioned for this person
            for obj in result.get('objects', []):
                if obj['name'] not in obj_names:
                    obj_names.append(obj['name'])
        
        if len(obj_names) == 0:
            print("No objects detected for any person.")
            data['objects'] = []
            return data
        
        # Deduplicate object names using LLM
        synonyms_map = {}
        if len(obj_names) > 1:
            question = f"""Please remove phrases with the same meaning and keep only one phrase for each meaning in the following list:

```json
{json.dumps(obj_names, indent=4, ensure_ascii=False)}
```

give the result in following JSON format:
```json
{{
    "result": [
        "object1", "object2", ...
    ],
    "synonyms_map": {{
        "object1": ["synonym1", "synonym2", ...], // synonyms of object1, which were removed from the result list
        ...
    }} 
}}
```
"""
            # json_str = ask_question(question, json_format=True)
            json_str = await asyncio.to_thread(ask_question, question, json_format=True)
            obj_names_info = json.loads(json_str)
            obj_names = obj_names_info['result']
            synonyms_map = obj_names_info.get('synonyms_map', {})
        
        # Use GroundingDINO to detect objects in the image
        # image_transformed = scale_down_image(rgb_image, 640)
        _, image_transformed = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image_transformed,
            caption=".".join(obj_names),
            box_threshold=0.35,
            text_threshold=0.35,
            remove_combined=False
        )
        annotated_image = annotate(cv.cvtColor(image.copy(), cv.COLOR_BGR2RGB), boxes, logits, phrases)
        cv.imwrite(os.path.join("./debug_images", data["image_path"]), annotated_image)
        
        # Process detected bounding boxes
        bounding_boxes = []
        detect_names = []
        for box, phrase in zip(boxes.cpu().numpy(), phrases):
            x1, y1, x2, y2 = (box[0] - box[2] / 2), (box[1] - box[3] / 2), (box[2]/2+box[0]), (box[3]/2+box[1])
            bounding_boxes.append((x1, y1, x2, y2, phrase))
            if phrase not in detect_names:
                detect_names.append(phrase)
        
        # Map detected names to original names using similarity
        detect_name_to_original_name = {}
        original_emb = None
        for detect_name in detect_names:
            if detect_name in obj_names:
                detect_name_to_original_name[detect_name] = detect_name
            else:
                detect_emb = self.sentence_model.encode([detect_name], convert_to_tensor=True)
                if original_emb is None:
                    original_emb = self.sentence_model.encode(obj_names, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(detect_emb.cpu(), original_emb.cpu()).squeeze(0)
                max_index = similarity.argmax().item()
                detect_name_to_original_name[detect_name] = obj_names[max_index]
        
        # Group boxes by phrase and merge overlapping ones
        phrase_to_boxes_map = defaultdict(list)
        for bounding_box in bounding_boxes:
            x1, y1, x2, y2, phrase = bounding_box
            # Skip boxes that are too large (likely errors)
            if x2 - x1 > 0.99 and y2 - y1 > 0.99:
                continue
            phrase = detect_name_to_original_name[phrase]
            
            # Merge overlapping boxes with same phrase
            box_index_to_replace = None
            for i, existing_box in enumerate(phrase_to_boxes_map[phrase]):
                if bounding_box_iou((x1, y1, x2, y2), existing_box) > 0.1:
                    x1 = min(x1, existing_box[0])
                    y1 = min(y1, existing_box[1])
                    x2 = max(x2, existing_box[2])
                    y2 = max(y2, existing_box[3])
                    box_index_to_replace = i
                    break
            
            if box_index_to_replace is not None:
                phrase_to_boxes_map[phrase][box_index_to_replace] = (x1, y1, x2, y2)
            else:
                phrase_to_boxes_map[phrase].append((x1, y1, x2, y2))
        
        # Validate detected objects using VLM
        objs = []
        for phrase, boxes in phrase_to_boxes_map.items():
            for box in boxes:
                info_img = image.copy()
                cv.rectangle(info_img, (int(box[0] * W), int(box[1] * H)), (int(box[2] * W), int(box[3] * H)), (0, 255, 0), 1)
                
                question = f"""Please provide information about the object in the green bounding box in following JSON format. The object should be a {phrase}.:

```json
{{
    "accurate": true/false, // whether the object in bounding box is "{phrase}" or not, true means it is.
    "is_clothing": true/false, // whether the object is a clothing or not, true means it is a clothing.
    "is_worn": true/false, // whether the object is worn by a person or not, true means worn, false means not worn. It can only be true if the object is a clothing.
    "is_person": true/false, // whether the object is a person, true means it is.
}}"""
                # json_str = ask_about_image(info_img, question, json_format=True)
                json_str = await asyncio.to_thread(ask_about_image, info_img, question, json_format=True)
                result = json.loads(json_str)
                
                # Skip inaccurate, worn clothing, or person detections
                if (not result.get('accurate', True) or 
                    result.get('is_worn', False) or 
                    result.get('is_person', False)):
                    continue
                    
                possible_names = list(set(synonyms_map.get(phrase, []) + [phrase]))
                objs.append({
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "name": phrase,
                    "possible_names": possible_names
                })
        
        data['objects'] = objs
        return data

class FacexTask(BaseTask):
    """Facexformer........"""
    
    def __init__(self):
        super().__init__(
            task_name="facex_detailing",
            data_dir="./final_labeling",
            progress_file_prefix="facex_detailing_progress"
        )
        device = "cuda"
        self.transforms_image = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224,224), interpolation=InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.model = FaceXFormer().to(device)
        checkpoint_path = "./repos/facexformer/ckpts/model.pt"
        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(self.checkpoint['state_dict_backbone'])
        self.model.eval()

    
    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............."""
        if shard_count > 1:
            return f"{self.progress_file_prefix}_shard_{shard_index}_of_{shard_count}.json"
        else:
            return f"{self.progress_file_prefix}.json"
    
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        """..........facex.."""
        return False
        persons = data.get('persons', [])
        if not persons:
            return True  # ..persons........
        
        # ......person..facex
        for person in persons:
            if person.get('face_box') is not None and ('facex_detailing' not in person or not person['facex_detailing']):
                return False
        return True
    
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """..Facexformer......"""
        console.print(f"[dim]🔍 ......: {os.path.basename(file_path)}[/dim]")
        
        body_boxes = data['detect_results']['body_boxes']
        image_path = os.path.join(self.data_dir, data['image_path'].split("/")[-1])
        
        console.print(f"[dim]📷 ....: {data['image_path']}[/dim]")
        image = Image.open(image_path)
        if image is None:
            raise Exception(f"........: {image_path}")

        W, H = image.size
        # cv_image = cv.imread(image_path)
        
        face_boxes = data['detect_results']['face_boxes']
        for person_idx, person in enumerate(data.get('persons', [])):
            if person['face_box'] is None:
                console.print(f"[dim]⏭️  ...{person_idx+1}.. (....)[/dim]")
                continue
            
            console.print(f"[dim]👤 ...{person_idx+1}.. (face_box..: {person['face_box']})[/dim]")
                
            # Create annotated image with person bounding box
            face_box = face_boxes[person['face_box']]
            x_min, y_min, x_max, y_max = (face_box[0] * W, face_box[1] * H, face_box[2] * W, face_box[3] * H)
            x_min, y_min, x_max, y_max = adjust_bbox(x_min, y_min, x_max, y_max, W, H)
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            image_face = image.copy().crop((x_min, y_min, x_max, y_max))
            # cv_image_face = cv_image[y_min:y_max, x_min:x_max]
            image_face = self.transforms_image(image_face)
            image_face = image_face.unsqueeze(0).to("cuda")
            landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output = self.model(image_face, None, None)

            xface_detailing = {
                "landmarks": [],
                "visibility": [],
                "headpose": {},
                "attributes": {},
                "age": [],
                "race": [],
                "gender": []
            }

            denorm_landmarks = denorm_points(landmark_output.view(-1,68,2)[0],224,224)
            for landmark in denorm_landmarks[0]:
                x, y = landmark[0].item(), landmark[1].item()
                x /= 224
                y /= 224
                x *= x_max - x_min
                y *= y_max - y_min
                x += x_min
                y += y_min
                x /= W
                y /= H
                xface_detailing["landmarks"].append((x, y))

            xface_detailing["headpose"] = {
                "pitch": headpose_output[0][0].item()*180/np.pi,
                "yaw": headpose_output[0][1].item()*180/np.pi,
                "roll": headpose_output[0][2].item()*180/np.pi
            }

            # im = visualize_head_pose(cv_image_face, headpose_output[0])
            # cv.imwrite(os.path.join("./temp", f"headpose_{os.path.basename(file_path).split('.')[0]}_person_{person_idx+1}.jpg"), im)

            probs = torch.sigmoid(attribute_output[0])
            features = [
                "5 oClock Shadow",
                "Arched Eyebrows",
                "Attractive",
                "Bags Under Eyes",
                "Bald",
                "Bangs",
                "Big Lips",
                "Big Nose",
                "Black Hair",
                "Blond Hair",
                "Blurry",
                "Brown Hair",
                "Bushy Eyebrows",
                "Chubby",
                "Double Chin",
                "Eyeglasses",
                "Goatee",
                "Gray Hair",
                "Heavy Makeup",
                "High Cheekbones",
                "Male",
                "Mouth Slightly Open",
                "Mustache",
                "Narrow Eyes",
                "No Beard",
                "Oval Face",
                "Pale Skin",
                "Pointy Nose",
                "Receding Hairline",
                "Rosy Cheeks",
                "Sideburns",
                "Smiling",
                "Straight Hair",
                "Wavy Hair",
                "Wearing Earrings",
                "Wearing Hat",
                "Wearing Lipstick",
                "Wearing Necklace",
                "Wearing Necktie",
                "Young"
            ]

            xface_detailing["attributes"] = {feat: prob.item() for feat, prob in zip(features, probs)}

            probs = torch.sigmoid(visibility_output[0])
            for prob in probs:
                xface_detailing["visibility"].append(prob.item())

            probs = torch.sigmoid(age_output[0])
            for prob in probs:
                xface_detailing["age"].append(prob.item())

            probs = torch.sigmoid(race_output[0])
            for prob in probs:
                xface_detailing["race"].append(prob.item())

            probs = torch.sigmoid(gender_output[0])
            for prob in probs:
                xface_detailing["gender"].append(prob.item())

            person["facex_detailing"] = xface_detailing

            # analyzes = DeepFace.analyze(cv_image_face, detector_backend="yolov8", enforce_detection=False, silent=True)
            # person["deepface_detailing"] = {}
            # max_hoi = -1
            # x_min, y_min, x_max, y_max = (face_box[0] * W, face_box[1] * H, face_box[2] * W, face_box[3] * H)
            # for analyze in analyzes:
            #     nx1, ny1, nx2, ny2 = analyze["region"]["x"], analyze["region"]["y"], analyze["region"]["x"] + analyze["region"]["w"], analyze["region"]["y"] + analyze["region"]["h"],
            #     hoi = bounding_box_iou((x_min, y_min, x_max, y_max), (nx1, ny1, nx2, ny2))
            #     if hoi > max_hoi:
            #         max_hoi = hoi
            #         person["deepface_detailing"] = analyze
            # if len(analyzes) > 1:
            #     console.print(f"[dim]⚠️  ..: DeepFace.......，....facex_box........ (...: {max_hoi:.2f})[/dim]")
            # elif len(analyzes) == 0:
            #     console.print(f"[dim]⚠️  ..: DeepFace......，..................[/dim]")
        return data


class ColorRemovalTask(BaseTask):
    """Qwen........"""
    
    def __init__(self):
        super().__init__(
            task_name="color_removal",
            data_dir="./final_labeling/final_labeling",
            progress_file_prefix="color_removal_progress"
        )
        self.color_words = set()
        for ws in "dark blue, black, blue, dark gray, gray, red, white, yellow, green, light pink, pink, light gray, brown, light beige, maroon, dark red, purple, dark brown, light blue, dark, gold, beige, dark green, orange, mustard yellow, clear, lavender, light yellow, silver, cream, soft cream, light-colored, light brown, off-white, light pastel, pearl, golden, metallic, teal, translucent, patterned, checkered, floral, grey, olive green, olive-green, light orange, peach, light green, light olive-green, light, neutral, bright pink, colorful, light purple, dark purple, denim, various, tie-dye, plaid, none, bare skin, multicolor, stained, camouflage, dark olive, dark olive green, tan, floral pattern, light peach, rainbow, various colors, light olive green, light olive, soft gray, dark polka dots, light cream, navy blue, rust, dark maroon, mauve, neon green, neon yellow-green, silver-grey, pale pink, khaki, transparent, muted brown, skin tone, dark navy blue, dark navy, mustard, light color, burgundy, dark teal, bright blue, deep purple, turquoise, bright green, tinted, light teal, teardrop, embroidered, glittery, pastel, diamond, deep red, lime green, multicolored, coral, neon yellow, grayish, pale, metallic gray, nude, pale blue, unknown, sheer, pale yellow, striped, emerald green, white gold, straw, mint green, light mint green, iridescent, reflective, blonde, tortoiseshell, orange-red, glossy, shiny, magenta, terracotta orange, burnt orange, reddish, soft pink, soft brown, semi-sheer, neon pink, dirty, pastel pink, light grayish-green, pinkish, floral patterns, pinkish-brown, soft white, pearl white, decorative, dark grey, leopard print, muted purple, bright turquoise, dusty pink, denim blue, bright red, metallic silver, grayish blue, red-brown, sparkling, amber, pinkish-red, camel, bright yellow-green, pastel purple, pastel blue, pastel green, light grey, olive, vibrant, yellow-green, gradient, dark stains, bright yellow, dark accent colors, wet, patterned with dark and light shades, polka dot, dark color, crimson, black and white, dark pattern, .., .., greyish-green, navy, cyan, greenish-brown, multi, checkered pattern of blue and white, blue denim, light tan, light salmon, multi-colored, dark-colored, dark spots, metal, teal blue, greenish-blue, dark stripes, faded, red-orange, light red, light khaki, pastel colors, plaid pattern, light floral pattern, lemon, various (seems to have beads of different colors), multi-color, semi-transparent, red and white, blue and white striped, various colors from the graphic, white stripes, multi-colored pattern, neon blue, faded blue, hunter green, animal print, dark colors, plaid with grey, white, and black, camouflage pattern, multicolored (rainbow pattern), plaid with blue, red, yellow, and white stripes, light shades, salmon pink, graphic print, dark-grey, stained with multiple colors, light color, possibly white or gray, light yellow-green, shiny metallic, greenish-gray, plaid with white, black, red and yellow, gray-green, skin, plaid with beige, black, and white, blond".split(","):
            for w in ws.strip().split(" "):
                self.color_words.add(w)

    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............."""
        if shard_count > 1:
            return f"{self.progress_file_prefix}_shard_{shard_index}_of_{shard_count}.json"
        else:
            return f"{self.progress_file_prefix}.json"
    
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        """..........qwen_detailing.."""
        return False
        persons = data.get('persons', [])
        if not persons:
            return True  # ..persons........
        
        # ......person..qwen_detailing
        for person in persons:
            if person.get('body_box') is not None and ('qwen_detailing' not in person or not person['qwen_detailing']):
                return False
        return True
    
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """..Qwen......"""
        console.print(f"[dim]🔍 ......: {os.path.basename(file_path)}[/dim]")
        
        
        processed_count = 0
        for person_idx, person in enumerate(data.get('persons', [])):
            if person.get("qwen_detailing") is None:
                continue
            for cloth in person["qwen_detailing"].get("clothing", []):
                need_check = False
                for word in cloth['name'].strip().lower().split(" "):
                    if word in self.color_words:
                        need_check = True
                        break
                if not need_check:
                    continue

                console.print(f"[dim]👗 ...{person_idx+1}.......: {cloth['name']}[/dim]")
                cloth['name'] = (await asyncio.to_thread(ask_question, f"{cloth['name']}\nPlease remove any color related information from the expression above. Give the original word if no color is mentioned. One single answer should be given with only lower case characters and space directly.")).strip()

        console.print(f"[dim]🎉 ......: {os.path.basename(file_path)}, ... {processed_count} ..[/dim]")
        return data


class ClothingCorrectionTask(BaseTask):
    """Qwen........"""
    
    def __init__(self):
        super().__init__(
            task_name="clothing_correction",
            data_dir="./final_labeling",
            progress_file_prefix="clothing_correction_progress"
        )
        self.filename_set = set()
        for jfn in ["further_check_result_hico.json", "further_check_result_org.json"]:
            with open(jfn, "r") as f:
                full_result = json.load(f)
                for result in full_result:
                    filename = result["filename"].split("/")[-1]
                    if os.path.exists(f"./final_labeling/{filename}"):
                        self.filename_set.add(filename)
    
    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............."""
        if shard_count > 1:
            return f"{self.progress_file_prefix}_shard_{shard_index}_of_{shard_count}.json"
        else:
            return f"{self.progress_file_prefix}.json"
    
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        """..........qwen_detailing.."""
        return file_path.split("/")[-1] in self.filename_set
        return False
        persons = data.get('persons', [])
        if not persons:
            return True  # ..persons........
        
        # ......person..qwen_detailing
        for person in persons:
            if person.get('body_box') is not None and ('qwen_detailing' not in person or not person['qwen_detailing']):
                return False
        return True
    
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """..Qwen......"""
        console.print(f"[dim]🔍 ......: {os.path.basename(file_path)}[/dim]")
        
        body_boxes = data['detect_results']['body_boxes']
        face_boxes = data['detect_results']['face_boxes']
        image_path = os.path.join(self.data_dir, data['image_path'].split("/")[-1])
        
        console.print(f"[dim]📷 ....: {data['image_path']}[/dim]")
        image = cv.imread(image_path)
        if image is None:
            raise Exception(f"........: {image_path}")
        
        H, W, C = image.shape
        console.print(f"[dim]📐 ....: {W}x{H}, ...: {C}[/dim]")
        
        persons_count = len(data.get('persons', []))
        persons_with_body_box = len([p for p in data.get('persons', []) if p.get('body_box') is not None])
        console.print(f"[dim]👥 ...: {persons_count}, .......: {persons_with_body_box}[/dim]")
        
        processed_count = 0
        for person_idx, person in enumerate(data.get('persons', [])):
            if person.get("deleted") is True:
                continue
            if person['body_box'] is None:
                console.print(f"[dim]⏭️  ...{person_idx+1}.. (....)[/dim]")
                continue
            
            console.print(f"[dim]👤 ...{person_idx+1}.. (body_box..: {person['body_box']})[/dim]")
                
            # Create annotated image with person bounding box
            info_img = image.copy()
            body_box = body_boxes[person['body_box']]
            x1, y1, x2, y2 = (body_box[0] * W, body_box[1] * H, body_box[2] * W, body_box[3] * H)
            console.print(f"[dim]📦 .....: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})[/dim]")
            if max(H, W) > 1000:
                cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
                cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            if person["face_box"] is not None and len(body_boxes) > 0:
                face_box = face_boxes[person["face_box"]]
                fx1, fy1, fx2, fy2 = (face_box[0] * W, face_box[1] * H, face_box[2] * W, face_box[3] * H)
                if max(H, W) > 1000:
                    cv.rectangle(info_img, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 0), 2)
                else:
                    cv.rectangle(info_img, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 0), 1)

            question = """Please provide detailed information about wearing including clothing and accessories of the person in the green bounding box in following JSON format:

```json
{
    "vague": true/false, // true if the boundingbox is too small or the person is not clearly visible
    "clothing": [
        {
            "possible_names": ["list of possible names in lowercase, no infomation about color should be included"],
            "name": "clothing name in lowercase, please choose the most concise one avoiding ambiguity from possible_names",
            "type": "type of clothing, please only use one of 'top', 'bottom', 'whole body', 'footwear', 'handwear', 'headwear', 'accessory', 'other'",
            "color": ["a list of color of clothing describing, please give the name of the color in lowercase"],
            "belonging_confident": true/false, // true if you are confident that the clothing belongs to the person in the green bounding box
            "existence_confident": true/false // true if you are confident that the clothing is really in the image
        },
        ...
    ],
}
```

BE AWARE! Don't describe any wearings that are not visible in the image. You are not allowed to imagine possibilities."""
            console.print(f"[dim]🤖 ....AI.....{person_idx+1}.....[/dim]")
            
            try:
                response = await asyncio.to_thread(ask_about_image, info_img, question, json_format=True)
                console.print(f"[dim]✅ AI...... (....: {len(response)} ..)[/dim]")
                
                # ........（....）
                response_preview = response[:200] + "..." if len(response) > 200 else response
                console.print(f"[dim]📝 ....: {response_preview}[/dim]")
                
            except Exception as api_error:
                console.print(f"[red]❌ AI...... (.{person_idx+1}..): {api_error}[/red]")
                raise Exception(f"AI......: {api_error}")
            
            try:
                console.print(f"[dim]🔧 ....JSON.....[/dim]")
                clothing_info = json.loads(response)
                person["qwen_detailing"]["clothing"] = clothing_info
                
            except json.JSONDecodeError as e:
                console.print(f"[red]❌ JSON.... (.{person_idx+1}..): {e}[/red]")
                console.print(f"[red]....: {response}[/red]")
                raise Exception(f"JSON....: {e}")
                

            processed_count += 1
            console.print(f"[dim]✅ .{person_idx+1}......[/dim]")
        
        console.print(f"[dim]🎉 ......: {os.path.basename(file_path)}, ... {processed_count} ..[/dim]")
        return data
    
class HoiUnifyTask(BaseTask):
    """Qwen........"""
    
    def __init__(self):
        super().__init__(
            task_name="hoi_unify",
            data_dir="./final_labeling",
            progress_file_prefix="hoi_unify_progress"
        )
    
    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............."""
        if shard_count > 1:
            return f"{self.progress_file_prefix}_shard_{shard_index}_of_{shard_count}.json"
        else:
            return f"{self.progress_file_prefix}.json"
    
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        return False
        return False
        persons = data.get('persons', [])
        if not persons:
            return True  # ..persons........
        
        # ......person..qwen_detailing
        for person in persons:
            if person.get('body_box') is not None and ('qwen_detailing' not in person or not person['qwen_detailing']):
                return False
        return True
    
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """..Qwen......"""
        console.print(f"[dim]🔍 ......: {os.path.basename(file_path)}[/dim]")
        
        body_boxes = data['detect_results']['body_boxes']
        face_boxes = data['detect_results']['face_boxes']
        image_path = os.path.join(self.data_dir, data['image_path'].split("/")[-1])
        
        image = cv.imread(image_path)
        if image is None:
            raise Exception(f"........: {image_path}")
        
        H, W, C = image.shape
        
        processed_count = 0
        for person_idx, person in enumerate(data.get('persons', [])):
            if person.get("deleted") is True:
                continue
            # if person['body_box'] is None:
            #     console.print(f"[dim]⏭️  ...{person_idx+1}.. (....)[/dim]")
            #     continue

            info_img = image.copy()
            if person['body_box'] is not None:
                body_box = body_boxes[person['body_box']]
                x1, y1, x2, y2 = (body_box[0] * W, body_box[1] * H, body_box[2] * W, body_box[3] * H)
                if max(H, W) > 1000:
                    cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                else:
                    cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            else:
                body_box = face_boxes[person['face_box']]
                x1, y1, x2, y2 = (body_box[0] * W, body_box[1] * H, body_box[2] * W, body_box[3] * H)
                if max(H, W) > 1000:
                    cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                else:
                    cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            
            for hoi in person.get("hoi", []):
                if hoi.get("deleted", False) is True:
                    continue
                if data["objects"][hoi["object"]].get("deleted", False) is True:
                    hoi["deleted"] = True
                    continue

                obj_name = data["objects"][hoi["object"]].get("name", "unknown")
                hoi = hoi["relationship"]
                if len(hoi["action"]) == 0 or (not isinstance(hoi["action"][0], str)):
                    continue
                default_position = hoi.get("position", None)
                if default_position is None:
                    result = await asyncio.to_thread(ask_about_image, info_img, f"The person in the green bounding box is interacting with {obj_name}, please provide the body part where the person is in physical contact with the object, provide your answer with following json fomat:\n"+"""
```json
{
    "contact": true/false, // true if there is a physical contact
    "body_part": "please provide answer only using one of following words: 'hand', 'mouth', 'head', 'body', 'thigh', 'foot'. if there is no physical contact, use 'standalone'"
}                                                  
```
""", json_format=True)
                    result = json.loads(result)
                    if result.get("contact", False) is False:
                        result = "standalone"
                    else:
                        result = result.get("body_part", "unknown")
                    hoi["position"] = result
                    default_position = result

                new_action = []
                for action in hoi.get("action", []):
                    if len(action.strip().split(" ")) < 3:
                        new_action.append((default_position, action))
                    else:
                        result = await asyncio.to_thread(ask_question, f"Please analyze the action expressed in the following text: '{action}' with following json format:\n"+"""

```json
{
    "body_part_action_pairs": [
        {
            "body_part": "the body part in the expression for the corresponding action, give no more than 2 words and must be selected from the origional text. If no corresponding body part is indicated use 'unknown'",
            "action": "a verb or action phrase selected from the origional text without body part information"
        },
        ...
    ]
}
```
""", json_format=True)
                        result = json.loads(result)
                        if not result.get("body_part_action_pairs"):
                            result["body_part_action_pairs"] = []
                        for pair in result["body_part_action_pairs"]:
                            if pair.get("body_part", "unknown") == "unknown":
                                pair["body_part"] = default_position
                            if pair.get("action", "").strip() == "":
                                pair["action"] = action
                            new_action.append((pair["body_part"], pair["action"]))
                hoi["action"] = new_action

            processed_count += 1
        return data


class AbstractTask(BaseTask):
    """WoXunSi........"""

    def __init__(self):
        super().__init__(
            task_name="abstract_labeling",
            data_dir="./abstract_final_labeling/emotion",
            progress_file_prefix="abstract_labeling_progress"
        )
    
    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............."""
        if shard_count > 1:
            return f"{self.progress_file_prefix}_shard_{shard_index}_of_{shard_count}.json"
        else:
            return f"{self.progress_file_prefix}.json"
    
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        return False
        # if "scene" in data:
        #     return True
        # ans = True
        # effective = False
        # for person in data.get("persons", []):
        #     if person.get("deleted", False) is True:
        #         continue
        #     if person.get("qwen_detailing") is None:
        #         continue
        #     # if person["qwen_detailing"].get("emotion") == "complex":
        #     #     return False
        #     effective = True
        #     if person["qwen_detailing"].get("meaningful") is False:
        #         ans = False
        #         break
        # return not (ans and effective)
        
    

    
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """.........."""
        console.print(f"[dim]🔍 ......: {os.path.basename(file_path)}[/dim]")
        
        body_boxes = data['detect_results']['body_boxes']
        face_boxes = data['detect_results']['face_boxes']
        image_path = os.path.join(self.data_dir, data['image_path'].split("/")[-1])
        
        console.print(f"[dim]📷 ....: {data['image_path']}[/dim]")
        image = cv.imread(image_path)
        if image is None:
            raise Exception(f"........: {image_path}")
        
        H, W, C = image.shape
        console.print(f"[dim]📐 ....: {W}x{H}, ...: {C}[/dim]")
        
        persons_count = len(data.get('persons', []))
        persons_with_body_box = len([p for p in data.get('persons', []) if p.get('body_box') is not None])
        console.print(f"[dim]👥 ...: {persons_count}, .......: {persons_with_body_box}[/dim]")
        model_name = "internvl"
        scene = await asyncio.to_thread(ask_about_image, image, "Please give a short description of the scene, you can describe about cultural background, enviroment, overall color, style, any notable objects or anything else. But do not give any description about the people in the image. Give your answer within one sentence without any punctuation.", model_name=model_name, json_format=False)
        data["scene"] = scene

        processed_count = 0
        for person_idx, person in enumerate(data.get('persons', [])):
            if person['body_box'] is None:
                console.print(f"[dim]⏭️  ...{person_idx+1}.. (....)[/dim]")
                continue
            if person.get("deleted") is True:
                continue
            
            console.print(f"[dim]👤 ...{person_idx+1}.. (body_box..: {person['body_box']})[/dim]")
                
            # Create annotated image with person bounding box
            info_img = image.copy()
            body_box = body_boxes[person['body_box']]
            x1, y1, x2, y2 = (body_box[0] * W, body_box[1] * H, body_box[2] * W, body_box[3] * H)
            console.print(f"[dim]📦 .....: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})[/dim]")
            if max(H, W) > 1000:
                cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
                cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            if person["face_box"] is not None and len(body_boxes) > 0:
                face_box = face_boxes[person["face_box"]]
                fx1, fy1, fx2, fy2 = (face_box[0] * W, face_box[1] * H, face_box[2] * W, face_box[3] * H)
                if max(H, W) > 1000:
                    cv.rectangle(info_img, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 0), 2)
                else:
                    cv.rectangle(info_img, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 0), 1)
            if f"{model_name}_detailing" not in person:
                person[f"{model_name}_detailing"] = {}
            if person["qwen_detailing"]["emotion"] not in ["complex", "unknown", "neutral"]:
                person[f"{model_name}_detailing"]["complex_emotion"] = await asyncio.to_thread(ask_about_image, image, f"The emotion of the person in the green bounding box is {person['qwen_detailing']['emotion']}. Please give a detailed analysis about the person's emotion and thought in few paragraphs without any summary, markdown, or bullet points. Don't give anything else.", model_name=model_name, json_format=False)
                person[f"{model_name}_detailing"]["complex_emotion_clean"] = (await asyncio.to_thread(ask_question, f"{person['qwen_detailing']['complex_emotion']}\n\nProvided above is a detailed analysis of the {person['qwen_detailing']['emotion']} emotion of someone. Please remove any information about the appearance, gender, clothing, overall action or context, just keep the analysis about their feelings, thoughts and emotion. Give your answer within one sentence without any punctuation in a assertive tone. Remove any information related to age, gender or any other personal attributes. Don't give anything else.", model_name=model_name, json_format=False)).strip()

            person[f"{model_name}_detailing"]["behaviour"] = (await asyncio.to_thread(ask_about_image, image, f"Please describe the behavior of the person in the green bounding box. You can describe their relationship with others or certain objects, their action and possible reason, their expression and possible motivations, and anything else. Provide your answer in a few sentences within one paragraph without any summary, markdown, or bullet points. Don't give anything else.", model_name=model_name, json_format=False)).strip()
            person[f"{model_name}_detailing"]["intention"] = (await asyncio.to_thread(ask_question, f"{person[f'{model_name}_detailing']['behaviour']}\n\nProvided above is a description of someone's behavior. Please analyze and infer their possible intention or motivation behind the behavior. Remove any information related to age, gender or any other personal attributes. Give your answer within one sentence without any punctuation in a assertive tone. Don't give anything else.", model_name=model_name, json_format=False))
            ans = (await asyncio.to_thread(ask_about_image, image, f"Is the person in the green bounding box likely to have following intentions or motivations: \n\n{person[f'{model_name}_detailing']['intention']}\n\nPlease give your analysis and provide 'yes' or 'no' in the last line of your answer.", model_name=model_name, json_format=False)).strip().lower()
            person[f"{model_name}_detailing"]["intention_ok"] = yes_or_no(ans)
            processed_count += 1
            console.print(f"[dim]✅ .{person_idx+1}......[/dim]")

        persons_desc = ""
        for person_idx, person in enumerate(data.get('persons', [])):
            if person.get("deleted", False) is True:
                continue
            if person.get("qwen_detailing"):
                persons_desc += f"Person {person_idx}: {person['qwen_detailing']['behaviour']} \n"
        if len(persons_desc) == 0:
            return data
        data[f"{model_name}_overall_past"] = (await asyncio.to_thread(ask_question, f"Scene: {scene}\n\n{persons_desc}\n\nListed above are descriptions about an image with person in it. According to the image descriptions, what might happened before the current scene? Give your analyze and provide one single plausible answer. Do not use expressions like 'Person 0' in your answer.", model_name=model_name, json_format=False)).strip()
        data[f"{model_name}_overall_past_clean"] = (await asyncio.to_thread(ask_question, f"{data[f'{model_name}_overall_past']}\n\nAn analyze of what might have happened before a scene is given above. Please describe what happened before the scene in a assertive manner as if it is happening. Do not give any information about current scene, specific wearing, or specific objects. Your description should be targeted to what happened before. Remove any information related to age, gender or any other personal attributes. Provide your answer within only one sentence without anything else.", model_name=model_name, json_format=False)).strip()

        

        ans = (await asyncio.to_thread(ask_about_image, image, f"Is it possible that the scene in the image is happening after what is described below:\n\n{data[f'{model_name}_overall_past_clean']}\n\nPlease give your analysis and provide 'yes' or 'no' in the last line of your answer.", model_name=model_name, json_format=False)).strip().lower()
        data[f"{model_name}_past_scene_ok"] = yes_or_no(ans)

        data[f"{model_name}_overall_future"] = (await asyncio.to_thread(ask_question, f"Scene: {scene}\n\n{persons_desc}\n\nListed above are descriptions about an image with person in it. According to the image descriptions, what might happen after the current scene? Give your analyze and provide one single plausible answer. Do not use expressions like 'Person 0' in your answer.", model_name=model_name, json_format=False)).strip()
        data[f"{model_name}_overall_future_clean"] = (await asyncio.to_thread(ask_question, f"{data[f'{model_name}_overall_future']}\n\nAn analyze of what might happen after a scene is given above. Please describe what might happen after the scene in a assertive manner as if it is happening. Do not give any information about current scene, specific wearing, or specific objects. Your description should be targeted to what might happen after. Remove any information related to age, gender or any other personal attributes. Provide your answer within only one sentence without anything else.", model_name=model_name, json_format=False)).strip()

        ans = (await asyncio.to_thread(ask_about_image, image, f"Is it possible that after what the scene in the image happened, following scene happens:\n\n{data[f'{model_name}_overall_future_clean']}\n\nPlease give your analysis and provide 'yes' or 'no' in the last line of your answer.", model_name=model_name, json_format=False)).strip().lower()
        data[f"{model_name}_future_scene_ok"] = yes_or_no(ans)

        if data[f"{model_name}_past_scene_ok"] and data[f"{model_name}_future_scene_ok"]:
            print(f"[dim]🔍 .......: {data[f'{model_name}_overall_past_clean']} -> {data[f'{model_name}_overall_future_clean']}[/dim]")
        console.print(f"[dim]🎉 ......: {os.path.basename(file_path)}, ... {processed_count} ..[/dim]")
        return data

class QwenDetailingTask(BaseTask):
    """Qwen........"""
    
    def __init__(self):
        super().__init__(
            task_name="qwen_detailing",
            data_dir="./final_labeling",
            progress_file_prefix="qwen_detailing_progress"
        )
    
    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............."""
        if shard_count > 1:
            return f"{self.progress_file_prefix}_shard_{shard_index}_of_{shard_count}.json"
        else:
            return f"{self.progress_file_prefix}.json"
    
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        return False
        ans = True
        for person in data.get("persons", []):
            if person.get("deleted", False) is True:
                continue
            if person.get("qwen_detailing") is None:
                continue

        persons = data.get('persons', [])
        if not persons:
            return True  # ..persons........
        
        # ......person..qwen_detailing
        for person in persons:
            if person.get('body_box') is not None and ('qwen_detailing' not in person or not person['qwen_detailing']):
                return False
        return True
    
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """..Qwen......"""
        console.print(f"[dim]🔍 ......: {os.path.basename(file_path)}[/dim]")
        
        body_boxes = data['detect_results']['body_boxes']
        face_boxes = data['detect_results']['face_boxes']
        image_path = os.path.join(self.data_dir, data['image_path'].split("/")[-1])
        
        console.print(f"[dim]📷 ....: {data['image_path']}[/dim]")
        image = cv.imread(image_path)
        if image is None:
            raise Exception(f"........: {image_path}")
        
        H, W, C = image.shape
        console.print(f"[dim]📐 ....: {W}x{H}, ...: {C}[/dim]")
        
        persons_count = len(data.get('persons', []))
        persons_with_body_box = len([p for p in data.get('persons', []) if p.get('body_box') is not None])
        console.print(f"[dim]👥 ...: {persons_count}, .......: {persons_with_body_box}[/dim]")
        
        processed_count = 0
        for person_idx, person in enumerate(data.get('persons', [])):
            if person['body_box'] is None:
                console.print(f"[dim]⏭️  ...{person_idx+1}.. (....)[/dim]")
                continue
            
            console.print(f"[dim]👤 ...{person_idx+1}.. (body_box..: {person['body_box']})[/dim]")
                
            # Create annotated image with person bounding box
            info_img = image.copy()
            body_box = body_boxes[person['body_box']]
            x1, y1, x2, y2 = (body_box[0] * W, body_box[1] * H, body_box[2] * W, body_box[3] * H)
            console.print(f"[dim]📦 .....: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})[/dim]")
            if max(H, W) > 1000:
                cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
                cv.rectangle(info_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            if person["face_box"] is not None and len(body_boxes) > 0:
                face_box = face_boxes[person["face_box"]]
                fx1, fy1, fx2, fy2 = (face_box[0] * W, face_box[1] * H, face_box[2] * W, face_box[3] * H)
                if max(H, W) > 1000:
                    cv.rectangle(info_img, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 0), 2)
                else:
                    cv.rectangle(info_img, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 0), 1)

            question = """Please provide detailed information about the person in the green bounding box in following JSON format:

# ```json
# {
#     "background": true/false, // whether the person is in the background or foreground, true means background, false means foreground,
#     "age": "please only use one of 'child', 'teenager', 'adult', 'senior', 'unknown'",
#     "gender": "please only use one of 'male', 'female', 'unknown'",
#     "emotion": "please only use one of 'happy', 'sad', 'angry', 'surprised', 'neutral', 'unknown'",
#     "clothing_description": "provide a detailed description of the person's clothing",
#     "clothing": [
#         {
#             "possible_names": ["list of possible names in lowercase"],
#             "name": "clothing name in lowercase, please choose the most concise one avoiding ambiguity from possible_names",
#             "type": "type of clothing, please only use one of 'top', 'bottom', 'whole body', 'footwear', 'handwear', 'headwear', 'accessory', 'other'",
#             "color": ["a list of color of clothing describing, please give the name of the color in lowercase"]
#         },
#         ...
#     ],
#     "objects": [ // list of objects relevant to the person described, including physical contact, eye contact, possible causal relationships, etc. DO NOT consider objects the person is wearing, they should be included in the clothing list instead.
#         {
#             "standalone": true/false, // true if the object has no physical contact
#             "possible_names": ["list of possible names in lowercase"],
#             "name": "object name in lowercase, please choose the most concise one avoiding ambiguity from possible_names.",
#             "position": "the part of the body that holds the object, please only use one of 'hand', 'head', 'body', 'foot' or 'other'. If the object is not held by the person, please use 'standalone' instead.",
#         },
#         ...
#     ],
#     "description": "detailed description of the person, including all the above information and any other relevant details"
# }
# ```"""
            question = """Please provide detailed information about the person in the green bounding box in following JSON format:

```json
{
    "no_person": true/false, // if no person or bounding box can be seen, set this to true and leave all following fields empty
    "background": true/false, // whether the person is in the background or foreground, true means background, false means foreground,
    "blurry": true/false, // whether the person is blurry
    "face_seen": true/false, // whether the person's face is recognizable
    "age": "please only use one of 'baby', 'child', 'teenager', 'adult', 'senior', 'unknown'",
    "gender": "please only use one of 'male', 'female', 'unknown'",
    "race": "please use one of 'white', 'black', 'asian', 'middle eastern', 'latino hispanic', 'unknown'",
    "emotion": "please only use one of 'happy', 'sad', 'angry', 'surprised', 'neutral', 'complex', 'unknown'", // provide "complex" if you think the person got mixed feeling
    "emotion_description": "provide a detailed description of the person's emotion",
    "meaningful": true/false, // whether the image provide a context where a story can be infered
    "story": "provide a brief story or context about the person. If no story can be infered, just provide 'unknown'",
    "text": "any actual text present on the person, including clothing labels, tattoos, etc. Separate multiple items with a comma. Give 'no_text' if there is none.",
    "text_relationship": "describe the relationship between the text and the person, including any relevant context or meaning. Give 'no_text' if there is none."
}
```"""
            console.print(f"[dim]🤖 ....AI.....{person_idx+1}.....[/dim]")
            
            try:
                response = await asyncio.to_thread(ask_about_image, info_img, question, json_format=True)
                console.print(f"[dim]✅ AI...... (....: {len(response)} ..)[/dim]")
                
                # ........（....）
                response_preview = response[:200] + "..." if len(response) > 200 else response
                console.print(f"[dim]📝 ....: {response_preview}[/dim]")
                
            except Exception as api_error:
                console.print(f"[red]❌ AI...... (.{person_idx+1}..): {api_error}[/red]")
                raise Exception(f"AI......: {api_error}")
            
            try:
                console.print(f"[dim]🔧 ....JSON.....[/dim]")
                person_info = json.loads(response)
                if person_info.get("no_person", False):
                    console.print(f"[yellow]⚠️  ........[/yellow]")
                    person["deleted"] = True
                    continue
                # .....JSON..
                required_fields = ["blurry", "face_seen", "age", "gender", "emotion", "emotion_description", "meaningful", "story", "race", "text", "text_relationship"]
                missing_fields = [field for field in required_fields if field not in person_info]
                if missing_fields:
                    console.print(f"[yellow]⚠️  JSON......: {missing_fields}[/yellow]")
                
                console.print(f"[dim]✅ JSON.... (.. {len(person_info)} ...)[/dim]")
                
            except json.JSONDecodeError as e:
                console.print(f"[red]❌ JSON.... (.{person_idx+1}..): {e}[/red]")
                console.print(f"[red]....: {response}[/red]")
                raise Exception(f"JSON....: {e}")
                
            person["qwen_detailing"]["blurry"] = person_info["blurry"]
            person["qwen_detailing"]["face_seen"] = person_info["face_seen"]
            person["qwen_detailing"]["age"] = person_info["age"]
            person["qwen_detailing"]["gender"] = person_info["gender"]
            person["qwen_detailing"]["emotion"] = person_info["emotion"]
            person["qwen_detailing"]["emotion_description"] = person_info["emotion_description"]
            person["qwen_detailing"]["meaningful"] = person_info["meaningful"]
            person["qwen_detailing"]["story"] = person_info["story"]
            person["qwen_detailing"]["race"] = person_info["race"]
            person["qwen_detailing"]["text"] = person_info["text"]
            person["qwen_detailing"]["text_relationship"] = person_info["text_relationship"]

            processed_count += 1
            console.print(f"[dim]✅ .{person_idx+1}......[/dim]")
        
        console.print(f"[dim]🎉 ......: {os.path.basename(file_path)}, ... {processed_count} ..[/dim]")
        return data


class ExampleTask(BaseTask):
    """.... - ........."""
    
    def __init__(self):
        super().__init__(
            task_name="example_task",
            data_dir="./example_data",
            progress_file_prefix="example_progress"
        )
    
    def get_progress_filename(self, shard_index: int = 0, shard_count: int = 1) -> str:
        """............."""
        if shard_count > 1:
            return f"{self.progress_file_prefix}_shard_{shard_index}_of_{shard_count}.json"
        else:
            return f"{self.progress_file_prefix}.json"
    
    def is_file_processed(self, file_path: str, data: Dict[str, Any]) -> bool:
        """........... - ...."""
        # ..：..........
        return 'example_field' in data and data['example_field'] is not None
    
    async def process_json_data(self, file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """..JSON.. - ...."""
        console.print(f"[dim]🔍 ........: {os.path.basename(file_path)}[/dim]")
        
        # ..：.........
        import time
        data['example_field'] = {
            'processed_at': time.time(),
            'file_name': os.path.basename(file_path)
        }
        
        console.print(f"[dim]✅ ........: {os.path.basename(file_path)}[/dim]")
        return data


# .....
AVAILABLE_TASKS = {
    'qwen_detailing': QwenDetailingTask,
    'object_detect': ObjectDetectTask,
    'hoi_detect': HoiDetectTask,
    'facex_detailing': FacexTask,
    'remove_color': ColorRemovalTask,
    'clothing_correction': ClothingCorrectionTask,
    'hoi_unify': HoiUnifyTask,
    "abstract_labeling": AbstractTask,
    'example': ExampleTask,
}

def get_task(task_name: str) -> BaseTask:
    """............"""
    if task_name not in AVAILABLE_TASKS:
        available = ', '.join(AVAILABLE_TASKS.keys())
        raise ValueError(f"....: {task_name}. ....: {available}")
    
    return AVAILABLE_TASKS[task_name]()

def list_available_tasks() -> List[str]:
    """........"""
    return list(AVAILABLE_TASKS.keys())

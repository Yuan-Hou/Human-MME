import base64
from io import BytesIO
import os
import json
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed
from mllm_models.base import BaseModel
from mllm_models.vllm_api_model import VllmApiModel
from mllm_models.zhipu_model import ZhipuModel
from mllm_models.intern_api_multikey_model import InternApiMultiKeyModel
from mllm_models.model_test import DummyModel

from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

import regex as re

from metrics import is_correct, composite_score, attr_candidates, max_guess_format_iou, evaluate_ranking

import csv

MODEL_NAME_MAP = {
    "qwen2.5-vl-72b-awq": VllmApiModel,
    "glm-4.5v": VllmApiModel,
    "qwen2.5-vl-72b": VllmApiModel,
    "internvl3.5-38b": VllmApiModel,
    "glm-4.1v-thinking-flash": ZhipuModel,
    "llava-next-72b": VllmApiModel,
    "gemma3-27b": VllmApiModel,
    "minicpmv": VllmApiModel,
    "intern-s1": InternApiMultiKeyModel,
    "phi4": VllmApiModel,
    "qwen2.5-vl-7b": VllmApiModel,
    "qwen2.5-vl-32b": VllmApiModel,
    "aya-vision-32b": VllmApiModel,
    "kimi-vl": VllmApiModel,
    "internvl3-78b": VllmApiModel,
    "llama4": VllmApiModel,
    "llama3.2": VllmApiModel,
    "model_test": DummyModel,
}

FORMAT_INSTRUCTIONS = {
    "bbox": "Your answer should follow this format strictly: \n\nAnalyze: <your analysis>\nAnswer: [x1,y1,x2,y2]",
    "choice": "Your answer should follow this format strictly: \n\nAnalyze: <your analysis>\nAnswer: A/B/C/D",
    "blank": "Your answer should follow this format strictly: \n\nAnalyze: <your analysis>\nAnswer: <your final answer in a short and concise expression>",
    "tf_bbox": "Your answer should follow this format strictly: \n\nAnalyze: <your analysis>\nAnswer: [x1,y1,x2,y2], if no match, answer [-1,-1,-1,-1]",
    "tf_blank": "Your answer should follow this format strictly: \n\nAnalyze: <your analysis>\nAnswer: <your final answer in a short and concise expression>, if no match, answer unknown",
    "open_hoi": "Your answer should follow this format strictly: \n\nAnalyze: <your analysis>\nName: <name of the object>\nBox: [x1,y1,x2,y2]",
    "sequence": "Your answer should follow this format strictly, the sequence must be uniquely determined: \n\nAnalyze: <your analysis>\nFirst: A/B/C/D\nSecond: A/B/C/D\nThird: A/B/C/D\nFourth: A/B/C/D",
    "double_choice": "Your answer should follow this format strictly: \n\nAnalyze: <your analysis>\nPast: A/B/C/D\nFuture: A/B/C/D",
}

REG_ANALYSIS = re.compile(r'Analyze:\s*(.*?)\s*(Answer:|Name:|Box:|First:|Second:|Third:|Fourth:|Past:|Future:)', re.IGNORECASE | re.DOTALL)

REG_BBOX = re.compile(r'Answer:\s*\[?\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\]?', re.IGNORECASE)
REG_CHOICE = re.compile(r'Answer:\s*([A-D])', re.IGNORECASE)
REG_BLANK = re.compile(r'Answer:\s*(.+)', re.IGNORECASE)
REG_HOI_NAME = re.compile(r'Name:\s*(.*?)\s*Box:', re.IGNORECASE | re.DOTALL)
# 允许[]嵌套两层出现，支持整数和小数
REG_HOI_BOX = re.compile(r'Box:\s*\[?\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\]?', re.IGNORECASE)
REG_SEQ_FIRST = re.compile(r'First:\s*([A-D])', re.IGNORECASE)
REG_SEQ_SECOND = re.compile(r'Second:\s*([A-D])', re.IGNORECASE)
REG_SEQ_THIRD = re.compile(r'Third:\s*([A-D])', re.IGNORECASE)
REG_SEQ_FOURTH = re.compile(r'Fourth:\s*([A-D])', re.IGNORECASE)
REG_DC_PAST = re.compile(r'Past:\s*([A-D])', re.IGNORECASE)
REG_DC_FUTURE = re.compile(r'Future:\s*([A-D])', re.IGNORECASE)

BBOX_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "white": (255, 255, 255),
}

def image_to_1080p(image: Image, upscale = 1.0) -> Image:
    W, H = image.size
    min_side = min(W, H)
    max_side = max(W, H)
    scale = min(1080 / min_side, 1920 / max_side, 1.0) * upscale
    new_W = int(W * scale)
    new_H = int(H * scale)
    return image.resize((new_W, new_H))

class Question():
    def __init__(self, dir_path):
        qa_file = os.path.join(dir_path, 'qa.json')
        with open(qa_file, 'r') as f:
            self.qa_data = json.load(f)
        self.id = [s for s in dir_path.split('/') if len(s) > 0][-1]
        self.path = dir_path.replace('\\', '/')
        self.q_type = self.qa_data['type']
        self.q_img_msg, self.q_text_msg = self.qa_data['question']
        if self.q_img_msg['type'] in ['image', 'image_bbox']:
            self.W = self.q_img_msg["width"]
            self.H = self.q_img_msg["height"]

        self.a_type = self.qa_data['answer']["type"]
        self.a_obj = self.qa_data['answer'].get("data", self.qa_data['answer'].get("answer", None))

        self.q_text = f"{self.q_text_msg['text']}\n\n{FORMAT_INSTRUCTIONS[self.a_type]}"
    
    def get_image(self):
        if self.q_img_msg['type'] == 'image':
            img_path = self.q_img_msg['image']
            return image_to_1080p(Image.open(img_path).convert('RGB'))
        elif self.q_img_msg['type'] == 'image_bbox':
            img_path = self.q_img_msg['image']
            image = image_to_1080p(Image.open(img_path).convert('RGB'))
            W, H = image.size
            for color, box in self.q_img_msg['bboxes']:
                ImageDraw.Draw(image).rectangle((box[0]*W, box[1]*H, box[2]*W, box[3]*H), outline=BBOX_COLORS[color], width=1)
            
            return image
        elif self.q_img_msg['type'] == 'multi_image':
            image_A = Image.open(self.q_img_msg['image_A']).convert('RGB')
            image_B = Image.open(self.q_img_msg['image_B']).convert('RGB')
            image_C = Image.open(self.q_img_msg['image_C']).convert('RGB')
            image_D = Image.open(self.q_img_msg['image_D']).convert('RGB')
            # 四张图片全部缩放并黑边pad到1024*1024，左上角标注相应字母
            def resize_and_pad(image, size=(1024, 1024), label=None):
                ratio = min(size[0] / image.width, size[1] / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size)
                new_image = Image.new("RGB", size, (0, 0, 0))
                new_image.paste(image, ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2))
                if label:
                    draw = ImageDraw.Draw(new_image)
                    font = ImageFont.truetype("arial.ttf", 60)
                    draw.text((10, 10), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font=font)
                return new_image
            image_A = resize_and_pad(image_A, label="A")
            image_B = resize_and_pad(image_B, label="B")
            image_C = resize_and_pad(image_C, label="C")
            image_D = resize_and_pad(image_D, label="D")
            # 四张图片横向拼接
            new_image = Image.new("RGB", (image_A.width + image_B.width + image_C.width + image_D.width, 1024), (255, 255, 255))
            new_image.paste(image_A, (0, 0))
            new_image.paste(image_B, (image_A.width, 0))
            new_image.paste(image_C, (image_A.width + image_B.width, 0))
            new_image.paste(image_D, (image_A.width + image_B.width + image_C.width, 0))
            # 画分割线
            draw = ImageDraw.Draw(new_image)
            draw.line((image_A.width, 0, image_A.width, 1024), fill=(255,255,255), width=3)
            draw.line((image_A.width + image_B.width, 0, image_A.width + image_B.width, 1024), fill=(255,255,255), width=3)
            draw.line((image_A.width + image_B.width + image_C.width, 0, image_A.width + image_B.width + image_C.width, 1024), fill=(255,255,255), width=3)
            
            return image_to_1080p(new_image, upscale=2)

    def solve(self, model: BaseModel):
        result = model.predict(self.get_image(), self.q_text)
        result_dict = {}
        if self.a_type == 'bbox':
            # 去除result中"```"开头的行
            result_prue = "\n".join([line for line in result.split("\n") if not line.strip().startswith("```")])
            match = REG_BBOX.search(result_prue)
            if match:
                result_dict['bbox'] = [float(match.group(i)) for i in range(1, 5)]
            else:
                result_dict['bbox'] = [-1, -1, -1, -1]
            analysis_match = REG_ANALYSIS.search(result)
            result_dict['analysis'] = analysis_match.group(1).strip() if analysis_match else ""
        elif self.a_type == 'tf_bbox':
            result_prue = "\n".join([line for line in result.split("\n") if not line.strip().startswith("```")])
            match = REG_BBOX.search(result_prue)
            if match:
                result_dict['bbox'] = [float(match.group(i)) for i in range(1, 5)]
            else:
                result_dict['bbox'] = [-1, -1, -1, -1]
            analysis_match = REG_ANALYSIS.search(result)
            result_dict['analysis'] = analysis_match.group(1).strip() if analysis_match else ""
        elif self.a_type == 'choice':
            match = REG_CHOICE.search(result)
            if match:
                result_dict['choice'] = match.group(1).upper()
            else:
                result_dict['choice'] = None
            analysis_match = REG_ANALYSIS.search(result)
            result_dict['analysis'] = analysis_match.group(1).strip() if analysis_match else ""
        elif self.a_type == 'blank':
            answer_idx = result.lower().find('answer:')
            result_pure = result if answer_idx == -1 else result[answer_idx:]
            match = REG_BLANK.search(result_pure)
            if match:
                result_dict['answer'] = match.group(1).strip()
                # 去除answer中的多余*号
                result_dict['answer'] = result_dict['answer'].replace('*', '').strip()
            else:
                result_dict['answer'] = None
            analysis_match = REG_ANALYSIS.search(result)
            result_dict['analysis'] = analysis_match.group(1).strip() if analysis_match else ""
        elif self.a_type == 'tf_blank':
            answer_idx = result.lower().find('answer:')
            result_pure = result if answer_idx == -1 else result[answer_idx:]
            match = REG_BLANK.search(result_pure)
            if match:
                result_dict['answer'] = match.group(1).strip()
                result_dict['answer'] = result_dict['answer'].replace('*', '').strip()
            else:
                result_dict['answer'] = None
            analysis_match = REG_ANALYSIS.search(result)
            result_dict['analysis'] = analysis_match.group(1).strip() if analysis_match else ""
        elif self.a_type == 'open_hoi':
            name_match = REG_HOI_NAME.search(result)
            box_match = REG_HOI_BOX.search(result)
            if name_match:
                result_dict['name'] = name_match.group(1).strip()
            else:
                result_dict['name'] = None
            if box_match:
                result_dict['box'] = [float(box_match.group(i)) for i in range(1, 5)]
            else:
                result_dict['box'] = [-1, -1, -1, -1]
            analysis_match = REG_ANALYSIS.search(result)
            result_dict['analysis'] = analysis_match.group(1).strip() if analysis_match else ""
        elif self.a_type == 'sequence':
            first_match = REG_SEQ_FIRST.search(result)
            second_match = REG_SEQ_SECOND.search(result)
            third_match = REG_SEQ_THIRD.search(result)
            fourth_match = REG_SEQ_FOURTH.search(result)
            if first_match:
                result_dict['first'] = first_match.group(1).upper()
            else:
                result_dict['first'] = None
            if second_match:
                result_dict['second'] = second_match.group(1).upper()
            else:
                result_dict['second'] = None
            if third_match:
                result_dict['third'] = third_match.group(1).upper()
            else:
                result_dict['third'] = None
            if fourth_match:
                result_dict['fourth'] = fourth_match.group(1).upper()
            else:
                result_dict['fourth'] = None
            result_dict["answer"] = [result_dict['first'], result_dict['second'], result_dict['third'], result_dict['fourth']]
            analysis_match = REG_ANALYSIS.search(result)
            result_dict['analysis'] = analysis_match.group(1).strip() if analysis_match else ""
        elif self.a_type == 'double_choice':
            past_match = REG_DC_PAST.search(result)
            future_match = REG_DC_FUTURE.search(result)
            if past_match:
                result_dict['past'] = past_match.group(1).upper()
            else:
                result_dict['past'] = None
            if future_match:
                result_dict['future'] = future_match.group(1).upper()
            else:
                result_dict['future'] = None
            result_dict["answer"] = [result_dict['past'], result_dict['future']]
            analysis_match = REG_ANALYSIS.search(result)
            result_dict['analysis'] = analysis_match.group(1).strip() if analysis_match else ""
        result_dict['raw'] = result
        return result_dict

    def ground_truth(self):
        return self.a_obj
    
    def qa_type(self):
        return self.q_type, self.a_type
    
    def metrics(self, result):
        m = {"path": self.path}
        if self.a_type == "bbox":
            iou = max_guess_format_iou(result['bbox'], self.a_obj, self.W, self.H)
            m['iou'] = iou
            m['analysis_len'] = len(result.get('analysis', ''))
            m['accuracy'] = iou >= 0.5
            m[f'iou_{self.qa_data["data"]["question"].get("attr_name", "default")}'] = iou
            m[f'accuracy_{self.qa_data["data"]["question"].get("attr_name", "default")}'] = iou >= 0.5
        if self.a_type == "tf_bbox":
            m["have_bbox"] = self.a_obj[0] != -1
            m["pred_have_bbox"] = result['bbox'][0] != -1
            if self.a_obj[0] != -1:
                m['iou'] = max_guess_format_iou(result['bbox'], self.a_obj, self.W, self.H)
                m['accuracy'] = m['iou'] >= 0.5
                m[f'iou_{self.qa_data["data"]["answer"].get("attr_name", "default")}'] = m['iou']
                m[f'accuracy_{self.qa_data["data"]["answer"].get("attr_name", "default")}'] = m['accuracy']
            m['tf_correct'] = result['bbox'][0] != -1
            m['analysis_len'] = len(result.get('analysis', ''))
        if self.a_type == "choice":
            m['accuracy'] = result['choice'] == self.a_obj[0]
            m['analysis_len'] = len(result.get('analysis', ''))
        if self.a_type == "blank":
            if self.qa_data["data"]["question"].get("attr_name", "") in attr_candidates:
                candidates = attr_candidates[self.qa_data["data"]["question"]["attr_name"]]
                for candidate in attr_candidates:
                    m[candidate] = 0
                m[self.qa_data["data"]["question"]["attr_name"]] = 1
                m["accuracy"], m["prob"] = is_correct(result['answer'], self.a_obj, candidates)
                m[f"accuracy_{self.qa_data['data']['question']['attr_name']}"] = m["accuracy"]
                m[self.a_obj] = 1
            m["bert_f1"], m["cos_sim"], m["kw_coverage"], m["composite_score"] = composite_score(result['answer'], self.a_obj)
            m['analysis_len'] = len(result.get('analysis', ''))
            if "accuracy" not in m:
                m['accuracy'] = m['composite_score'] >= 0.75
        if self.a_type == "tf_blank":
            m['have_answer'] = self.a_obj != "unknown"
            m["pred_have_bbox"] = result['answer'] is not None and result['answer'].lower() != "unknown"
            if self.a_obj != "unknown":
                candidates = attr_candidates.get(self.qa_data["data"]["answer"].get("attr_name", ""), [])
                if candidates:
                    for candidate in attr_candidates:
                        m[candidate] = 0
                    m[self.qa_data["data"]["answer"]["attr_name"]] = 1
                    m["accuracy"], m["prob"] = is_correct(result['answer'], self.a_obj, candidates)
                    m[f"accuracy_{self.qa_data['data']['answer']['attr_name']}"] = m["accuracy"]
                    m[self.a_obj] = 1
                m["bert_f1"], m["cos_sim"], m["kw_coverage"], m["composite_score"] = composite_score(result['answer'], self.a_obj)
                if "accuracy" not in m:
                    m['accuracy'] = m['composite_score'] >= 0.75

            m["tf_correct"] = result['answer'] is not None and result['answer'].lower() != "unknown"
            m['analysis_len'] = len(result.get('analysis', ''))
        if self.a_type == "open_hoi":
            m['iou'] = max_guess_format_iou(result['box'], self.a_obj['bbox'], self.W, self.H)
            m["accuracy"] = m["iou"] >= 0.5
            m["bert_f1"], m["cos_sim"], m["kw_coverage"], m["composite_score"] = composite_score(result['name'], self.a_obj['name'])
            m['analysis_len'] = len(result.get('analysis', ''))
            m["object_accuracy"] = m["composite_score"] >= 0.75
        if self.a_type == "sequence":
            if len(set(result['answer'])) != 4 or any(x not in ['A', 'B', 'C', 'D'] for x in result['answer']):
                m['accuracy'] = False
            else:
                m['accuracy'] = all(result["answer"][k] == self.a_obj[k] for k in range(4))
                m["tau"], m["rho"], m["ndcg"] = evaluate_ranking(result["answer"], self.a_obj)
            m['analysis_len'] = len(result.get('analysis', ''))
        if self.a_type == "double_choice":
            m["past_accuracy"] = result['past'] == self.a_obj["past"]
            m["future_accuracy"] = result['future'] == self.a_obj["future"]
            m["accuracy"] = m["past_accuracy"] and m["future_accuracy"]
            m['analysis_len'] = len(result.get('analysis', ''))
        assert len(m) > 0, "No metrics calculated"
        return m



class Benchmark:
    def __init__(self, qa_dir):
        self.qa_dir = qa_dir
        # 遍历qa_dir下的所有子目录，找到一个qa.json就初始化一个Question对象
        self.questions = []
        self.path_to_question = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]Loading questions...", total=None)
            
            for root, dirs, files in os.walk(qa_dir):
                if 'qa.json' in files:
                    # root = root.replace('\\', '/')
                    self.questions.append(Question(root))
                    self.path_to_question[root] = self.questions[-1]
                progress.update(task, advance=1)

    def run(self, model: BaseModel, concurrency=8, existing_results=None):
        if not model.concurrency() and concurrency > 1:
            concurrency = 1
        print(f"Running benchmark on model: {model.model_name} with concurrency: {concurrency}")

        results = {}
        if existing_results:
            results.update(existing_results)
            
            # 从self.path_to_question中移除已经存在于existing_results中且不为None的问题
            for path in existing_results.keys():
                if path in self.path_to_question and (existing_results[path] is not None and (len(existing_results[path]['response'].get("raw", "")) > 0)):
                    del self.path_to_question[path]
            print(f"Loaded {len(existing_results)} existing results, continuing with {len(self.path_to_question)} new questions.")

        save_path = os.path.join("results", f'results_{model.model_name}.json')
        os.makedirs("results", exist_ok=True)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_path = {executor.submit(q.solve, model): path for path, q in self.path_to_question.items()}
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("[green]Processing...", total=len(future_to_path))
                for future in (as_completed(future_to_path)):
                    path = future_to_path[future]
                    try:
                        result = {}
                        result['response'] = future.result()
                        print(f"[blue]Processed {path}: {result['response']}")
                        result['ground_truth'] = self.path_to_question[path].ground_truth()
                        result['q_type'], result['a_type'] = self.path_to_question[path].qa_type()
                        results[path] = result
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Error processing {path}: {e}")
                        results[path] = None
                    progress.update(task, advance=1)
                    if len(results) % 500 == 0:
                        with open(save_path, 'w') as f:
                            json.dump(results, f, indent=4)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        return results
    def export_question_batch(self, model_name, batch_size):
        img_cnt = 0
        os.makedirs(os.path.join("batch", model_name, f"images"), exist_ok=True)
        for batch_no in range((len(self.questions) + batch_size - 1) // batch_size):
            batch_questions = self.questions[batch_no*batch_size:(batch_no+1)*batch_size]
            export_data = []
            for q in batch_questions:
                img = q.get_image()
                image_path = os.path.join("batch", model_name, f"images/{img_cnt}.png")
                img_cnt += 1
                img.save(image_path, format="PNG")
                
                item = {
                    "custom_id": q.id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": image_path}},
                                {"type": "text", "text": q.q_text}
                            ]}
                        ],
                    }
                }
                
                export_data.append(item)
            save_path = os.path.join("batch", model_name, f'batch_{model_name}_{batch_no}.jsonl')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                for item in export_data:
                    f.write(json.dumps(item) + '\n')
            print(f"Exported batch {batch_no} to {save_path}")
    def metrics(self, result_json_path):

        with open(result_json_path, 'r', encoding="utf-8") as f:
            results = json.load(f)
        for result_key in list(results.keys()):
            if "\\" in result_key:
                new_key = result_key.replace("\\", "/")
                results[new_key] = results.pop(result_key)

        cate_scores = {}
        cate_metric_score_sum = {}
        cate_metric_count = {}
        skipped = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]Calculating metrics...", total=len(self.path_to_question))
            for q_path, question in self.path_to_question.items():
                q_path = q_path.replace("\\", "/")
                if q_path not in results:
                    raise ValueError(f"Question path {q_path} not found in results")
                result = results[q_path]
                if result is None:
                    print(f"Skipping {q_path} due to None result")
                    skipped.append(q_path)
                    continue
                if question.q_type not in cate_scores:
                    cate_scores[question.q_type] = []
                if question.q_type not in cate_metric_score_sum:
                    cate_metric_score_sum[question.q_type] = {}
                    cate_metric_count[question.q_type] = {}
                metrics = question.metrics(result["response"])
                cate_scores[question.q_type].append(metrics)
                for k, v in metrics.items():
                    if k not in cate_metric_score_sum[question.q_type]:
                        cate_metric_score_sum[question.q_type][k] = 0
                        cate_metric_count[question.q_type][k] = 0
                    if isinstance(v, (int, float, bool)):
                        cate_metric_score_sum[question.q_type][k] += v
                        cate_metric_count[question.q_type][k] += 1
                progress.update(task, advance=1)
        cate_metric_avg = {}
        for cate, metric_sum in cate_metric_score_sum.items():
            cate_metric_avg[cate] = {}
            for k, v in metric_sum.items():
                if cate_metric_count[cate][k] > 0:
                    cate_metric_avg[cate][k] = v / cate_metric_count[cate][k]
                else:
                    cate_metric_avg[cate][k] = None
        # avg导出txt
        save_path = result_json_path.replace('.json', '_metrics.txt')
        with open(save_path, 'w') as f:
            for cate, metrics in cate_metric_avg.items():
                f.write(f"Category: {cate}\n")
                for k, v in metrics.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
        print(f"Metrics saved to {save_path}")
        # scores导出json
        save_path = result_json_path.replace('.json', '_detailed_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(cate_scores, f, indent=4)
        print(f"Detailed metrics saved to {save_path}")

        if skipped:
            print(f"Skipped {len(skipped)} questions due to None results:")
            for s in skipped:
                print(f"  {s}")


            
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmarking script for MLLM models")
    parser.add_argument('--qa_dir', type=str, default='./final_qa', help='Directory containing QA pairs')
    parser.add_argument('--model_name', type=str, help='Name of the model to benchmark')
    parser.add_argument('--concurrency', type=int, default=8, help='Number of concurrent threads')
    parser.add_argument('--model_params', type=str, default='', help='Additional model parameters in JSON format')
    parser.add_argument('--export_batch', type=int, default=0, help='Batch size for batch prediction (if supported), will give jsonl file as output, provide it to batching apis')
    parser.add_argument('--continuing', action='store_true', help='Continue from existing results file if exists')
    parser.add_argument('--calc_metrics', type=str, help='Calculate metrics from existing results file if exists')
    args = parser.parse_args()

    if args.calc_metrics:
        benchmark = Benchmark(args.qa_dir)
        benchmark.metrics(args.calc_metrics)
        exit(0)

    if args.export_batch > 0:
        benchmark = Benchmark(args.qa_dir)
        benchmark.export_question_batch(args.model_name, args.export_batch)
        exit(0)
    
    if args.model_name not in MODEL_NAME_MAP:
        print(f"Model {args.model_name} not recognized. Available models: {list(MODEL_NAME_MAP.keys())}")
        exit(1)

    
    ModelClass = MODEL_NAME_MAP[args.model_name]
    if args.model_params:
        print(f"Using additional model parameters: {args.model_params}")
        model_params = json.loads(args.model_params)
        model = ModelClass(args.model_name, **model_params)
    else:
        model = ModelClass(args.model_name)

    existing_results = None
    if args.continuing and os.path.exists(os.path.join("results", f'results_{model.model_name}.json')):
        print(f"Continuing from existing results file for model {model.model_name}")
        existing_results_path = os.path.join("results", f'results_{model.model_name}.json')
        with open(existing_results_path, 'r') as f:
            existing_results = json.load(f)
        
    benchmark = Benchmark(args.qa_dir)
    benchmark.run(model, concurrency=args.concurrency, existing_results=existing_results)

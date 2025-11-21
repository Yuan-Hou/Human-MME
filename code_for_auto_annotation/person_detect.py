import json
import os
import sys
import shutil
import argparse
import cv2
import numpy as np
import torch
from copy import deepcopy
from rich.progress import Progress
from ultralytics import YOLO

from utils import key_points_to_bounding_box, bounding_box_iou

sys.path.append("./repos/DWPose/ControlNet-v1-1-nightly")
from annotator.dwpose import DWposeDetector, Wholebody, draw_pose

DEBUG = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt
face_model = YOLO("yolov11m-face.pt")

# https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
body_model = YOLO("yolo11x.pt")

pose_predictor = Wholebody()

def detect_pose(image):
    """
    检测图片中的人体骨骼
    Args:
        img: 输入图片
    Returns:
        骨骼检测结果
    """
    H, W, C = image.shape
    with torch.no_grad():
        candidate, subset = pose_predictor(image)
        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:,:18].copy()
        body = body.reshape(nums*18, locs)
        score = subset[:,:18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.2:
                    score[i][j] = int(18*i+j)
                else:
                    score[i][j] = -1

        un_visible = subset<0.3
        candidate[un_visible] = -1

        foot = candidate[:,18:24]

        faces = candidate[:,24:92]

        hands = candidate[:,92:113]
        hands = np.vstack([hands, candidate[:,113:]])
        
        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces, foot=foot)

        
        persons = []
        for i in range(nums):
            if candidate[i,:18].max() == -1: # body部分全部被遮挡就不算人了
                continue
            persons.append({
                "dw_body": candidate[i,:18].reshape(18, 2),
                "dw_hand_1": candidate[i,92:113].reshape(21, 2),
                "dw_hand_2": candidate[i,113:].reshape(21, 2),
                "dw_face": candidate[i,24:92].reshape(68, 2),
                "dw_foot_1": candidate[i,18:21].reshape(3, 2),
                "dw_foot_2": candidate[i,21:24].reshape(3, 2)
            })
        return persons

def detect_person(image_metas, output_dir=None, resume=False):
    """
    检测图片元数据中的人体与人脸信息
    Args:
        image_metas: 图片元数据列表
        output_dir: 输出目录，用于保存进度
        resume: 是否从之前的进度继续
    Returns:
        符合条件的图片元数据列表
    """
    result = []
    image_metas = deepcopy(image_metas)
    
    # 进度文件路径
    progress_file = None
    processed_files = set()
    
    if output_dir:
        progress_file = os.path.join(output_dir, "detection_progress.json")
        
        # 如果启用resume且进度文件存在，加载已处理的文件列表
        if resume and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    processed_files = set(progress_data.get('processed_files', []))
                    result = progress_data.get('results', [])
                print(f"Resume from previous progress: {len(processed_files)} files already processed")
            except Exception as e:
                print(f"Warning: Failed to load progress file: {e}")
                processed_files = set()
                result = []
    
    with Progress() as progress:
        task = progress.add_task("Detecting persons...", total=len(image_metas))
        progress.update(task, completed=len(processed_files))
        
        for i, meta in enumerate(image_metas):
            if not os.path.exists(meta['image_path']):
                meta['image_path'] = os.path.join("./ref_datasets/hico_det", meta['image_path'])
            # 跳过已处理的文件
            if meta['image_path'] in processed_files:
                continue
            # 跳过已处理的文件
            if meta['image_path'] in processed_files:
                continue
                
            progress.update(task, description=f"Processing {i+1}/{len(image_metas)}: {meta['image_path']}")
            img = cv2.imread(meta['image_path'])
            # 保持原始比例缩放图片尺寸到1k
            if img is not None:
                w, h = img.shape[1], img.shape[0]
                if w > 1000 or h > 1000:
                    scale = min(1000 / w, 1000 / h)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            

            W, H = img.shape[1], img.shape[0]
            if img is None:
                print(f"Warning: Image not found {meta['image_path']}")
                # 标记为已处理（即使失败）
                processed_files.add(meta['image_path'])
                continue
        
            body_boxes = []
            face_boxes = []
            skeletons = []
            persons = []
            detect_results = {
                'body_boxes': body_boxes,
                'face_boxes': face_boxes,
                'skeletons': skeletons,
            }

            # 检测人体
            body_results = body_model(img, verbose=False)
            body_detections = body_results[0].boxes.data.cpu().numpy()
            for box in body_detections:
                x1, y1, x2, y2, conf, cls = box
                if cls != 0:  # 确保是人体类别
                    continue
                if conf > 0.2:
                    body_boxes.append((x1 / W, y1 / H, x2 / W, y2 / H))
            if DEBUG:
                display_img = img.copy()
                for box in body_detections:
                    x1, y1, x2, y2, conf, cls = box
                    cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.imshow("Body Detection", display_img)
                cv2.waitKey(0)

            # 检测人脸
            face_results = face_model(img, verbose=False)
            face_detections = face_results[0].boxes.data.cpu().numpy()
            for box in face_detections:
                x1, y1, x2, y2, conf, cls = box
                if cls != 0:  # 确保是人脸类别
                    continue
                if conf > 0.2:
                    face_boxes.append((x1 / W, y1 / H, x2 / W, y2 / H))
            if DEBUG:
                display_img = img.copy()
                for box in face_detections:
                    x1, y1, x2, y2, conf, cls = box
                    cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.imshow("Face Detection", display_img)
                cv2.waitKey(0)

            # 检测骨骼姿态
            skeletons.extend(detect_pose(img))

            # 利用姿态信息获得整体标注框
            dw_person_boxes = []
            for skeleton in skeletons:
                person_key_points = np.concatenate(list(skeleton.values()), axis=0)
                dw_person_boxes.append(key_points_to_bounding_box(person_key_points))

            # 与检测到的bounding box对比计算IoU得到混淆矩阵
            iou_matrix = np.zeros((len(dw_person_boxes), len(body_boxes)))
            for i, dw_box in enumerate(dw_person_boxes):
                for j, body_box in enumerate(body_boxes):
                    iou_matrix[i, j] = bounding_box_iou(dw_box, body_box)

            # IoU大于0.3的认为是正确匹配，取最大IoU的作为正确匹配
            matched_pairs = set()
            for i in range(len(dw_person_boxes)):
                max_iou = 0
                max_j = -1
                for j in range(len(body_boxes)):
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_j = j
                if max_iou > 0.3 and max_j != -1:
                    matched_pairs.add((i, max_j))
                elif max_iou <= 0.3 and DEBUG:
                    print(f"Warning: Person {i} in {meta['image_path']} has no matching body box, IoU: {max_iou}, matched with box {max_j}")

            types = [] # 图片的类型，face：只要有人脸即可（合格的人脸不需要匹配到人体）；person：需要有合格的人体（合格的人体不需要匹配到人脸）
            # 匹配成功才记作一个person
            for i, j in matched_pairs:
                persons.append({
                    'body_box': j,
                    'skeleton': i,
                })
            # 尽可能将人体和人脸匹配起来
            dw_person_face_boxes = []
            for i, person in enumerate(persons):
                skeleton = skeletons[person['skeleton']]
                face_key_points = skeleton['dw_face']
                dw_person_face_boxes.append(key_points_to_bounding_box(face_key_points))
            face_iou_matrix = np.zeros((len(dw_person_face_boxes), len(face_boxes)))
            for i, dw_box in enumerate(dw_person_face_boxes):
                for j, face_box in enumerate(face_boxes):
                    face_iou_matrix[i, j] = bounding_box_iou(dw_box, face_box)
            # IoU大于0.3的认为是正确匹配，取最大IoU的作为正确匹配
            matched_indices = np.where(face_iou_matrix > 0.3)
            for i in range(len(dw_person_face_boxes)):
                if i in matched_indices[0]:
                    max_j = matched_indices[1][np.argmax(face_iou_matrix[i, matched_indices[1]])]
                    persons[i]['face_box'] = max_j
                else:
                    persons[i]['face_box'] = None

            # 如果匹配成功的人体数量与检测框/骨骼数量一致，则认为是合格的人体图片
            if len(persons) > 0 and len(persons) == len(skeletons) and len(persons) == len(body_boxes):
                types.append('person')
            elif DEBUG:
                # 不合格的打印一下不合格的原因
                if len(persons) == 0:
                    print(f"Warning: No persons detected in {meta['image_path']}")
                elif len(persons) != len(skeletons):
                    print(f"Warning: Mismatched persons and skeletons in {meta['image_path']}, skeletons: {len(skeletons)}, persons: {len(persons)}, body_boxes: {len(body_boxes)}")
                elif len(persons) != len(body_boxes):
                    print(f"Warning: Mismatched persons and body boxes in {meta['image_path']}, skeletons: {len(skeletons)}, persons: {len(persons)}, body_boxes: {len(body_boxes)}")
            
            # 为没有匹配到person的face_box创建person对象
            for j, face_box in enumerate(face_boxes):
                if j not in matched_indices[1]:
                    persons.append({
                        'body_box': None,
                        'skeleton': None,
                        'face_box': j,
                    })
                    if DEBUG:
                        print(f"Warning: Face box {j} in {meta['image_path']} has no matching person, creating dummy person entry")

            # 如果有检测到人脸，并且原先meta中"detected_types"存在且包含'face'，则认为是合格的人脸图片
            if len(face_boxes) > 0 and ('detected_types' in meta and 'face' in meta['detected_types']):
                types.append('face')
            
            # 移除原有的类型信息
            meta.pop("detected_types", None)
            meta.pop("type", None) 

            meta['types'] = types
            meta['persons'] = persons
            meta['detect_results'] = detect_results

            if len(types) > 0:
                result.append(meta)
            
            # 标记文件为已处理
            processed_files.add(meta['image_path'])
            progress.advance(task)
            
            # 每处理50个文件保存一次进度
            if progress_file and len(processed_files) % 50 == 0:
                try:
                    progress_data = {
                        'processed_files': list(processed_files),
                        'results': result
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f, indent=2, cls=NumpyEncoder)
                    print(f"Detected {len(result)} images with matched persons")
                    print(f"Count of type 'person': {sum('person' in meta.get('types', []) for meta in result)}")
                    print(f"Count of type 'face': {sum('face' in meta.get('types', []) for meta in result)}")
                    print(f"Count of persons: {sum(len(meta.get('persons', [])) for meta in result)}")
                    print(f"Count of images with more than one person: {sum(len(meta.get('persons', [])) > 1 for meta in result)}")
                except Exception as e:
                    print(f"Warning: Failed to save progress: {e}")
    
    # 保存最终进度
    if progress_file:
        try:
            progress_data = {
                'processed_files': list(processed_files),
                'results': result
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            print(f"Warning: Failed to save final progress: {e}")
            
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检测图片中的人脸、人体和骨骼")
    parser.add_argument('--input_dir', type=str, help='包含去重结果的目录', default='./ref_datasets/extracted_frames')
    parser.add_argument('--output_dir', type=str, help='输出目录', default='./ref_datasets/person_detected')
    parser.add_argument('--clean_output', action='store_true', help='清理输出目录')
    parser.add_argument('--resume', action='store_true', help='从之前的进度继续处理')
    
    args = parser.parse_args()
    
    if args.clean_output and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"Cleaned output directory: {args.output_dir}")
    elif args.resume and not os.path.exists(args.output_dir):
        print(f"Warning: Cannot resume - output directory {args.output_dir} does not exist")
        args.resume = False

    os.makedirs(args.output_dir, exist_ok=True)

    dedup_metas_path = os.path.join(args.input_dir, "deduplicated_image_metas.json")
    if not os.path.exists(dedup_metas_path):
        print(f"Error: {dedup_metas_path} not found!")
        print("Please run deduplicate_images.py first.")
        exit(1)
    
    with open(dedup_metas_path, 'r') as f:
        kept_images = json.load(f)["images"]

    print(f"Found {len(kept_images)} deduplicated images")
    person_images = detect_person(kept_images, args.output_dir, args.resume)
    with open(os.path.join(args.output_dir, "person_detected_metas_before_copy.json") , 'w') as f:
        json.dump(person_images, f, indent=4, cls=NumpyEncoder) 
    
    if DEBUG:
        exit(0) 
    copied_count = 0
    error_count = 0
    
    # 检查是否需要跳过复制步骤（如果使用resume且已经有最终结果文件）
    final_meta_file = os.path.join(args.output_dir, "person_detected_metas.json")
    copy_progress_file = os.path.join(args.output_dir, "copy_progress.json")
    copied_files = set()
    
    if args.resume:
        # 如果最终文件已存在，询问是否跳过复制
        if os.path.exists(final_meta_file):
            print(f"Final result file {final_meta_file} already exists.")
            response = input("Skip copying step? (y/n): ").lower().strip()
            if response == 'y':
                print("Skipping copy step as requested.")
                exit(0)
        
        # 加载复制进度
        if os.path.exists(copy_progress_file):
            try:
                with open(copy_progress_file, 'r') as f:
                    copy_progress_data = json.load(f)
                    copied_files = set(copy_progress_data.get('copied_files', []))
                    copied_count = len(copied_files)
                print(f"Resume copying: {copied_count} files already copied")
            except Exception as e:
                print(f"Warning: Failed to load copy progress: {e}")
                copied_files = set()
    
    with Progress() as progress:
        task = progress.add_task("Copying images...", total=len(person_images))
        progress.update(task, completed=copied_count)
        
        for meta in person_images:
            src_path = meta['image_path']
            filename = os.path.basename(src_path)
            output_path = os.path.join(args.output_dir, filename)
            
            # 跳过已复制的文件
            if src_path in copied_files:
                continue
                
            try:
                if not os.path.exists(src_path):
                    print(f"Warning: Source image not found: {src_path}")
                    error_count += 1
                    copied_files.add(src_path)  # 标记为已处理（即使失败）
                    progress.advance(task)
                    continue
                
                shutil.copy(src_path, output_path)
                meta['image_path'] = output_path  # 更新路径到输出目录
                copied_files.add(src_path)
                copied_count += 1
                
                # 每复制500个文件保存一次进度
                if len(copied_files) % 500 == 0:
                    try:
                        copy_progress_data = {'copied_files': list(copied_files)}
                        with open(copy_progress_file, 'w') as f:
                            json.dump(copy_progress_data, f, indent=2)
                    except Exception as e:
                        print(f"Warning: Failed to save copy progress: {e}")
                        
            except Exception as e:
                print(f"Error copying {src_path} to {output_path}: {e}")
                error_count += 1
                copied_files.add(src_path)  # 标记为已处理（即使失败）
            finally:
                progress.advance(task)

    print(f"Copied {copied_count} images to {args.output_dir}")
    if error_count > 0:
        print(f"Finished with {error_count} errors.")

    # 保存最终复制进度
    try:
        copy_progress_data = {'copied_files': list(copied_files)}
        with open(copy_progress_file, 'w') as f:
            json.dump(copy_progress_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save final copy progress: {e}")

    with open(final_meta_file, 'w') as f:
        json.dump(person_images, f, indent=4, cls=NumpyEncoder)
    
    # 清理进度文件（可选）
    try:
        if os.path.exists(os.path.join(args.output_dir, "detection_progress.json")):
            os.remove(os.path.join(args.output_dir, "detection_progress.json"))
        if os.path.exists(copy_progress_file):
            os.remove(copy_progress_file)
        print("Cleaned up progress files.")
    except Exception as e:
        print(f"Warning: Failed to clean up progress files: {e}")
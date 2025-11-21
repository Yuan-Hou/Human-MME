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
    ..........
    Args:
        img: ....
    Returns:
        ......
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
            if candidate[i,:18].max() == -1: # body............
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
    ................
    Args:
        image_metas: .......
        output_dir: ....，......
        resume: ..........
    Returns:
        ............
    """
    result = []
    image_metas = deepcopy(image_metas)
    
    # ......
    progress_file = None
    processed_files = set()
    
    if output_dir:
        progress_file = os.path.join(output_dir, "detection_progress.json")
        
        # ....resume.......，..........
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
            # ........
            if meta['image_path'] in processed_files:
                continue
            # ........
            if meta['image_path'] in processed_files:
                continue
                
            progress.update(task, description=f"Processing {i+1}/{len(image_metas)}: {meta['image_path']}")
            img = cv2.imread(meta['image_path'])
            # .............1k
            if img is not None:
                w, h = img.shape[1], img.shape[0]
                if w > 1000 or h > 1000:
                    scale = min(1000 / w, 1000 / h)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            

            W, H = img.shape[1], img.shape[0]
            if img is None:
                print(f"Warning: Image not found {meta['image_path']}")
                # ......（....）
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

            # ....
            body_results = body_model(img, verbose=False)
            body_detections = body_results[0].boxes.data.cpu().numpy()
            for box in body_detections:
                x1, y1, x2, y2, conf, cls = box
                if cls != 0:  # .......
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

            # ....
            face_results = face_model(img, verbose=False)
            face_detections = face_results[0].boxes.data.cpu().numpy()
            for box in face_detections:
                x1, y1, x2, y2, conf, cls = box
                if cls != 0:  # .......
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

            # ......
            skeletons.extend(detect_pose(img))

            # .............
            dw_person_boxes = []
            for skeleton in skeletons:
                person_key_points = np.concatenate(list(skeleton.values()), axis=0)
                dw_person_boxes.append(key_points_to_bounding_box(person_key_points))

            # .....bounding box....IoU......
            iou_matrix = np.zeros((len(dw_person_boxes), len(body_boxes)))
            for i, dw_box in enumerate(dw_person_boxes):
                for j, body_box in enumerate(body_boxes):
                    iou_matrix[i, j] = bounding_box_iou(dw_box, body_box)

            # IoU..0.3........，...IoU.......
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

            types = [] # .....，face：.......（.............）；person：........（.............）
            # .........person
            for i, j in matched_pairs:
                persons.append({
                    'body_box': j,
                    'skeleton': i,
                })
            # .............
            dw_person_face_boxes = []
            for i, person in enumerate(persons):
                skeleton = skeletons[person['skeleton']]
                face_key_points = skeleton['dw_face']
                dw_person_face_boxes.append(key_points_to_bounding_box(face_key_points))
            face_iou_matrix = np.zeros((len(dw_person_face_boxes), len(face_boxes)))
            for i, dw_box in enumerate(dw_person_face_boxes):
                for j, face_box in enumerate(face_boxes):
                    face_iou_matrix[i, j] = bounding_box_iou(dw_box, face_box)
            # IoU..0.3........，...IoU.......
            matched_indices = np.where(face_iou_matrix > 0.3)
            for i in range(len(dw_person_face_boxes)):
                if i in matched_indices[0]:
                    max_j = matched_indices[1][np.argmax(face_iou_matrix[i, matched_indices[1]])]
                    persons[i]['face_box'] = max_j
                else:
                    persons[i]['face_box'] = None

            # .............../......，...........
            if len(persons) > 0 and len(persons) == len(skeletons) and len(persons) == len(body_boxes):
                types.append('person')
            elif DEBUG:
                # ..............
                if len(persons) == 0:
                    print(f"Warning: No persons detected in {meta['image_path']}")
                elif len(persons) != len(skeletons):
                    print(f"Warning: Mismatched persons and skeletons in {meta['image_path']}, skeletons: {len(skeletons)}, persons: {len(persons)}, body_boxes: {len(body_boxes)}")
                elif len(persons) != len(body_boxes):
                    print(f"Warning: Mismatched persons and body boxes in {meta['image_path']}, skeletons: {len(skeletons)}, persons: {len(persons)}, body_boxes: {len(body_boxes)}")
            
            # ......person.face_box..person..
            for j, face_box in enumerate(face_boxes):
                if j not in matched_indices[1]:
                    persons.append({
                        'body_box': None,
                        'skeleton': None,
                        'face_box': j,
                    })
                    if DEBUG:
                        print(f"Warning: Face box {j} in {meta['image_path']} has no matching person, creating dummy person entry")

            # ........，....meta."detected_types".....'face'，...........
            if len(face_boxes) > 0 and ('detected_types' in meta and 'face' in meta['detected_types']):
                types.append('face')
            
            # .........
            meta.pop("detected_types", None)
            meta.pop("type", None) 

            meta['types'] = types
            meta['persons'] = persons
            meta['detect_results'] = detect_results

            if len(types) > 0:
                result.append(meta)
            
            # ........
            processed_files.add(meta['image_path'])
            progress.advance(task)
            
            # ...50.........
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
    
    # ......
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
    parser = argparse.ArgumentParser(description="........、.....")
    parser.add_argument('--input_dir', type=str, help='.........', default='./ref_datasets/extracted_frames')
    parser.add_argument('--output_dir', type=str, help='....', default='./ref_datasets/person_detected')
    parser.add_argument('--clean_output', action='store_true', help='......')
    parser.add_argument('--resume', action='store_true', help='..........')
    
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
    
    # ............（....resume..........）
    final_meta_file = os.path.join(args.output_dir, "person_detected_metas.json")
    copy_progress_file = os.path.join(args.output_dir, "copy_progress.json")
    copied_files = set()
    
    if args.resume:
        # .........，........
        if os.path.exists(final_meta_file):
            print(f"Final result file {final_meta_file} already exists.")
            response = input("Skip copying step? (y/n): ").lower().strip()
            if response == 'y':
                print("Skipping copy step as requested.")
                exit(0)
        
        # ......
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
            
            # ........
            if src_path in copied_files:
                continue
                
            try:
                if not os.path.exists(src_path):
                    print(f"Warning: Source image not found: {src_path}")
                    error_count += 1
                    copied_files.add(src_path)  # ......（....）
                    progress.advance(task)
                    continue
                
                shutil.copy(src_path, output_path)
                meta['image_path'] = output_path  # .........
                copied_files.add(src_path)
                copied_count += 1
                
                # ...500.........
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
                copied_files.add(src_path)  # ......（....）
            finally:
                progress.advance(task)

    print(f"Copied {copied_count} images to {args.output_dir}")
    if error_count > 0:
        print(f"Finished with {error_count} errors.")

    # ........
    try:
        copy_progress_data = {'copied_files': list(copied_files)}
        with open(copy_progress_file, 'w') as f:
            json.dump(copy_progress_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save final copy progress: {e}")

    with open(final_meta_file, 'w') as f:
        json.dump(person_images, f, indent=4, cls=NumpyEncoder)
    
    # ......（..）
    try:
        if os.path.exists(os.path.join(args.output_dir, "detection_progress.json")):
            os.remove(os.path.join(args.output_dir, "detection_progress.json"))
        if os.path.exists(copy_progress_file):
            os.remove(copy_progress_file)
        print("Cleaned up progress files.")
    except Exception as e:
        print(f"Warning: Failed to clean up progress files: {e}")
# pip install sentence-transformers bert-score nltk
from bert_score import score
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from functools import lru_cache

nltk.download('punkt')


attr_candidates = {
    "age": ['child', 'teenager', 'adult', 'senior'],
    "gender": ['male', 'female'],
    "emotion": ['happy', 'sad', 'angry', 'surprised', 'neutral'],
    "race": ['white', 'black', 'asian', 'middle eastern', 'latino hispanic'],
    "pitch": ['up', 'down'],
    "yaw": ['left side', 'right side'],
}

keywords = {
    "child": ["child", "kid", "little boy", "little girl", "baby"],
    "teenager": ["teen", "teenager", "adolescent", "young teen"],
    "adult": ["adult", "man", "woman", "young adult"],
    "senior": ["elderly", "senior", "old man", "old woman"],
    "male": ["man", "boy", "gentleman", "male"],
    "female": ["woman", "girl", "lady", "female"],
    "happy": ["happy", "smiling", "joyful", "cheerful"],
    "sad": ["sad", "unhappy", "sorrowful", "downcast"],
    "angry": ["angry", "mad", "furious", "irate"],
    "surprised": ["surprised", "astonished", "amazed", "startled"],
    "neutral": ["neutral", "expressionless", "impassive", "blank"], 
    "white": ["white", "caucasian", "european"],
    "black": ["black", "african", "afro-american"],
    "asian": ["asian", "chinese", "japanese", "korean"],
    "middle eastern": ["middle eastern", "arab", "persian"],
    "latino hispanic": ["latino", "hispanic", "mexican", "spanish"],
    "up": ["up", "looking up", "head up", "face up"],
    "down": ["down", "looking down", "head down", "face down"],
    "left side": ["left side", "looking left", "head turned left", "face left"],
    "right side": ["right side", "looking right", "head turned right", "face right"],
}

model = SentenceTransformer('all-MiniLM-L6-v2')

# .. BERTScore ....，.... 1000 ...
@lru_cache(maxsize=1000)
def cached_bert_score(candidate_text, gt_text):
    """
    .. BERTScore ....
    """
    P, R, F1 = score([candidate_text], [gt_text], lang='en', model_type='roberta-large', verbose=False)
    return F1.item()

def semantic_mapping(prediction, candidate_text):
    """
    ...........（soft score）.....
    """
    pred_emb = model.encode(prediction, convert_to_tensor=True)
    scores = []
    for cat in candidate_text:
        kw_embs = model.encode(keywords[cat], convert_to_tensor=True)
        sim_scores = util.cos_sim(pred_emb, kw_embs)
        scores.append(sim_scores.max().item())
    scores = np.array(scores)
    pred_cat = candidate_text[scores.argmax()]
    return pred_cat, scores[scores.argmax()]

def is_correct(prediction, ground_truth, candidate_text):
    if candidate_text is None or prediction is None or len(candidate_text) == 0:
        return False, 0.0
    pred, prob = semantic_mapping(prediction, candidate_text)
    ground, _ = semantic_mapping(ground_truth, candidate_text)
    return pred == ground, prob

def composite_score(candidate_text, ground_truth,
                    w_bert=0.5, w_cos=0.3, w_kw=0.2):
    if candidate_text is None or candidate_text.strip() == "":
        return 0.0, 0.0, 0.0, 0.0
    # . ground truth ....
    gt_text = ground_truth
    
    # BERTScore (....)
    bert_f1 = cached_bert_score(candidate_text, gt_text)
    
    # Sentence embedding cosine
    cand_emb = model.encode(candidate_text, convert_to_tensor=True)
    gt_emb = model.encode(gt_text, convert_to_tensor=True)
    cos_sim = util.cos_sim(cand_emb, gt_emb).item()
    
    # Keyword coverage
    gt_tokens = word_tokenize(gt_text.lower())
    cand_tokens = word_tokenize(candidate_text.lower())
    matched = sum(1 for tok in gt_tokens if tok in cand_tokens)
    kw_coverage = matched / len(gt_tokens)
    
    # ....
    final_score = w_bert * bert_f1 + w_cos * cos_sim + w_kw * kw_coverage
    return bert_f1, cos_sim, kw_coverage, final_score


def bounding_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 < x2 and y1 < y2:
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    return 0

def max_guess_format_iou(guess, norm_gt_box, W, H):
    if guess is None or len(guess) != 4 or any(k is None for k in guess):
        return 0.0
    # .....
    if all(0 <= k <= 1 for k in guess):
        return bounding_box_iou(guess, norm_gt_box)
    # ....
    candidates = [bounding_box_iou((guess[0] / W, guess[1] / H, guess[2] / W, guess[3] / H), norm_gt_box)]
    # ....1080p..
    if max([W, H]) > 1920 or min([W, H]) > 1080:
        scale = min(1920 / max(W, H), 1080 / min(W, H))
        candidates.append(bounding_box_iou((guess[0] / (W * scale), guess[1] / (H * scale), guess[2] / (W * scale), guess[3] / (H * scale)), norm_gt_box))
    # 1000...
    if all(0 <= k <= 1000 for k in guess):
        candidates.append(bounding_box_iou([k / 1000 for k in guess], norm_gt_box))
    return max(candidates)

from scipy.stats import kendalltau, spearmanr
import numpy as np

def evaluate_ranking(pred, gt):
    if pred is None:
        return 0.0, 0.0, 0.0
    # ..........
    gt_pos = {item: i for i, item in enumerate(gt)}
    pred_pos = {item: i for i, item in enumerate(pred)}

    # .......
    gt_ranks = [gt_pos[item] for item in gt]
    pred_ranks = [pred_pos[item] for item in gt]

    # 1. Kendall's tau
    tau, _ = kendalltau(gt_ranks, pred_ranks)
    tau_score = (tau + 1) / 2  # ... [0,1]

    # 2. Spearman's rho
    rho, _ = spearmanr(gt_ranks, pred_ranks)
    rho_score = (rho + 1) / 2  # ... [0,1]

    # 3. nDCG@4
    n = len(gt)
    # .....：.....，... relevance = n - rank
    rel = {item: n - i for i, item in enumerate(gt)}

    def dcg(order):
        return sum((rel[item] / np.log2(idx + 2)) for idx, item in enumerate(order))

    dcg_pred = dcg(pred)
    dcg_gt = dcg(gt)
    ndcg = dcg_pred / dcg_gt if dcg_gt > 0 else 0.0

    return tau_score, rho_score, ndcg


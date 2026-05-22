import json
import re
from rapidfuzz import fuzz
from dateutil import parser as date_parser

def normalize_text(text):
    """Gentle normalization for names and addresses."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text).strip()
    return text

def normalize_strict(text):
    """Fallback strict normalizer."""
    if not text: return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def compare_amounts(gt, pred):
    """Semantic comparison for financial amounts."""
    if not gt and not pred: return 100.0
    if not gt or not pred: return 0.0
    
    try:
        # Strip everything except numbers and the decimal point
        val_gt = float(re.sub(r'[^\d.]', '', str(gt)))
        val_pred = float(re.sub(r'[^\d.]', '', str(pred)))
        return 100.0 if val_gt == val_pred else 0.0
    except ValueError:
        # Fallback to strict string match if it's garbled text instead of a number
        return 100.0 if normalize_strict(gt) == normalize_strict(pred) else 0.0

def compare_dates(gt, pred):
    """Semantic comparison for dates."""
    if not gt and not pred: return 100.0
    if not gt or not pred: return 0.0
    
    try:
        # fuzzy=True tells the parser to ignore random surrounding text
        dt_gt = date_parser.parse(str(gt), fuzzy=True).date()
        dt_pred = date_parser.parse(str(pred), fuzzy=True).date()
        return 100.0 if dt_gt == dt_pred else 0.0
    except Exception:
        # Fallback to strict string match if date parsing fails entirely
        return 100.0 if normalize_strict(gt) == normalize_strict(pred) else 0.0

def main():
    report_path = "results/eval_report.json"
    
    try:
        with open(report_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Could not find {report_path}.")
        return

    scores = {
        "vendor_name": [],
        "vendor_address": [],
        "amount_total_gross": [],
        "date_issue": []
    }

    print(f"Scoring {len(data)} documents with Semantic Matching...\n")

    for item in data:
        gt = item.get("ground_truth", {})
        pred = item.get("predicted", {})
        
        # 1. Evaluate Text Fields (Fuzzy Matching)
        for field in ["vendor_name", "vendor_address"]:
            gt_val = normalize_text(gt.get(field))
            pred_val = normalize_text(pred.get(field))
            
            if not gt_val and not pred_val:
                continue 
                
            score = fuzz.token_set_ratio(pred_val, gt_val)
            scores[field].append(score)

        # 2. Evaluate Amounts (Float Matching)
        amt_score = compare_amounts(gt.get("amount_total_gross"), pred.get("amount_total_gross"))
        scores["amount_total_gross"].append(amt_score)
        
        # 3. Evaluate Dates (Datetime Matching)
        date_score = compare_dates(gt.get("date_issue"), pred.get("date_issue"))
        scores["date_issue"].append(date_score)

    total_avg = []
    # Calculate and Print the Final Metrics
    print("====== TRUE EXTRACTION ACCURACY ======")
    for field, score_list in scores.items():
        if not score_list:
            print(f"{field:<22}: N/A")
            continue
            
        avg_score = sum(score_list) / len(score_list)
        print(f"{field:<22}: {avg_score:.2f}%")
        total_avg.append(avg_score)
    print("======================================")
    # get the average of all the averages
    total_avg = sum(total_avg) / len(total_avg)
    print(f"Total average: {total_avg:.2f}%")
    print("======================================")

if __name__ == "__main__":
    main()
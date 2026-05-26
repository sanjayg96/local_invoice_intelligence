

import json
import re
from rapidfuzz import fuzz
from dateutil import parser as date_parser
import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="dateutil")
warnings.filterwarnings("ignore")

def normalize_text(text):
    """Gentle normalization for names and addresses."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text).strip()
    return text

def parse_first_float(text):
    """Finds and extracts the true financial value out of messy strings, ignoring structural indices."""
    if not text:
        return None
        
    # Clean out commas to keep numeric blocks unbroken
    cleaned = str(text).replace(",", "")
    
    # Extract all numbers that have a decimal point FIRST (highly likely to be prices/totals)
    decimal_matches = re.findall(r'\d+\.\d+', cleaned)
    if decimal_matches:
        # If the model returned conversational garbage, the actual price is almost always the LAST decimal in the string
        return float(decimal_matches[-1])
        
    # Fallback: if no decimals exist, find any sequence of numbers and grab the largest one
    all_matches = re.findall(r'\d+', cleaned)
    if all_matches:
        return float(max(all_matches, key=int))
        
    return None

def compare_amounts(gt, pred):
    """Semantic comparison for financial amounts that ignores repetitions and leading characters."""
    gt_num = parse_first_float(gt)
    pred_num = parse_first_float(pred)
    
    if gt_num is None and pred_num is None: return 100.0
    if gt_num is None or pred_num is None: return 0.0
    
    return 100.0 if gt_num == pred_num else 0.0

def try_parse_date(token):
    """Attempts to parse an isolated string token into a datetime date object."""
    try:
        return date_parser.parse(str(token), fuzzy=True).date()
    except Exception:
        return None

def compare_dates(gt, pred):
    """
    Semantic comparison for dates. Splitting by tokens to handle 
    duplicated strings or added chatty explanations from the model.
    """
    if not gt and not pred: return 100.0
    if not gt or not pred: return 0.0
    
    # Extract all possible words/date strings separated by spaces
    gt_tokens = str(gt).split()
    pred_tokens = str(pred).split()
    
    # Parse all valid dates found in the ground truth tokens
    gt_dates = [try_parse_date(t) for t in gt_tokens if try_parse_date(t) is not None]
    
    # Parse all valid dates found in the predicted tokens
    # Also try parsing the entire prediction string in case it's a format like "June 7, 1999"
    pred_dates = [try_parse_date(t) for t in pred_tokens if try_parse_date(t) is not None]
    full_pred_parsed = try_parse_date(pred)
    if full_pred_parsed:
        pred_dates.append(full_pred_parsed)
        
    if not gt_dates or not pred_dates:
        # Fallback to absolute strict string comparison if date parsing completely choked
        gt_norm = re.sub(r'[^a-z0-9]', '', str(gt).lower())
        pred_norm = re.sub(r'[^a-z0-9]', '', str(pred).lower())
        return 100.0 if gt_norm == pred_norm else 0.0

    # If ANY parsed date in the prediction matches ANY parsed date in the ground truth, it's a hit!
    for g_date in gt_dates:
        for p_date in pred_dates:
            if g_date == p_date:
                return 100.0
                
    return 0.0

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
        
        # 1. Evaluate Text Fields
        for field in ["vendor_name", "vendor_address"]:
            gt_val = normalize_text(gt.get(field))
            pred_val = normalize_text(pred.get(field))
            if not gt_val and not pred_val:
                continue 
            score = fuzz.token_set_ratio(pred_val, gt_val)
            scores[field].append(score)

        # 2. Evaluate Amounts
        # Look for the alias first, then fall back to the old key
        pred_amount = pred.get("invoice_total") or pred.get("amount_total_gross")
        amt_score = compare_amounts(gt.get("amount_total_gross"), pred_amount)
        scores["amount_total_gross"].append(amt_score)

        # # --- DIAGNOSTIC PRINT ---
        # if amt_score == 0.0:
        #     print(f"❌ MISMATCH [ID: {item.get('doc_id')}]:")
        #     print(f"   Ground Truth Text: '{gt.get('amount_total_gross')}'")
        #     print(f"   Model Predicted:   '{pred.get('invoice_total')}'\n")
        
        # 3. Evaluate Dates
        date_score = compare_dates(gt.get("date_issue"), pred.get("date_issue"))
        scores["date_issue"].append(date_score)

        # # --- DIAGNOSTIC PRINT ---
        # if date_score == 0.0:
        #     print(f"❌ MISMATCH [ID: {item.get('doc_id')}]:")
        #     print(f"   Ground Truth Text: '{gt.get('date_issue')}'")
        #     print(f"   Model Predicted:   '{pred.get('date_issue')}'\n")

    # Calculate and Print the Final Metrics
    print("====== TRUE EXTRACTION ACCURACY ======")
    total_sum = 0
    for field, score_list in scores.items():
        if not score_list:
            print(f"{field:<22}: N/A")
            continue
        avg_score = sum(score_list) / len(score_list)
        print(f"{field:<22}: {avg_score:.2f}%")
        total_sum += avg_score
        
    print("======================================")
    print(f"Total average: {total_sum / 4:.2f}%")
    print("======================================")

if __name__ == "__main__":
    main()


# --- DIAGNOSTIC PRINT ---
        # if amt_score == 0.0:
        #     print(f"❌ MISMATCH [ID: {item.get('doc_id')}]:")
        #     print(f"   Ground Truth Text: '{gt.get('amount_total_gross')}'")
        #     print(f"   Model Predicted:   '{pred.get('amount_total_gross')}'\n")


import json
import re
import argparse
import statistics
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
    cleaned = re.sub(r"(?<=\d)\s+\.(?=\d{1,2}\b)", ".", cleaned)
    cleaned = re.sub(r"(?<=\d)\s+(?=\d{2}\b)", ".", cleaned)
    cleaned = cleaned.replace("^", "0")
    
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
    token = str(token)
    token = re.sub(r"(?<=\d)[Il|](?=\d)", "1", token)
    token = re.sub(r"(?<=[A-Za-z])[Il|](?=[/-])", "1", token)
    token = re.sub(r"(?<=\d)[oO](?=\d)", "0", token)
    try:
        return date_parser.parse(token, fuzzy=True).date()
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


FIELDS = ["vendor_name", "vendor_address", "amount_total_gross", "date_issue"]


def score_record(item):
    gt = item.get("ground_truth", {})
    pred = item.get("predicted", {})

    scores = {}
    for field in ["vendor_name", "vendor_address"]:
        gt_val = normalize_text(gt.get(field))
        pred_val = normalize_text(pred.get(field))
        if not gt_val and not pred_val:
            continue
        scores[field] = fuzz.token_set_ratio(pred_val, gt_val)

    pred_amount = pred.get("invoice_total") or pred.get("amount_total_gross")
    scores["amount_total_gross"] = compare_amounts(gt.get("amount_total_gross"), pred_amount)
    scores["date_issue"] = compare_dates(gt.get("date_issue"), pred.get("date_issue"))
    return scores


def summarize_scores(score_lists):
    field_averages = {
        field: (sum(values) / len(values) if values else None)
        for field, values in score_lists.items()
    }
    populated = [value for value in field_averages.values() if value is not None]
    return {
        "fields": field_averages,
        "total_average": sum(populated) / len(populated) if populated else None,
    }


def score_records(data):
    score_lists = {field: [] for field in FIELDS}
    for item in data:
        for field, score in score_record(item).items():
            score_lists[field].append(score)
    return score_lists

def parse_args():
    parser = argparse.ArgumentParser(description="Score an invoice extraction report.")
    parser.add_argument(
        "--report",
        default="results/eval_report.json",
        help="Path to the eval report JSON produced by eval_runner.py.",
    )
    return parser.parse_args()


def _latency_summary(data):
    latencies = [
        item.get("pipeline_metadata", {}).get("latency_s")
        for item in data
        if item.get("pipeline_metadata", {}).get("latency_s") is not None
    ]
    if not latencies:
        return None
    ordered = sorted(float(value) for value in latencies)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    page_buckets = {}
    for item in data:
        metadata = item.get("pipeline_metadata", {})
        pages = metadata.get("page_count")
        latency = metadata.get("latency_s")
        if pages is None or latency is None:
            continue
        page_buckets.setdefault(str(pages), []).append(float(latency))

    return {
        "count": len(ordered),
        "avg_s": statistics.mean(ordered),
        "p50_s": statistics.median(ordered),
        "p95_s": ordered[p95_index],
        "max_s": max(ordered),
        "by_page_count": {
            page_count: statistics.mean(values)
            for page_count, values in sorted(page_buckets.items(), key=lambda item: int(item[0]))
        },
    }


def main():
    args = parse_args()
    report_path = args.report
    
    try:
        with open(report_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Could not find {report_path}.")
        return

    scores = {field: [] for field in FIELDS}

    print(f"Scoring {len(data)} documents with Semantic Matching...\n")

    for item in data:
        for field, score in score_record(item).items():
            scores[field].append(score)

    # Calculate and Print the Final Metrics
    print("====== TRUE EXTRACTION ACCURACY ======")
    summary = summarize_scores(scores)
    for field, score_list in scores.items():
        if not score_list:
            print(f"{field:<22}: N/A")
            continue
        avg_score = summary["fields"][field]
        print(f"{field:<22}: {avg_score:.2f}%")
        
    print("======================================")
    print(f"Total average: {summary['total_average']:.2f}%")
    print("======================================")

    latency = _latency_summary(data)
    if latency:
        print("\n====== LATENCY ======")
        print(f"Documents             : {latency['count']}")
        print(f"Average               : {latency['avg_s']:.2f}s")
        print(f"P50                   : {latency['p50_s']:.2f}s")
        print(f"P95                   : {latency['p95_s']:.2f}s")
        print(f"Max                   : {latency['max_s']:.2f}s")
        for page_count, avg_latency in latency["by_page_count"].items():
            print(f"Avg for {page_count} page(s)     : {avg_latency:.2f}s")
        print("=====================")

if __name__ == "__main__":
    main()


# --- DIAGNOSTIC PRINT ---
        # if amt_score == 0.0:
        #     print(f"❌ MISMATCH [ID: {item.get('doc_id')}]:")
        #     print(f"   Ground Truth Text: '{gt.get('amount_total_gross')}'")
        #     print(f"   Model Predicted:   '{pred.get('amount_total_gross')}'\n")

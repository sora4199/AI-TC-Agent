
import json
from rule_router import RuleBasedRouter

router = RuleBasedRouter()

with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

correct = 0
total = len(dataset)

print(f"{'QUERY':<55} | {'DỰ ĐOÁN':<15} | {'THỰC TẾ':<15} | KẾT QUẢ")
print("-" * 95)

for item in dataset:
    q = item["query"]
    expected = item["label"]
    
    pred = router.route(q)
    predicted_route = pred["route"]
    
    is_correct = (predicted_route == expected)
    if is_correct:
        correct += 1
        res_str = "PASS"
    else:
        res_str = "FAIL"
        
    print(f"{q[:52]:<55} | {predicted_route:<15} | {expected:<15} | {res_str}")

accuracy = (correct / total) * 100
print("-" * 95)
print(f"Tổng kết Accuracy Baseline 2 (Rule-based): {accuracy:.1f}% ({correct}/{total})")

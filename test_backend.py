"""Backend API test script."""
import requests
import json
import sys

BASE = "http://localhost:8080"
results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    detail_str = f"  =>  {detail}" if detail != "" else ""
    print(f"  [{status}] {name}{detail_str}")
    results.append((name, condition))


print()
print("=" * 50)
print("  GET /health")
print("=" * 50)
r = requests.get(f"{BASE}/health", timeout=10)
check("Status 200", r.status_code == 200, r.status_code)
d = r.json()
check("model_loaded = True", d.get("model_loaded") is True, d.get("model_loaded"))
check("status = healthy", d.get("status") == "healthy", d.get("status"))

print()
print("=" * 50)
print("  GET /info")
print("=" * 50)
r = requests.get(f"{BASE}/info", timeout=10)
check("Status 200", r.status_code == 200, r.status_code)
print(f"  Response: {json.dumps(r.json(), indent=4)}")

print()
print("=" * 50)
print("  POST /predict  (valid input)")
print("=" * 50)
payload = {
    "text": (
        "Amanda: Hey, are we meeting today?\n"
        "Jerry: Sure! What time works for you?\n"
        "Amanda: How about 3pm at the coffee shop?\n"
        "Jerry: Perfect, see you there!\n"
        "Amanda: Great, I'll bring the project reports.\n"
        "Jerry: Awesome, I'll review them beforehand."
    ),
    "max_length": 60,
}
r = requests.post(f"{BASE}/predict", json=payload, timeout=120)
check("Status 200", r.status_code == 200, r.status_code)
d = r.json()
check("summary not empty", bool(d.get("summary")), repr(d.get("summary")))
check("input_length > 0", (d.get("input_length") or 0) > 0, d.get("input_length"))
check("summary_length > 0", (d.get("summary_length") or 0) > 0, d.get("summary_length"))
print(f"  Summary: {d.get('summary')}")

print()
print("=" * 50)
print("  POST /predict  (max_length out of range)")
print("=" * 50)
r = requests.post(f"{BASE}/predict", json={"text": payload["text"], "max_length": 9999}, timeout=10)
check("Status 422 (max_length > 512)", r.status_code == 422, r.status_code)

print()
print("=" * 50)
print("  POST /predict  (text too short — validation)")
print("=" * 50)
r = requests.post(f"{BASE}/predict", json={"text": "hi"}, timeout=10)
check("Status 422", r.status_code == 422, r.status_code)

print()
print("=" * 50)
print("  POST /predict  (missing body)")
print("=" * 50)
r = requests.post(f"{BASE}/predict", json={}, timeout=10)
check("Status 422", r.status_code == 422, r.status_code)

print()
print("=" * 50)
print("  POST /predict/batch  (2 texts)")
print("=" * 50)
batch_payload = {
    "texts": [
        "Alice: Can you send me the slides?\nBob: Sure, give me 5 mins.\nAlice: Thanks a lot!",
        "Tom: Did you call the plumber?\nSue: Yes, he comes on Friday morning.\nTom: Great, finally.",
    ],
    "max_length": 50,
}
r = requests.post(f"{BASE}/predict/batch", json=batch_payload, timeout=180)
check("Status 200", r.status_code == 200, r.status_code)
d = r.json()
check("count == 2", d.get("count") == 2, d.get("count"))
for i, res in enumerate(d.get("results", [])):
    check(f"batch[{i}] summary not empty", bool(res.get("summary")), repr(res.get("summary")))
    print(f"  batch[{i}] summary: {res.get('summary')}")

print()
print("=" * 50)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  {passed}/{total} tests passed")
print("=" * 50)
print()

if passed < total:
    sys.exit(1)

import json
import pandas as pd
from collections import Counter, defaultdict

def load_dataset(path="intents.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_dataset(data):
    intents = data.get("intents", [])

    # Extract statistics
    tags = []
    patterns = []
    responses = []
    pattern_counts = defaultdict(int)
    response_counts = defaultdict(int)
    missing_fields = []
    duplicate_tags = []
    
    # Count occurrences of each tag
    tag_counter = Counter([intent.get("tag") for intent in intents])
    duplicate_tags = [tag for tag, count in tag_counter.items() if count > 1]

    # Flatten patterns and responses
    for intent in intents:
        tag = intent.get("tag")

        p = intent.get("patterns", [])
        r = intent.get("responses", [])

        # Missing fields
        if not p or not r:
            missing_fields.append(tag)

        pattern_counts[tag] = len(p)
        response_counts[tag] = len(r)

        for x in p:
            patterns.append((tag, x))
        for x in r:
            responses.append((tag, x))
        
        tags.append(tag)

    # Detect duplicate patterns within the same tag
    duplicate_patterns = []
    for tag in pattern_counts:
        p_list = [p for t, p in patterns if t == tag]
        duplicates = [item for item, count in Counter(p_list).items() if count > 1]
        if duplicates:
            duplicate_patterns.append({tag: duplicates})

    # Build pandas DataFrame
    df = pd.DataFrame({
        "tag": list(pattern_counts.keys()),
        "num_patterns": list(pattern_counts.values()),
        "num_responses": list(response_counts.values())
    }).sort_values("tag")

    return {
        "total_intents": len(intents),
        "duplicate_tags": duplicate_tags,
        "missing_fields": missing_fields,
        "duplicate_patterns": duplicate_patterns,
        "summary": df
    }

def main():
    print("\n📊 DATASET STATISTICS\n")
    
    data = load_dataset()
    stats = analyze_dataset(data)

    print(f"Total intents: {stats['total_intents']}")
    print("\nDuplicate tags:", stats["duplicate_tags"])
    print("\nTags with missing patterns/responses:", stats["missing_fields"])
    print("\nDuplicate patterns inside tags:")
    for item in stats["duplicate_patterns"]:
        print("  ", item)

    print("\n📘 Summary Table:")
    print(stats["summary"].to_string(index=False))

if __name__ == "__main__":
    main()

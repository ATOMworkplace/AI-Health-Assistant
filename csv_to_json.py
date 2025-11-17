import csv
import json

def main():
    # Dictionary to store grouped data
    # Structure: { tag: { "patterns": set(), "responses": set() } }
    dct = {}

    try:
        with open('intents.csv', 'r', encoding='utf-8') as f:
            # DictReader automatically uses the header row (tag,pattern,response)
            rdr = csv.DictReader(f)
            
            for row in rdr:
                t = row['tag'].strip()
                p = row['pattern'].strip()
                r = row['response'].strip()

                if not t:
                    continue

                if t not in dct:
                    dct[t] = {'patterns': set(), 'responses': set()}

                if p:
                    dct[t]['patterns'].add(p)
                if r:
                    dct[t]['responses'].add(r)

        # Convert to list of dicts for JSON
        lst = []
        for t, v in dct.items():
            obj = {
                "tag": t,
                "patterns": sorted(list(v['patterns'])),
                "responses": sorted(list(v['responses']))
            }
            lst.append(obj)

        out = {"intents": lst}

        with open('intents.json', 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=4, ensure_ascii=False)
            
        print("Success: intents.csv converted to intents.json")

    except FileNotFoundError:
        print("Error: intents.csv not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
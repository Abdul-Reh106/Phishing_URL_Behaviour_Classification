from pathlib import Path
import matplotlib.pyplot as plt
from features import extract_features
from prepare_data import load_clean_data

data = load_clean_data()
valid_rows = []
for url in data["url"]:
    try:
        extract_features(url)
        valid_rows.append(True)
    except ValueError:
        valid_rows.append(False)

data = data.loc[valid_rows]
counts = data["label"].value_counts().reindex([0, 1], fill_value=0)

print("Usable URLs:", len(data))
print("\nURLs in each class:")
print(counts)
Path("plots").mkdir(exist_ok=True)
plt.bar(
    ["Phishing (0)", "Benign (1)"],
    counts.to_list(),
    color=["#e76f51", "#457b9d"],
)
plt.ylabel("Number of URLs")
plt.title("Class Distribution in the Final Dataset")
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=150)
plt.show()
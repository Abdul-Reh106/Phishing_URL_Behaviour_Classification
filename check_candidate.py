from urllib.parse import urlsplit
import pandas as pd


def has_path_content(url):
    text = str(url)

    # Some URLs have no http:// or https://. Add it only for parsing.
    if "://" not in text:
        text = "http://" + text

    try:
        return bool(urlsplit(text).path.strip("/"))
    except ValueError:
        return False


data = pd.read_csv(
    "candidate_data/malicious_phish.csv",
    usecols=["url", "type"]
)

data = data[data["type"].isin(["benign", "phishing"])].copy()
data["has_path"] = data["url"].map(has_path_content)

print("Rows and columns:", data.shape)
print("\nURLs in each class:")
print(data["type"].value_counts())
print("\nURLs with path content by class (%):")
print((data.groupby("type")["has_path"].mean() * 100).round(2))
data["has_http_prefix"] = data["url"].str.startswith(
    ("http://", "https://"), na=False
)

print("\nURLs starting with http:// or https:// by class (%):")
print((data.groupby("type")["has_http_prefix"].mean() * 100).round(2))
print("\nMissing URLs:", data["url"].isna().sum())
print("Repeated URLs:", data["url"].duplicated().sum())

labels_per_url = data.groupby("url")["type"].nunique()
print("URLs with conflicting labels:", (labels_per_url > 1).sum())
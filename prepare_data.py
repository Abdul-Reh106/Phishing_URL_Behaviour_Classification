import pandas as pd


def load_clean_data():
    data = pd.read_csv(
        "candidate_data/malicious_phish.csv",
        usecols=["url", "type"]
    )

    data = data[data["type"].isin(["benign", "phishing"])].dropna(
        subset=["url"]
    ).copy()

    # Give every URL the same prefix before checking for duplicates.
    data["url"] = (
        "https://"
        + data["url"].str.strip().str.replace(
            r"^https?://", "", regex=True, case=False
        )
    )

    labels_per_url = data.groupby("url")["type"].nunique()
    conflicting_urls = labels_per_url[labels_per_url > 1].index
    print("Conflicting URLs after prefix cleanup:", len(conflicting_urls))

    data = data[~data["url"].isin(conflicting_urls)]
    data = data.drop_duplicates(subset="url").reset_index(drop=True)

    # Keep the same label numbers as our first project version.
    data["label"] = data["type"].map({"phishing": 0, "benign": 1})
    return data[["url", "label"]]


if __name__ == "__main__":
    clean_data = load_clean_data()
    print("Rows after cleaning:", len(clean_data))
    print("\nLabel counts (0 = phishing, 1 = benign):")
    print(clean_data["label"].value_counts())
    print("\nRepeated URLs remaining:", clean_data["url"].duplicated().sum())
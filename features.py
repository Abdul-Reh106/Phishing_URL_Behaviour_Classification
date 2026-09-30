import re
from urllib.parse import urlsplit

def extract_features(url):
    url = re.sub(r"^https?://", "", url.strip(), flags=re.IGNORECASE)
    url = "https://" + url

    hostname = urlsplit(url).hostname
    if not hostname:
        raise ValueError("Enter a full URL, such as https://example.com")

    return {
        "url_length": len(url),
        "digit_ratio": len(re.findall(r"[0-9]", url)) / len(url),
        "special_char_count": len(re.findall(r"[\W_]", url)),
        "hostname_length": len(hostname),
        "hostname_digit_ratio": len(re.findall(r"[0-9]", hostname)) / len(hostname),
        "hostname_dot_count": hostname.count("."),
    }
if __name__ == "__main__":
    print(extract_features("https://www.example.com"))
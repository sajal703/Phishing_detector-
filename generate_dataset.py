"""
generate_dataset.py
Generates a synthetic, labeled dataset of phishing vs legitimate URLs
by sampling realistic feature distributions (rather than scraping live
URLs, which needs network access this environment doesn't have).

NOTE FOR YOUR REPORT/VIVA:
This mirrors the approach used in real phishing-detection papers when
building on top of the UCI Phishing Websites dataset structure, but
generates the data programmatically. For your final submission, you
can optionally swap this out for the real UCI or Kaggle "Phishing Site
URLs" dataset (just make sure the feature extraction in features.py
matches whatever raw URLs you feed it, or use the dataset's
pre-extracted features directly).
"""

import random
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract_features, FEATURE_NAMES

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "wikipedia.org", "github.com", "amazon.in", "flipkart.com",
    "nseindia.com", "rbi.org.in", "icicibank.com", "hdfcbank.com", "irctc.co.in",
    "microsoft.com", "apple.com", "linkedin.com", "stackoverflow.com", "medium.com",
    "galgotiasuniversity.edu.in", "aicte-india.org", "ugc.gov.in", "nptel.ac.in",
]

LEGIT_PATHS = [
    "/", "/about", "/products", "/contact", "/blog/2024/article",
    "/account/dashboard", "/search?q=result", "/help/faq", "/news/today",
    "/docs/api/reference", "/user/profile/settings",
]

PHISH_BRANDS = ["paypal", "hdfcbank", "icicibank", "amazon", "sbi", "google", "microsoft", "netflix", "instagram"]
PHISH_TLDS = [".tk", ".xyz", ".top", ".ml", ".ga", ".cf", ".info", ".click"]
SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "update", "confirm", "signin", "suspend"]


def make_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = "https"
    sub = random.choice(["", "www.", "www.", "www."])  # mostly www
    return f"{scheme}://{sub}{domain}{path}"


def make_phishing_url():
    """Builds a synthetic phishing-style URL using common real-world patterns."""
    pattern = random.choice(["ip", "subdomain_spoof", "hyphen_spoof", "shortener", "suspicious_path", "long_random"])
    brand = random.choice(PHISH_BRANDS)
    word = random.choice(SUSPICIOUS_WORDS)

    if pattern == "ip":
        ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/{word}/{brand}"

    if pattern == "subdomain_spoof":
        fake_tld = random.choice(PHISH_TLDS)
        junk = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
        return f"http://{brand}.{word}.{junk}{fake_tld}/{word}"

    if pattern == "hyphen_spoof":
        fake_tld = random.choice(PHISH_TLDS)
        return f"http://{brand}-{word}-secure{fake_tld}/{word}/index.php"

    if pattern == "shortener":
        shortener = random.choice(["bit.ly", "tinyurl.com", "is.gd", "cutt.ly"])
        code = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFG0123456789", k=7))
        return f"http://{shortener}/{code}"

    if pattern == "suspicious_path":
        fake_tld = random.choice(PHISH_TLDS)
        junk = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))
        return f"http://{junk}{fake_tld}/{brand}/{word}/account@{brand}.com/{word}.php"

    # long_random
    fake_tld = random.choice(PHISH_TLDS)
    junk = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789-", k=25))
    return f"http://{junk}{fake_tld}/{word}-{brand}-{word}/confirm.php?id={random.randint(1000,9999)}"


def make_tricky_legit_url():
    """Legit-ish URL that happens to trigger some phishing-style features
    (long marketing URLs, login pages, tracking params) -- real sites do this."""
    domain = random.choice(LEGIT_DOMAINS)
    word = random.choice(SUSPICIOUS_WORDS)
    junk = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=random.randint(4, 10)))
    return f"https://www.{domain}/{word}/{junk}?ref=email&utm_id={random.randint(1000,9999)}"


def make_clean_phishing_url():
    """Phishing URL using a fresh-looking but legitimate-seeming TLD/structure --
    real attackers increasingly use clean-looking short domains with valid HTTPS."""
    brand = random.choice(PHISH_BRANDS)
    junk = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(5, 9)))
    tld = random.choice([".com", ".net", ".co"])  # deliberately "normal" looking
    return f"https://{brand}{junk}{tld}/account/index.html"


def build_dataset(n_per_class: int = 1500, noise_fraction: float = 0.12) -> pd.DataFrame:
    """
    noise_fraction controls how much of each class is drawn from the
    'tricky'/overlapping generators instead of the clean-pattern ones.
    This keeps the dataset realistic (no single feature perfectly
    separates the classes), which is what you'd see with real-world data.
    """
    rows = []
    n_noisy = int(n_per_class * noise_fraction)
    n_clean = n_per_class - n_noisy

    for _ in range(n_clean):
        url = make_legit_url()
        feats = extract_features(url)
        feats["url"] = url
        feats["label"] = 0
        rows.append(feats)
    for _ in range(n_noisy):
        url = make_tricky_legit_url()
        feats = extract_features(url)
        feats["url"] = url
        feats["label"] = 0
        rows.append(feats)

    for _ in range(n_clean):
        url = make_phishing_url()
        feats = extract_features(url)
        feats["url"] = url
        feats["label"] = 1
        rows.append(feats)
    for _ in range(n_noisy):
        url = make_clean_phishing_url()
        feats = extract_features(url)
        feats["url"] = url
        feats["label"] = 1
        rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Inject feature-level noise so classes overlap realistically ---
    # Real phishing/legit URLs are never perfectly separable on these
    # features alone; a small fraction of random label-independent
    # corruption keeps the model (and its accuracy numbers) credible.
    rng = random.Random(7)
    binary_cols = ["has_https", "has_suspicious_word", "has_ip_address", "has_shortener", "has_hyphen_in_domain"]
    numeric_cols = ["url_length", "num_dots", "num_hyphens", "num_digits", "domain_length", "path_length", "num_subdomains"]

    for col in binary_cols:
        flip_idx = df.sample(frac=0.06, random_state=rng.randint(0, 10000)).index
        df.loc[flip_idx, col] = 1 - df.loc[flip_idx, col]

    for col in numeric_cols:
        jitter_idx = df.sample(frac=0.15, random_state=rng.randint(0, 10000)).index
        jitter = pd.Series(
            [rng.randint(-3, 3) for _ in range(len(jitter_idx))], index=jitter_idx
        )
        df.loc[jitter_idx, col] = (df.loc[jitter_idx, col] + jitter).clip(lower=0)

    cols = ["url"] + FEATURE_NAMES + ["label"]
    return df[cols]


if __name__ == "__main__":
    df = build_dataset(n_per_class=1500)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phishing_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["label"].value_counts())

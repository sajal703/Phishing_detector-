"""
features.py
Extracts lexical and structural features from a URL to feed the
phishing classification model. No network calls required for the
core feature set (fast, works offline, good for a minor project).

Each feature function returns a numeric value (0/1 for booleans,
counts for lexical features). The extract_features() function
returns them in a fixed order that MUST match the order used
during training (see FEATURE_NAMES).
"""

import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "banking", "password", "signin", "ebayisapi", "webscr", "suspend",
]

SHORTENER_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "cutt.ly",
]

FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_at_symbols",
    "num_subdomains",
    "has_ip_address",
    "has_https",
    "has_double_slash_redirect",
    "has_suspicious_word",
    "has_shortener",
    "domain_length",
    "path_length",
    "has_hyphen_in_domain",
    "num_query_params",
]


def has_ip_address(hostname: str) -> int:
    ip_pattern = re.compile(
        r"^(\d{1,3}\.){3}\d{1,3}$|"          # IPv4
        r"^0x[0-9a-fA-F]+$|"                  # hex encoded
        r"^(\d{1,3}-){3}\d{1,3}$"             # dashed
    )
    return 1 if hostname and ip_pattern.match(hostname) else 0


def extract_features(url: str) -> dict:
    """
    Takes a raw URL string and returns a dict of {feature_name: value}.
    Robust to missing scheme (adds https:// if absent so urlparse works).
    """
    raw_url = url.strip()
    if not re.match(r"^[a-zA-Z]+://", raw_url):
        parseable_url = "http://" + raw_url
    else:
        parseable_url = raw_url

    parsed = urlparse(parseable_url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Count subdomains: split hostname by '.', subtract domain+tld (2 parts)
    host_parts = hostname.split(".") if hostname else []
    num_subdomains = max(0, len(host_parts) - 2)

    feats = {
        "url_length": len(raw_url),
        "num_dots": raw_url.count("."),
        "num_hyphens": raw_url.count("-"),
        "num_underscores": raw_url.count("_"),
        "num_slashes": raw_url.count("/"),
        "num_digits": sum(c.isdigit() for c in raw_url),
        "num_at_symbols": raw_url.count("@"),
        "num_subdomains": num_subdomains,
        "has_ip_address": has_ip_address(hostname),
        "has_https": 1 if raw_url.lower().startswith("https://") else 0,
        "has_double_slash_redirect": 1 if raw_url.rfind("//") > 7 else 0,
        "has_suspicious_word": 1 if any(w in raw_url.lower() for w in SUSPICIOUS_WORDS) else 0,
        "has_shortener": 1 if any(s in hostname.lower() for s in SHORTENER_SERVICES) else 0,
        "domain_length": len(hostname),
        "path_length": len(path),
        "has_hyphen_in_domain": 1 if "-" in hostname else 0,
        "num_query_params": query.count("&") + 1 if query else 0,
    }
    return feats


def extract_feature_vector(url: str) -> list:
    """Returns features as an ordered list matching FEATURE_NAMES, for model input."""
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]


def explain_prediction(url: str, top_n: int = 3) -> list:
    """
    Rule-based human-readable explanation of why a URL looks suspicious.
    Used alongside the ML prediction so the result isn't a black box —
    important for a security tool's usability and for viva defense.
    """
    feats = extract_features(url)
    reasons = []

    if feats["has_ip_address"]:
        reasons.append("URL uses a raw IP address instead of a domain name")
    if not feats["has_https"]:
        reasons.append("Connection is not secured with HTTPS")
    if feats["num_at_symbols"] > 0:
        reasons.append("URL contains an '@' symbol, which can hide the real destination")
    if feats["has_suspicious_word"]:
        reasons.append("URL contains suspicious keywords (e.g. login, verify, secure)")
    if feats["has_shortener"]:
        reasons.append("URL uses a link-shortening service, which can mask the true destination")
    if feats["num_subdomains"] >= 3:
        reasons.append("Unusually high number of subdomains")
    if feats["has_hyphen_in_domain"]:
        reasons.append("Domain name contains hyphens, often used to mimic legitimate brands")
    if feats["url_length"] > 75:
        reasons.append("URL is unusually long")
    if feats["has_double_slash_redirect"]:
        reasons.append("URL contains a redirect pattern ('//') outside the protocol prefix")

    if not reasons:
        reasons.append("No major red-flag patterns detected in the URL structure")

    return reasons[:top_n]

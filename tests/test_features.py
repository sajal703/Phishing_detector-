
from features import extract_features, extract_feature_vector, explain_prediction, FEATURE_NAMES, has_ip_address


def test_https_detected():
    feats = extract_features("https://www.google.com/search")
    assert feats["has_https"] == 1

def test_http_not_https():
    feats = extract_features("http://example.com")
    assert feats["has_https"] == 0

def test_ip_address_detected():
    feats = extract_features("http://192.168.1.1/login")
    assert feats["has_ip_address"] == 1

def test_domain_not_flagged_as_ip():
    feats = extract_features("https://www.amazon.in/")
    assert feats["has_ip_address"] == 0

def test_at_symbol_counted():
    feats = extract_features("http://example.com@evil.com/verify")
    assert feats["num_at_symbols"] == 1

def test_suspicious_word_detected():
    feats = extract_features("http://secure-login-update.tk/account")
    assert feats["has_suspicious_word"] == 1

def test_no_suspicious_word():
    feats = extract_features("https://www.wikipedia.org/wiki/Python")
    assert feats["has_suspicious_word"] == 0

def test_shortener_detected():
    feats = extract_features("http://bit.ly/3xAbCde")
    assert feats["has_shortener"] == 1

def test_subdomain_count():
    feats = extract_features("http://a.b.c.example.com/path")
    assert feats["num_subdomains"] >= 2

def test_feature_vector_length_matches_names():
    vec = extract_feature_vector("https://www.github.com/user/repo")
    assert len(vec) == len(FEATURE_NAMES)

def test_feature_vector_all_numeric():
    vec = extract_feature_vector("http://192.168.0.5/login/verify")
    assert all(isinstance(v, (int, float)) for v in vec)

def test_explain_returns_reasons():
    reasons = explain_prediction("http://192.168.1.1/verify-account")
    assert isinstance(reasons, list)
    assert len(reasons) > 0

def test_explain_clean_url_has_fallback_reason():
    reasons = explain_prediction("https://www.google.com")
    assert isinstance(reasons, list)
    assert len(reasons) > 0

def test_url_without_scheme_still_parses():
    feats = extract_features("www.example.com/path")
    assert feats["domain_length"] > 0

def test_has_ip_address_helper_direct():
    assert has_ip_address("192.168.1.1") == 1
    assert has_ip_address("example.com") == 0
    assert has_ip_address("") == 0

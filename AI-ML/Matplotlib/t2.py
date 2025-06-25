import re

text = "Order placed on 2025-06-10"
pattern = r"(\d{4}-\d{2}-\d{2})"

match = re.search(pattern, text)
if match:
    print("Extracted substring:", match.group(1))
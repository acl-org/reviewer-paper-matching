import suggest_utils
import sys
import json

for line in sys.stdin:
    suggest_utils.print_text_report(json.loads(line), sys.stdout)

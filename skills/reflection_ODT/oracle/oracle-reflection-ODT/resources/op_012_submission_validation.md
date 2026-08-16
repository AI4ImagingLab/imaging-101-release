Validation commands

Run from the evaluation run root after writing the submission:

```
python ${CLAUDE_SKILL_DIR}/scripts/op_012_validate_submission.py
```

This script checks only public schema and numerical sanity. Passing it does not imply a high reconstruction metric, but failing it means the file is malformed for the task contract.
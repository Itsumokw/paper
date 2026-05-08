# PerLTQA Dataset Source

Downloaded on 2026-05-06.

Official paper:
- PerLTQA: A Personal Long-Term Memory Dataset for Memory Classification, Retrieval, and Fusion in Question Answering
- ACL Anthology: https://aclanthology.org/2024.sighan-1.18/
- PDF: https://aclanthology.org/2024.sighan-1.18.pdf

Official code/data repository:
- https://github.com/Elvin-Yiming-Du/PerLTQA

Downloaded repository commit:
- 8d9e19868e239740ef701e603ec205cd581f221b

License:
- CC BY-NC 4.0
- Non-commercial research use only.

Local Chinese files:
- Dataset/zh/perltmem.json
- Dataset/zh/perltqa.json

Chinese dataset counts:
- PerLT memory characters: 141
- PerLT QA characters: 32
- Total QA records: 8,593
- Profile QA: 357
- Social relationship QA: 897
- Event QA: 4,501
- Dialogue QA: 2,838

Format note:
- `perltmem.json` stores long-term memory for characters, including profile, profile_description, social_relationship, events, and dialogues.
- `perltqa.json` stores QA records grouped by character and memory category. Each QA record contains Question, Answer, Reference Memory, and Memory Anchors.

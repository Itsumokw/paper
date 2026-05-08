# CareCall-Memory Dataset Application Draft

Date: 2026-05-07

## Application Information

| Field | Content |
|---|---|
| Applicant name | Zhao Liangzhe |
| Chinese name | 赵亮哲 |
| Institution | Fudan University |
| School / Department | School of Software |
| Major | Software Engineering |
| Status | Undergraduate student, Class of 2022 |
| Applicant email | 22302010032@m.fudan.edu.cn |
| Advisor | Prof. Xu Yingxiao |
| Advisor email | xuyx@fudan.edu.cn |
| Requested dataset | Original CareCall-Memory dataset in Korean |
| Purpose | Non-commercial undergraduate thesis research |
| Redistribution | I will not copy, redistribute, publish, upload, or transfer the original dataset to any third party. |

## Proposed Thesis Title

Multilingual Long-Term Conversational Memory Evaluation via LoCoMo-style Question Answering

## Research Direction

My research focuses on long-term conversational memory for large language models. Specifically, I plan to construct and evaluate multilingual long-term memory benchmarks by converting publicly available multi-session dialogue datasets into a LoCoMo-style format with sessions, memory facts, questions, answers, and evidence annotations. The goal is to study whether existing memory-augmented LLM methods can robustly retrieve, update, and reason over user-related information across long-term, multi-session conversations in different languages.

## Planned Baselines

I plan to compare the proposed method with representative long-term memory and retrieval baselines, including:

- Full Context
- Naive RAG
- MemoryBank
- Mem0
- A-Mem
- SimpleMem
- HingMem
- AriadneMem

The evaluation will mainly follow a LoCoMo-style QA setting, using answer-level F1 and related retrieval/evidence analysis where applicable.

## Email Subject

Request for Access to the Original Korean CareCall-Memory Dataset for Non-commercial Undergraduate Thesis Research

## Email Body

Dear CareCall-Memory authors,

My name is Zhao Liangzhe, an undergraduate student in Software Engineering at the School of Software, Fudan University. I am currently working on my undergraduate thesis under the supervision of Prof. Xu Yingxiao.

I am writing to request access to the original Korean version of the CareCall-Memory dataset for non-commercial academic research.

My thesis is tentatively titled "Multilingual Long-Term Conversational Memory Evaluation via LoCoMo-style Question Answering." The research focuses on long-term conversational memory for large language models. Specifically, I plan to convert multi-session dialogue datasets into a unified LoCoMo-style evaluation format consisting of dialogue sessions, memory facts, questions, answers, and evidence annotations. I would like to use CareCall-Memory as a Korean long-term conversation dataset because it naturally contains multi-session dialogues, user memory, and session summaries, which makes it highly suitable for evaluating long-term memory modeling.

In my experiments, I plan to compare representative memory and retrieval baselines such as Full Context, Naive RAG, MemoryBank, Mem0, A-Mem, SimpleMem, HingMem, and AriadneMem. The evaluation will focus on long-term memory question answering, answer-level F1, and evidence-based analysis.

I understand and agree to the dataset license terms stated in the CareCall-Memory repository. The dataset will be used only for non-commercial AI research and undergraduate thesis work. I will not use the dataset for any commercial purpose. I will not copy, redistribute, publish, upload, or transfer the original dataset to any third party or to the public. I will only report aggregated experimental results and analysis in my thesis, and I will clearly cite NAVER Corp. and the CareCall-Memory paper as the source of the dataset.

My information is as follows:

- Name: Zhao Liangzhe
- Institution: Fudan University
- School: School of Software
- Major: Software Engineering
- Status: Undergraduate student, Class of 2022
- Email: 22302010032@m.fudan.edu.cn
- Advisor: Prof. Xu Yingxiao
- Advisor email: xuyx@fudan.edu.cn

I would be very grateful if you could grant me access to the original Korean CareCall-Memory dataset or let me know if there are any additional application steps I should complete.

Thank you very much for your time and for releasing this valuable dataset for long-term conversational memory research.

Sincerely,

Zhao Liangzhe

School of Software, Fudan University

22302010032@m.fudan.edu.cn

## Suggested Recipients

The CareCall-Memory README lists the following contacts:

- Sangwhan Bae: sanghwan.bae@navercorp.com
- Sungdong Kim: sungdong.kim@navercorp.com

Suggested sending strategy:

- To: sanghwan.bae@navercorp.com
- Cc: sungdong.kim@navercorp.com, xuyx@fudan.edu.cn

## Note

The original NAVER form link in the CareCall-Memory README is:

- Short link: https://naver.me/5zovK7N5
- Resolved old NAVER Office form URL: https://form.office.naver.com/form/responseView.cmd?formkey=NThlYmUzYmEtMDU2OS00ZTY1LTk4MTktOGIzZDI1ZmExYjA5&sourceId=urlshare

As of 2026-05-07, this old NAVER Office form URL redirects to `https://notice.naver.com/notices/form/14352`, which is a generic NAVER Form notice page rather than a usable CareCall-Memory application form. I also checked the GitHub issues and did not find a newer replacement form link. If the form remains unavailable, email application is the safest fallback.

SYSTEM_ENRICHMENT = "You are an helpful AI assistant expert in summarizing, enriching and expanding data samples."

PROMPT_CAPTION_PS = """
Given a functional description of a {category} and a code representation in the code language {language} that implements this functionality, generate a caption for a picture which will be rendered from this code.
The picture will represent exactly what is contained in the code and the description. The caption should be abstracting and contain only the most relevant information.
Consider that in the end only the rendered picture and the caption are visible, not the code or the descripton. Therefore, the coding language itself or anything invisible should not be mentioned in the caption.

Functionality:
{description}

Code:
```
{code}
```

Now, please output your results in the format below by filling in the placeholders in <...>:

Caption: <caption>
""".lstrip()

PROMPT_CAPTION_QA = """
Given a code representation in the code language {language} that implements a {category}, generate a caption for a picture which will be rendered from this code.
The picture will represent exactly what is contained in the code. The caption should be abstracting and contain only the most relevant information.
Consider that in the end only the rendered picture and the caption are visible, not the code. Therefore, the coding language itself or anything invisible should not be mentioned in the caption.

Code:
```
{code}
```

Now, please output your results in the format below by filling in the placeholders in <...>:

Caption: <caption>
""".lstrip()

SYSTEM_LLM_QA = "You are an helpful AI assistant expert in generating realistic question-answer pairs."

PROMPT_LLM_QA = """
Given a code representation in the code language {language}, generate a natural user question that refers to the image of a {category} which will be rendered from this code.
The image will represent exactly what is contained in the code. Use the code internally as ground truth to understand the contents and visual structure of the image. 
Also generate an assistant response that gives the correct answer, derived directly from the code. The answer must match exactly what is in the image.
The code will **not** be present in the final dataset, so the question and answer must sound like they refer to the image itself, not the code. Do not mention, quote, or reference the code or the code language in the question or answer. Derive every detail of the question and answer only from what is unambiguously implied by the code. Do not invent content.
Frame the question such that it requires visual interpretation of the rendered image. {question_type} The assistant response should begin directly with the correct answer, followed by a short explanation on a new line.

Code:
```
{code}
```

Now, please output your results in the format below by filling in the placeholders in <...>:

User: <user question>
Assistant: <assistant answer>
""".lstrip()


SYSTEM_SCORE_PS = """
You are a professional expert in {category}, with deep experience in reviewing code and solution specifications.

You will be given:
- A problem statement that describes a task
- The programming language used to implement a solution
- The actual code implementation
- A functional description of what the code is supposed to do

Your task is to judge whether 
1. the code and the functional description are factually, technically and logically correct,
2. the functional description correctly reflect what the code does.

Use your domain expertise in {category} for the assessment. Don’t assume missing context; do not run code; judge from text only.

Output your result as binary value "correct".
Output "true" when all is correct by tolerating minor deviations or irrelevant inconsistencies.
Output "false" only when you detect clear, material errors or contradictions.
Otherwise output "null" for an unsure/indeterminate result.

Also provide a short factual explanation for your decision, free of speculation, of max. 30 words.
""".lstrip()


PROMPT_SCORE_PS = """
Given the following problem statement, code language, code and its functional description, assess the correctness.

Problem Statement:
{problem}

Code Language: {language}

Code:
```
{code}
```

Functional Description:
{description}


Now, please output your results in the format below by filling in the placeholders in <...>:

correct: <true|false|null>
explanation: <brief factual explanation>
""".lstrip()


SYSTEM_SCORE_QA = """
You are a professional expert in {category}, with deep experience in reviewing code and solution specifications.

You will be given:
- A problem statement that describes a task refering to a code implementation
- The programming language used for implementing the code
- The actual code implementation
- An answer to the question addressed in the problem statement refering to the code implementation

Your task is to judge whether 
1. the answer is factually, technically and logically correct,
2. the answer correctly identifies issues in the code addressed by the problem statement.

Use your domain expertise in {category} for the assessment. Don’t assume missing context; do not run code; judge from text only.

Output your result as binary value "correct".
Output "true" when all is correct by tolerating minor deviations or irrelevant inconsistencies.
Output "false" only when you detect clear, material errors or contradictions.
Otherwise output "null" for an unsure/indeterminate result.

Also provide a short factual explanation for your decision, free of speculation, of max. 30 words.
""".lstrip()


PROMPT_SCORE_QA = """
Given the following problem statement, code language, code and answer to the problem statement, assess the correctness.

Problem Statement:
{problem}

Code Language: {language}

Code:
```
{code}
```

Answer:
{answer}


Now, please output your results in the format below by filling in the placeholders in <...>:

correct: <true|false|null>
explanation: <brief factual explanation>
""".lstrip()

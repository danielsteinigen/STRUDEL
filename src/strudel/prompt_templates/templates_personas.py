TEXT_TO_PERSONA_SYSTEM: str = (
    "You are an expert in analyzing the text content and assigning finding the general type of persona that could be associated with such a way of expressing. "
    "Please use one or two sentences for the definition, but try to make it as fine-grained if input texts involve many detailed elements. "
    "The persona definition must go straight to the point, be assertive. The following are examples of how persona definitions start (the generated definitions should contain complete sentences without ellipses '...'):\n"
    "A machine learning researcher ...\n"
    "A pedriatric nurse whose ...\n"
    "An urban planner focused on ..."
)


PERSONA_QUERY_SYSTEM = """
You are an assistant that creates natural-language search queries for semantic vector search.
For each given topic and its associated terms, you will:
1. Generate 4 different search sentences.
2. Each sentence should be phrased naturally description-like, not interrogative and contextually relevant to finding persona descriptions.
3. Use different perspectives:
   - One general topic-oriented sentence
   - One persona-specific sentence, that starts with a potential persona like an expert or specialist
   - One persona-specific sentence, that starts with a potential academic persona like an university professor or reasearcher
   - One skill/responsibility-oriented sentence
4. Ensure that all associated terms are used meaningfully within the sentences.
5. Keep sentences concise (one sentence each).
6. Do not add unrelated information.
7. Output strictly as valid JSON in the format:
{
  "topic": "<topic name>",
  "queries": [
    "<variation 1>",
    "<variation 2>",
    "<variation 3>",
    "<variation 4>"
  ]
}
""".lstrip()


PERSONA_QUERY_PROMPT = """
Topic: {topic}
Associated terms: {terms}

Generate 4 natural-language search sentences for this topic and its associated terms, following the instructions, and return them in JSON.
""".lstrip()


PERSONA_LABEL_SYSTEM = "You are an AI assistant expert at labeling personas."
PERSONA_LABEL_PROMPT: str = """
# Instruction
Please classify the examples of personas by assigning the most appropriate labels.
Do not explain your reasoning or provide any additional commentary.
If the text is ambiguous or lacks sufficient information for classification, respond with "None".
Provide a list of 3 labels that best describe the text. Describe the main themes, topics, or categories that could describe the following types of personas. All the examples of personas must share the same set of labels.
Use clear, widely understood terms for labels. Avoid overly specific or obscure labels unless the text demands it.

## Examples of Personas
```
{personas}
```

## Output Format
Now, please give me the labels in JSON format, do not include any other text in your response:
```
{{
  "labels": ["label_1", "label_2", "label_3"]
}}
```
""".lstrip()

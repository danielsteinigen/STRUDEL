SYSTEM_PROBLEM = "You are an AI assistant expert at simulating user interactions."
SYSTEM_SOLUTION = (
    "You are an helpful AI assistant expert in generating solutions with functional descriptions for the problem statements of users."
)
SYSTEM_END2END = "You are an AI assistant expert at simulating user interactions and generating solutions with functional descriptions."


PROMPT_PROBLEM_2 = """
Generate a prompt with one problem statement that the following persona might ask to an AI assistant. Consider that the problem should be {complexity} complex and lead to a solution using a {category} with the Code Language {lang}. Do not mention and not use the Code Language in the problem statement.

Persona: {persona}

Now, please output your results in the format below by filling in the placeholders in <...>:
Problem: <problem statement>
""".lstrip()


PROMPT_SOLUTION_2 = """
Generate a code implementation as solution for the following problem statement of the given persona. Also provide a functional description of the implementation. Use the code language {lang}. Don't use Python or any other programming language. 
Don't add any comments in the code. The code should be syntactically correct and compile and run without errors. Do not mention the code language in the description. 

Persona: {persona}

Problem: {problem}

Now, please output your results in the format below by filling in the placeholders in <...>:

Functionality:
<functional description>

Code:
```
<code implementation>
```

Language: <used code language>
""".lstrip()


# adapted prompt END2END_4 with answer instead of language-generated
PROMPT_END2END_5 = """
Given a persona, generate a prompt with one problem statement that this persona might give to an AI assistant. The problem should deal with a {category} and should lead to a solution using the code language {lang}. Consider that the problem should be of {complexity} complexity.
Then generate a code implementation as solution to the problem statement using the specific defined code language.{code_instruct} Don't use Python or any other programming language. Don't add any comments in the code. The code should be syntactically correct and compile and run without errors.
Also provide a functional description of the solution implementation. Do not mention and not use the code language in the problem statement or in the functional description.{phrasing}
Finally, generate an answer with which the AI assistent will respond to the problem statement described in the persona's prompt.

Persona: {persona}

Now, please output your results in the format below by filling in the placeholders in <...>:

Problem:
<problem statement>

Functionality:
<functional description>

Code:
```
<code implementation>
```

Answer:
<assistant answer>
""".lstrip()


# prompt, where the user/persona provides the code to the assistant
PROMPT_END2END_10 = """
Given a persona, generate a prompt with one problem statement that this persona might give to an AI assistant as an command or question. The problem should deal with a {category} and should refer to a code implementation provided by the persona in the code language {lang}. Consider that the problem should be of {complexity} complexity.
Also generate the code implementation to which the problem statement refers using the specific defined code language.{code_instruct} Don't use Python or any other programming language. Don't add any comments in the code. The code should be syntactically correct and compile and run without errors.
Then generate an answer with which the AI assistent, who is an expert in this field, will respond to the problem statement described in the persona's prompt considering the related code of the persona. Do not mention and not use the code language in the problem statement or in the answer, instead refer to it as '{category}'.
It is important that the problem statement does not request generation of code, but rather assumes the code to be given and the question or command of the problem statement refers to a given code.{phrasing}

Persona: {persona}

Now, please output your results in the format below by filling in the placeholders in <...>:

Problem:
<problem statement>

Code:
```
<code implementation>
```

Answer:
<assistant answer>
""".lstrip()

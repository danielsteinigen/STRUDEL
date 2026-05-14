PROMPT_CONSISTENCY_SHORT = """
{question}

{subject}: {text}{resp_format}
""".lstrip()

PROMPT_ASSOCIATION_SHORT = """
{question}

(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}{resp_format}
""".lstrip()

USER_DESCRIPTION = [
    "Describe the functionality of what is depicted in this image.",
    "What can be seen here in the visualization?",
    "Summarize the function of what is shown in the visualization.",
    "What is this visualization intended to show?",
    "Describe how the different parts of this visualization contribute to the overall functionality.",
    "Summarize the problem this visualization addresses and the solution it depicts.",
    "What is the main function or purpose of what is shown here?",
    "Explain what this visualization represents and how it works.",
    "What process or concept is being demonstrated in this visualization?",
    "Explain the key functionality that this visualization is conveying.",
]
USER_CAPTION = [
    "Generate a caption about this visualization.",
    "What could be a good caption for this image?",
    "Write a descriptive caption for the given diagram.",
    "Create a caption that explains this visualization.",
    "Can you generate a meaningful caption for this picture?",
    "Produce a clear caption that summarizes the content of the image.",
    "Give me a caption that describes what this diagram shows.",
    "I need an informative caption for the visualization.",
    "Formulate a caption that captures the main idea of this image.",
    "Please provide a caption that explains the content of this picture.",
]
USER_CODE = [
    "Convert this visualization into corresponding code.",
    "Translate the content of this diagram into code.",
    "Generate code that represents this image.",
    "Provide the code version of this visualization.",
    "Turn this diagram into its equivalent code representation.",
    "Write code that reproduces the structure shown in this image.",
    "Express the content of this visualization in code.",
    "Can you generate the code that matches this diagram?",
    "What is the code representation of this picture?",
    "Transform this visualization into a code-based format.",
]
USER_ASSOCIATION_PERSONA = [
    "Among the following personas, which one deals with the content shown in the picture?",
    "Among the following personas, identify which one is the most relevant to the content of the image.",
    "Which persona from the given list corresponds to the visualization shown?",
    "Out of the listed personas, which one is associated with what the picture represents?",
    "Which persona best aligns with the scenario illustrated in the image?",
    "From the options provided, identify the persona that deals with the content of this visualization.",
    "Which persona is correctly represented by the content of the diagram?",
    "Which persona corresponds logically with the subject matter of the visualization?",
    "Based on the image, which persona from the list is most applicable?",
    "Which persona option best matches the content displayed in the picture?",
]
USER_ASSOCIATION_CAPTION = [
    "Among the following captions, which one matches the content shown in the picture?",
    "Among the following captions, identify which one describes the content of this image best.",
    "Which caption from the provided list corresponds to the visualization shown?",
    "Out of the listed captions, which one matches the subject matter of the picture?",
    "From the following captions, identify the one that correctly fits the image content.",
    "Which caption aligns with what is represented in this visualization?",
    "Among the given captions, which one explains the content of the image?",
    "Which caption corresponds to the idea conveyed by the diagram?",
    "Based on the content shown, which caption is the correct match?",
    "Which of the listed captions is associated with this visualization?",
]
USER_CONSISTENCY_PROBLEM = [
    "Given a problem statement and a visual representation as picture, does the content shown in the picture represent a solution to the given problem?",
    "Given a problem statement and an image, identify if the visualization provide a solution to the stated problem.",
    "Does the content of this image represent a valid solution for the given problem statement?",
    "Based on the provided problem statement and the visual diagram, is the diagram illustrating a solution to that problem?",
    "Given the problem statement, is the image associated with its resolution?",
    "Is the visualization an appropriate solution to the specified problem?",
    "Identify if the diagram correspond to targeting the problem described in the statement.",
    "Given the problem statement, does the picture address a solution?",
    "Is the content of the image intended to address the given problem?",
    "Does the diagram provide an answer or resolution to the described problem statement?",
]
USER_CONSISTENCY_SOLUTION = [
    "Given a functional description and a visual representation as picture, does the description correspond logically with the content shown in the picture?",
    "Given a functional description and a visualization, does the image logically correspond to the described function?",
    "Does the content of the image align with the provided functional description?",
    "Based on the functional description, is the diagram consistent with what is being described?",
    "Given the description of functionality, does this image illustrate that function?",
    "Does the visual representation match the behavior described in the functional statement?",
    "Does the diagram illustrate the operation or function described in the text?",
    "Given a functional description, is the content of the visualization a correct match?",
    "Does this image represent the process or function stated in the description?",
    "Is the described functionality reflected in the diagram provided?",
]
ASSISTANT_CODE = [
    "Here is a corresponding code implementation using {language}.",
    "I’ve translated the visualization into {language} code below.",
    "Here is the equivalent representation expressed in {language}.",
    "This diagram can be implemented with the following {language} code.",
    "Below is a {language} implementation that matches the visualization.",
    "Here’s how the content of the image can be represented in {language}.",
    "I generated the following {language} code based on the diagram.",
    "This is the {language} version of the visualization.",
    "Here’s the code in {language} that corresponds to the image.",
    "The visualization can be expressed in {language} as shown below.",
]
ASSISTANT_STRUCTURAL = [
    "{cnt}\nThere are {cnt} {component} in this image.",
    "{cnt}\nThe image contains {cnt} {component}",
    "{cnt}\nI can identify {cnt} {component} in the visualization.",
    "{cnt}\nThis diagram shows {cnt} distinct {component}",
    "{cnt}\nA total of {cnt} {component} are represented here.",
    "{cnt}\nThe visualization includes {cnt} {component}.",
    "{cnt}\nI count {cnt} {component} present in the image.",
    "{cnt}\nOverall, the diagram illustrates {cnt} {component}.",
    "{cnt}\nThis image depicts {cnt} {component} in total.",
    "{cnt}\nThere appear to be {cnt} {component} included in the visualization.",
]
ASSISTANT_ASSOCIATION = [
    "{answer}\nThis is a visualization of a {category}. So the {subject} {answer} is associated with this image.",
    "{answer}\n{answer} is the correct {subject} that matches the content of this image. The diagram represents a {category}.",
    "{answer}\nThis visualization shows a {category}. This image corresponds to the {subject} {answer}.",
    "{answer}\nThe image depicts a {category}. The most relevant {subject} for this image is {answer}.",
    "{answer}\nBased on the content, {answer} is the {subject} linked to this visualization. This is an illustration of a {category}.",
    "{answer}\nThis visualization depicts a {category}. The visualization best represents the {subject} {answer}.",
    "{answer}\nAmong the options, the {subject} {answer} matches this image. Shown here is a {category} visualization.",
    "{answer}\nA {category} is illustrated in this diagram. This diagram is associated with the {subject} {answer}.",
    "{answer}\nHere we see a {category}. The content of the image corresponds to the {subject} {answer}.",
    "{answer}\nThe content represents a {category}. The correct {subject} for this visualization is {answer}.",
]
ASSISTANT_ASSOCIATION_NONE = [
    "None\nNo {subject} is associated with the content of this image. This diagram is a representation of a {category}.",
    "None\nNone of the provided {subject}s correspond to this image. The visualization illustrates a {category}.",
    "None\nThe image does not match any of the given {subject}s. This picture shows a {category}.",
    "None\nThere is no {subject} from the provided options that matches this visualization. This chart is an example of a {category}.",
    "None\nNo suitable {subject} can be identified from the given options. The content of this image corresponds to a {category}.",
    "None\nNo relevant {subject} from the list can be linked to this visualization. Depicted here is a {category}.",
    "None\nThis visualization is not associated with any of the provided {subject}s. What is displayed is a {category}.",
    "None\nNone of the listed {subject}s are applicable to the content of this image. Illustrated here is a {category}.",
    "None\nThe content of the image does not correspond to any listed {subject}. Pictured here is a {category}.",
    "None\nThe image does not represent any {subject} from the provided list. A visualization of a {category} is shown here.",
]
ASSISTANT_CONSISTENCY = [
    "{answer}\n{answer}, the {subject} is {neg}associated with this image. This figure illustrates a {category}.",
    "{answer}\n{answer}, the {subject} matches {neg}the content of this image. The content of this image corresponds to a {category}.",
    "{answer}\n{answer}, this image does {neg}represent the {subject}. This diagram corresponds to a {category}.",
    "{answer}\n{answer}, the visualization does {neg}illustrate the {subject}. This graphic represents a {category}.",
    "{answer}\n{answer}, this image does {neg}address the {subject}. This diagram shows an instance of a {category}.",
    "{answer}\n{answer}, the {subject} is {neg}captured by this visualization. This figure is an example of a {category}.",
    "{answer}\n{answer}, the content of the image does {neg}correspond to the {subject}. A {category} is represented in this visualization.",
    "{answer}\n{answer}, the {subject} is {neg}expressed in this image. Displayed in the image is a {category}.",
    "{answer}\n{answer}, this visualization is {neg}consistent with the {subject}. This diagram corresponds to a {category} type.",
    "{answer}\n{answer}, the {subject} is {neg}reflected in the content of the visualization. What the image conveys is a {category}.",
]

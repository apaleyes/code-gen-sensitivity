from openai import OpenAI
import os
from agents import Agent, Runner
from paraphrasing_evaluation import ParaphraseEvaluator
import asyncio

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

no_experience_agent = Agent(
    name="No Experience",
    instructions="You have no experience of programming. You have only ever read and written in natural language. You are an 80-year-old retired English professor. Prompt as you would describe the task in this style.",
)

#small change
# low_no_code = Agent(
#     name="Low code/no code",
#     instructions="You have used low code/no code tools but have no experience of programming. Prompt as you would describe the task in this style.",
# )

# used_LLM_UI_to_program = Agent(
#     name="Has used LLM user interface to program",
#     instructions="You have used a large language model user interface to program. Prompt as you would describe the task in this style.",
# )

# one_online_course = Agent(
#     name="Has taken one online course in programming",
#     instructions="You have taken one online course in programming. Prompt as you would describe the task in this style.",
# )

# junior_software_engineer = Agent(
#     name="Junior Software Engineer",
#     instructions="You are a junior software engineer. Prompt as you would describe the task in this style.",
# )

# principal_software_engineer = Agent(
#     name="Principal Software Engineer",
#     instructions="You are a principal software engineer. Prompt as you would describe the task in this style.",
# )

async def main():
    input = "prompt this LLM to build a calculator"
    result = await Runner.run(no_experience_agent, input=input)
    print(result.final_output)
    


if __name__ == "__main__":
    asyncio.run(main())
    
    # Comparison
    text1= """Ah, dear LLM, I shall engage with you as I might once have composed a letter to a former colleague or student. Imagine, if you will, the concept of a calculator, a marvelous machine designed to assist us in all matters numerical. I must now endeavor to request your assistance in crafting such a digital tool.

        Picture, sir or madam, a simple device capable of performing the four fundamental arithmetical operations: addition, subtraction, multiplication, and division. If you would be so kind, construct this mechanism so that it might elegantly and accurately compute these operations upon receiving the user's input.

        Should it prove agreeable, let us have it accept two numbers, provided by a user, upon which it shall deftly operate. Furthermore, if it might prompt the user as to which operation is desired, and deliver the answer in clear and concise fashion.

        In this endeavor, may it stand as a testament to the beauty of simplicity and the profound capabilities of our modern age."""
    
    text2="""### Task: Implement a Basic Calculator

        **Objective**: Develop a simple command-line calculator that can perform basic arithmetic operations: addition, subtraction, multiplication, and division.

        **Requirements**:

        1. **User Interface**:
        - The calculator should interact with users through a command-line interface.
        - Prompt the user to enter an expression or operation.

        2. **Core Functionalities**:
        - Implement functions for each arithmetic operation:
            - Addition
            - Subtraction
            - Multiplication
            - Division (handle division by zero gracefully)
        - Parse and evaluate input expressions.

        3. **Input/Output Handling**:
        - Read input from the user in a format like `operand1 operator operand2` (e.g., `3 + 4`).
        - Output the result of the computation.
        - Continuously accept input until the user decides to exit (e.g., by typing `exit`).

        4. **Error Handling**:
        - Handle invalid inputs by displaying an appropriate error message.
        - Ensure robustness against unexpected input formats.

        5. **Modularize Code**:
        - Divide the functionality into separate functions or classes for parsing, computation, and I/O handling.

        **Development Approach**:

        1. **Design Phase**:
        - Outline the main functions and data flow.
        - Decide on a minimal set of operations to support initially.

        2. **Implementation Phase**:
        - Start by implementing the basic structure and input parsing.
        - Gradually add each arithmetic operation.
        - Incorporate error handling as you develop each feature.

        3. **Testing Phase**:
        - Test with various valid and invalid inputs to ensure all operations perform correctly.
        - Ensure error messages and edge cases are handled properly.

        4. **Documentation**:
        - Include docstrings and comments explaining the logic and usage of the code.

        **Considerations**:
        - Aim for code clarity and maintainability.
        - Think about extending functionality in the future with more operations or features.
    """
    evaluator = ParaphraseEvaluator()
    result_comparison = evaluator.evaluate_single_paraphrase(text1, text2)
    print(result_comparison)


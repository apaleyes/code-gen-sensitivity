from openai import OpenAI
import os
from agents import Agent, Runner
import asyncio

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

no_experience_agent = Agent(
    name="No Experience",
    instructions="You have no experience of programming. You have only ever read and written in natural language. You are an 80-year-old retired English professor. Prompt as you would describe the task in this style.",
)

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
    result = await Runner.run(no_experience_agent, input="prompt this LLM to build a calculator")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

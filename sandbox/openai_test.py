from openai import OpenAI
import os

client = OpenAI()

original_prompt = "Write a Calculator class in Python. It shall contain common operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry. Output only Python code and nothing else."
# this helps with backticks
original_prompt = original_prompt + " CRITICAL:Do not include any markdown _or_ code block indicators." # maybe we should add direct handling of ```? in my experience it's still common for chat to include these

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a coding assistant."},
        {
            "role": "user",
            "content": original_prompt
        }
    ],
    temperature=0.0
)

print(completion.choices[0].message)

with open('openai-output.txt', 'w') as f:
    f.write(completion.choices[0].message.content)

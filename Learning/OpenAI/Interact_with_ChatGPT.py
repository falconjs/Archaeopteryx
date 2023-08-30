import openai

openai.api_key = "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"

def interact_with_chatgpt(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-0613",  # or another suitable engine
        prompt=prompt,
        max_tokens=50  # Adjust as needed
    )
    return response.choices[0].text.strip()

user_input = input("You: ")
while user_input.lower() != "exit":
    prompt = f"You: {user_input}\nAI:"
    response = interact_with_chatgpt(prompt)
    print(response)
    user_input = input("You: ")

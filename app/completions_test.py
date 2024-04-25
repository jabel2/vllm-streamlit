from openai import OpenAI

client = OpenAI(
    base_url = "http://localhost:3000/v1",
    api_key = "n/a",
)

models = client.models.list()
print("models", models.model_dump_json(indent=2))
model = models.data[0].id

stream = client.chat.completions.create(
    model=model,
    messages=[{"role":"user", "content":"How do you run a streamlit application and on what port?"}],
    stream=True, 
    extra_body={"stop_token_ids":[128001, 128009]}
)

for chunk in stream:
    text = chunk.choices[0].delta.content
    if text:
        print(text, flush=True, end="")
print('\n')
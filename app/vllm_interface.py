from langchain_openai import ChatOpenAI
import warnings
warnings.simplefilter("ignore")

llm = ChatOpenAI(
    base_url="http://localhost:3000/v1",
    api_key="n/a",
    #model="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    model="MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ",
    temperature=0,
    max_tokens=None,
    streaming=True,
    extra_body={"stop_token_ids":[128001, 128009]}
)
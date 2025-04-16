from langchain.chat_models import init_chat_model
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def text_analyzer(text_to_analyze):
  #######------------- Prompt Template-------------####
  temp = """List the key points with details from the context:
  Context: {text}
  """

  # Initialize LLM
  llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0.3)

  pt = PromptTemplate(
    input_variables=["text"],
    template=temp)

  prompt_to_llm = LLMChain(llm=llm, prompt=pt)

  result = prompt_to_llm.invoke(text_to_analyze) 

  return result

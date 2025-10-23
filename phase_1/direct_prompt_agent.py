# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TODO: 2 - Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key)
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

print("The DirectPromptAgent used general world knowledge embedded in the GPT-3.5-turbo model to answer the prompt. Since the model has been trained on a wide range of publicly available data, it knows that the capital of France is Paris. The agent simply forwarded the prompt without any persona or external knowledge.")

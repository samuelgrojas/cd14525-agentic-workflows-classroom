from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmentedpromptagent = AugmentedPromptAgent(openai_api_key, persona)

augmented_agent_response = augmentedpromptagent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)


# - What knowledge the agent likely used to answer the prompt.
# - How the system prompt specifying the persona affected the agent's response.

# The agent likely used general world knowledge embedded in the LLM to answer the prompt,
# specifically that the capital of France is Paris. This is common factual information
# that the model has been trained on.

# The system prompt specifying the persona ("You are a college professor; your answers always start with: 'Dear students,'")
# influenced the tone and style of the response. Instead of a plain factual answer,
# the agent's reply was likely formal, educational, and began with "Dear students," as instructed,
# simulating the behavior of a college professor addressing a class.

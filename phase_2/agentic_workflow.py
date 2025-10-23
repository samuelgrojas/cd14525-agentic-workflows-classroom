# agentic_workflow.py

from workflow_agents.base_agents import (ActionPlanningAgent,
                                        KnowledgeAugmentedPromptAgent,
                                        EvaluationAgent,
                                        RoutingAgent)

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
path = os.path.join(os.path.dirname(__file__), "Product-Spec-Email-Router.txt")
with open(path, "r", encoding="utf-8") as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

action_planning_agent = ActionPlanningAgent(
    knowledge=knowledge_action_planning,
    openai_api_key=openai_api_key
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    + product_spec
)

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
    openai_api_key=openai_api_key
)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_product_manager = (
    "The answer should be user stories that follow this exact structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    openai_api_key=openai_api_key,
    max_interactions=3
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
    openai_api_key=openai_api_key
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)

program_manager_evaluation_agent = EvaluationAgent(
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent,
    openai_api_key=openai_api_key,
    max_interactions=3
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
    openai_api_key=openai_api_key
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)

development_engineer_evaluation_agent = EvaluationAgent(
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,
    openai_api_key=openai_api_key,
    max_interactions=3
)


# Routing Agent
routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=[
    {
        "name": "product manager agent",
        "description": "Answer a question about product management",
        "func": lambda x: product_manager_knowledge_agent.respond(x)
    },
    {
        "name": "program manager agent",
        "description": "Answer a question about program management",
        "func": lambda x: program_manager_knowledge_agent.respond(x)
    },
    {
        "name": "development engineer agent",
        "description": "Answer a question about development engineering",
        "func": lambda x: development_engineer_knowledge_agent.respond(x)
    }
]
)

# Job function persona support functions
def product_manager_support_function(query: str) -> str:
    response = product_manager_knowledge_agent.respond(query)
    evaluation = product_manager_evaluation_agent.evaluate(response)
    return evaluation["final_response"]

def program_manager_support_function(query: str) -> str:
    response = program_manager_knowledge_agent.respond(query)
    evaluation = program_manager_evaluation_agent.evaluate(response)
    return evaluation["final_response"]

def development_engineer_support_function(query: str) -> str:
    response = development_engineer_knowledge_agent.respond(query)
    evaluation = development_engineer_evaluation_agent.evaluate(response)
    return evaluation["final_response"]

# Run the workflow
print("\n*** Workflow execution started ***\n")

# Prompt principal
workflow_prompt = "Create a development plan for the product described in the following product specification:\n" + product_spec
print(f"Task to complete in this workflow:\n{workflow_prompt}\n")

# Extraer pasos como lista de strings
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"Workflow steps defined:\n{workflow_steps}\n")

# Ejecutar cada paso con enrutamiento
project_plan = {
    "user_stories": [],
    "features": [],
    "tasks": [],
    "timeline": [],
    "review": []
}

for step in workflow_steps:
    step_lower = step.lower()
    if step_lower.startswith(("development plan", "identify user stories", "group user stories", "define tasks", "create a development timeline", "review and iterate")):
        continue  # Saltar encabezados

    print(f"\nExecuting step: {step}")
    
    if "User Story" in step:
        response = product_manager_support_function(step)
        project_plan["user_stories"].append(response)
    elif "Feature" in step:
        response = program_manager_support_function(step)
        project_plan["features"].append(response)
    elif "Task" in step:
        response = development_engineer_support_function(step)
        project_plan["tasks"].append(response)
    elif "Month" in step:
        project_plan["timeline"].append(step)
    elif "review" in step_lower or "feedback" in step_lower:
        project_plan["review"].append(step)
    else:
        response = routing_agent.route(step)
        project_plan.setdefault("others", []).append(response)

# Mostrar el plan final en consola
def print_project_plan(plan):
    print("\n*** Final Project Plan ***\n")

    for feature, story, task in zip(plan["features"], plan["user_stories"], plan["tasks"]):
        print("🧩 Feature:")
        print(f"- {feature.strip()}\n")

        print("👤 User Story:")
        print(f"{story.strip()}\n")

        print("🧰 Development Task:")
        print(f"{task.strip()}\n")
        print("---\n")

    if plan.get("timeline"):
        print("📅 Development Timeline:")
        for item in plan["timeline"]:
            print(f"- {item.strip()}")
        print("\n")

    if plan.get("review"):
        print("🔍 Review and Iteration:")
        for item in plan["review"]:
            print(f"- {item.strip()}")
        print("\n")

print_project_plan(project_plan)

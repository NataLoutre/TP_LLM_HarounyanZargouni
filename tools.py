from dotenv import load_dotenv
from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel, tool
from langfuse import get_client
from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client
import json
import os

#Configuration Langfuse (optionnel si déjà dans .env)
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com/"

load_dotenv()
groq_client = Groq()

# -----------------------------------------------------------------------
# --- 4.1 - Definissez 3 outils ---

# model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile")
model = LiteLLMModel(model_id="gemini/gemini-3-pro-preview")
tools = [
    {
        "name": "check_fridge",
        "description": "Retourne la liste des ingrédients disponibles dans le frigo.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_recipe",
        "description": "Retourne une recette détaillée pour un plat donné.",
        "parameters": {
            "type": "object",
            "properties": {
                "dish_name": {
                    "type": "string",
                    "description": "Nom du plat dont on veut la recette"
                }
            },
            "required": ["dish_name"]
        }
    },
    {
        "name": "check_dietary_info",
        "description": "Retourne les informations nutritionnelles et allergéniques d'un ingrédient.",
        "parameters": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": "Nom de l'ingrédient"
                }
            },
            "required": ["ingredient"]
        }
    }
]

## Définitions du tools (implémentations simulées)
def check_fridge():
    """
    Retourne une liste d'ingrédients disponibles dans le frigo (données simulées).
    """
    return [
        "oeufs",
        "lait",
        "fromage",
        "tomates",
        "poulet",
        "riz",
        "oignons",
        "huile d'olive"
    ]

def get_recipe(dish_name: str):
    """
    Retourne une recette détaillée pour un plat donné (données simulées).
    """
    recipes = {
        "omelette": {
            "ingredients": ["oeufs", "fromage", "huile d'olive", "sel", "poivre"],
            "steps": [
                "Battre les oeufs dans un bol",
                "Chauffer l'huile dans une poêle",
                "Verser les oeufs battus",
                "Ajouter le fromage",
                "Cuire 3 à 4 minutes et servir"
            ],
            "prep_time_minutes": 10,
            "difficulty": "facile"
        },
        "riz au poulet": {
            "ingredients": ["riz", "poulet", "oignons", "huile d'olive", "sel"],
            "steps": [
                "Faire revenir les oignons dans l'huile",
                "Ajouter le poulet et le faire dorer",
                "Ajouter le riz",
                "Ajouter de l'eau et laisser cuire 15 minutes"
            ],
            "prep_time_minutes": 30,
            "difficulty": "moyenne"
        }
    }

    return recipes.get(
        dish_name.lower(),
        {"error": f"Aucune recette trouvée pour '{dish_name}'"}
    )


def check_dietary_info(ingredient: str):
    """
    Retourne les informations nutritionnelles et allergéniques d'un ingrédient (données simulées).
    """
    dietary_db = {
        "oeufs": {
            "calories_per_100g": 155,
            "protein_g": 13,
            "fat_g": 11,
            "allergens": ["oeufs"]
        },
        "lait": {
            "calories_per_100ml": 42,
            "protein_g": 3.4,
            "fat_g": 1,
            "allergens": ["lactose"]
        },
        "fromage": {
            "calories_per_100g": 350,
            "protein_g": 25,
            "fat_g": 28,
            "allergens": ["lactose"]
        },
        "poulet": {
            "calories_per_100g": 165,
            "protein_g": 31,
            "fat_g": 3.6,
            "allergens": []
        }
    }

    return dietary_db.get(
        ingredient.lower(),
        {"error": f"Aucune information nutritionnelle trouvée pour '{ingredient}'"}
    )

# -----------------------------------------------------------------------
# --- 4.2 - Boucle de tool calling manuelle ---

### Agent d'appel d'outils 
@observe()
def tool_calling_agent(user_message: str) -> str:

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided tools when needed to answer questions accurately."
        },
        {"role": "user", "content": user_message}
    ]

    for iteration in range(5):  # Max 5 iterations to avoid infinite loops
        print(f"\n  [Iteration {iteration + 1}]")

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
        )

        message = response.choices[0].message

        # If no tool calls, the LLM is giving its final answer
        if not message.tool_calls:
            print(f"  Final answer ready.")
            return message.content

        # Process each tool call
        messages.append(message)  # Add assistant's tool-call message to history

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"  Tool call: {name}({args})")

            # Execute the tool
            func = TOOL_REGISTRY.get(name)
            if func:
                result = func(**args)
            else:
                result = f"Error: unknown tool '{name}'"

            print(f"  Result: {result}")

            # Add tool result to message history
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Error: max iterations reached"


TOOL_REGISTRY = { 
    "check_fridge": check_fridge, 
    "get_recipe": get_recipe, 
    "check_dietary_info": check_dietary_info }


# -----------------------------------------------------------------------
# --- 4.3 - Migration vers smolagents ---

# model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile")
model = LiteLLMModel(model_id="gemini/gemini-3-pro-preview")

@tool
def check_fridge(): return [ "oeufs", "lait", "fromage", "tomates", "poulet", "riz", "oignons", "huile d'olive" ]

@tool
def get_recipe(dish_name: str):
    recipes = { "omelette": { "ingredients": ["oeufs", "fromage", "huile d'olive", "sel", "poivre"], 
                        "steps": [ "Battre les oeufs dans un bol", "Chauffer l'huile dans une poêle", "Verser les oeufs battus", "Ajouter le fromage", "Cuire 3 à 4 minutes et servir" ], 
                        "prep_time_minutes": 10, 
                        "difficulty": "facile" }, 
            "riz au poulet": { "ingredients": ["riz", "poulet", "oignons", "huile d'olive", "sel"], 
                        "steps": [ "Faire revenir les oignons dans l'huile", "Ajouter le poulet et le faire dorer", "Ajouter le riz", "Ajouter de l'eau et laisser cuire 15 minutes" ], 
                        "prep_time_minutes": 30,
                        "difficulty": "moyenne" } } 
    return recipes.get( dish_name.lower(), {"error": f"Aucune recette trouvée pour '{dish_name}'"} )

@tool
def check_dietary_info(ingredient: str):
    dietary_db = { "oeufs": { "calories_per_100g": 155, "protein_g": 13, "fat_g": 11, "allergens": ["oeufs"] },
                  "lait": { "calories_per_100ml": 42, "protein_g": 3.4, "fat_g": 1, "allergens": ["lactose"] },
                  "fromage": { "calories_per_100g": 350, "protein_g": 25, "fat_g": 28, "allergens": ["lactose"] },
                  "poulet": { "calories_per_100g": 165, "protein_g": 31, "fat_g": 3.6, "allergens": [] } }
    return dietary_db.get( ingredient.lower(), {"error": f"Aucune information nutritionnelle trouvée pour '{ingredient}'"} )

def run_code_agent():
    agent = CodeAgent(model=model,
                      tools=[check_fridge, get_recipe, check_dietary_info],
                      max_iterations=5)
    result = agent.run("Quels plats puis-je faire avec du poulet et du riz ?") 
    print("Résultat final :", result)


if __name__ == "__main__":
    run_code_agent()
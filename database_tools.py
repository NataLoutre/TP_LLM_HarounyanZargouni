from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, Tool, ManagedAgent
from langfuse import observe, get_client
import litellm
import json

from tools import check_fridge, get_recipe, check_dietary_info

load_dotenv()

# --- PARTIE 5 - LE RESTAURANT INTELLIGENT ---
# -----------------------------------------------------------------------
# --- 5.1 - Outil de base de donnees ---

# --- Langfuse tracing for LiteLLM (v3 — OpenTelemetry) ---
litellm.callbacks = ["langfuse_otel"]

model_llm = LiteLLMModel(model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct")
class MenuDatabaseTool(Tool):

    name = "database_lookup"
    description = "Look up a product by name in the database. Returns price and stock."
    inputs = {
        "product_name": {
            "type": "string",
            "description": "The name of the product to look up."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # Simulate a database
        self.products = {
            "omelette": {"nom": "Omelette", "prix": 15,"prep_time": 10, "allergènes": ["oeufs"],"catégorie": "petitdéjeuner"},
            "salade césar": {"nom": "Salade César", "prix": 12,"prep_time": 15, "allergènes": ["lait", "poisson"],"catégorie": "déjeuner"},
            "pâtes bolognaises": {"nom": "Pâtes Bolognaises", "prix": 18,"prep_time": 30, "allergènes": ["gluten"],"catégorie": "dîner"},
            "soupe de légumes": {"nom": "Soupe de Légumes", "prix": 10,"prep_time": 20, "allergènes": [],"catégorie": "entrée"},
            "tarte aux pommes": {"nom": "Tarte aux Pommes", "prix": 8,"prep_time": 45, "allergènes": ["gluten", "lait"],"catégorie": "dessert"},
            "smoothie aux fruits": {"nom": "Smoothie aux Fruits", "prix": 6,"prep_time": 5, "allergènes": ["fruits"],"catégorie": "boisson"},
            "quiche lorraine": {"nom": "Quiche Lorraine", "prix": 14,"prep_time": 25, "allergènes": ["gluten", "lait", "oeufs"],"catégorie": "déjeuner"},
            "risotto aux champignons": {"nom": "Risotto aux Champignons", "prix": 16,"prep_time": 25, "allergènes": ["gluten", "lait"],"catégorie": "dîner"},
            "salade de quinoa": {"nom": "Salade de Quinoa", "prix": 12,"prep_time": 15, "allergènes": [],"catégorie": "entrée"},
            "crème brûlée": {"nom": "Crème Brûlée", "prix": 9,"prep_time": 40, "allergènes": ["lait", "oeufs"],"catégorie": "dessert"}
        }

    def forward(self, categorie=None, prix_max=None, sans_allergene=None):
        res = [p for p in self.products.values() if (not categorie or p['catégorie'] == categorie) and (not prix_max or p['prix'] <= prix_max)]
        if sans_allergene:
            res = [p for p in res if sans_allergene not in p['allergènes']]
        return json.dumps(res)

# -----------------------------------------------------------------------
# --- 5.2 - Agent avec planification ---

@tool
def calculate(expression: str) -> str:
    """Fait des calculs. Args: expression: l'opération."""
    try: return str(eval(expression))
    except: return "Erreur"

restaurant_agent = CodeAgent(tools=[MenuDatabaseTool(), calculate], model=model_llm, planning_interval=2)

# -----------------------------------------------------------------------
# --- 5.3 - Agent conversationnel ---

@observe(name="run_partie_5")
def run_restaurant_session():
    
    print(restaurant_agent.run("Budget 60€, un sans gluten. Proposez un menu.", reset=False))
    print(restaurant_agent.run("L'addition détaillée ?", reset=False))


# --- PARTIE 6 - L'EMPIRE CHEFBOT ---
# -----------------------------------------------------------------------

nutritionist = ManagedAgent(agent=CodeAgent(tools=[tool(check_dietary_info)], model=model_llm), 
                            name="nutritionist", description="Vérifie allergènes")
chef = ManagedAgent(agent=CodeAgent(tools=[tool(check_fridge), tool(get_recipe)], model=model_llm), 
                    name="chef", description="Recettes et frigo")
budget = ManagedAgent(agent=CodeAgent(tools=[calculate, MenuDatabaseTool()], model=model_llm), 
                     name="budget", description="Prix et calculs")

manager = CodeAgent(tools=[], model=model_llm, managed_agents=[nutritionist, chef, budget])

@observe(name="Partie 6 Raphaelgabriel - Empire")
def run_empire_test():
    requete = """
    Je recois 8 personnes samedi soir. Parmi eux : 2 vegetariens, 1 intolerant au gluten,
    1 allergique aux fruits a coque. Budget total : 120 euros.
    Je veux un aperitif, une entree, un plat principal et un dessert.
    Il faut que tout le monde puisse manger chaque service.
    """
    print(manager.run(requete))
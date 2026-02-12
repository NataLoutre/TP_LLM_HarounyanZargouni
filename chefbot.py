import json
import os
from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client

#Chargement des variables d'environnement
load_dotenv()

#Configuration Langfuse (optionnel si déjà dans .env)
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com/"

#Initialisation explicite du client Langfuse
langfuse = get_client()

#Client Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------------------------------------------------
# --- PARTIE 1 : PREMIER CONTACT ---
@observe(name="ask_chef")
def ask_chef(question: str, temperature: float = 0.7) -> str:
    """
    Appel LLM + monitoring Langfuse et température variable.
    """

    system_prompt = (
        "Tu es ChefBot, un chef cuisinier français spécialisé en cuisine de saison. "
        "Tu es concis, professionnel et passionné. Ton objectif est de valoriser les produits frais du moment et de donner des conseils techniques précis."
    )

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=temperature,
    )

    return completion.choices[0].message.content


@observe(name="run_partie_1")
def run_temperature_tests():
    """1.3 - Jouez avec la temperature"""

    # Ajout tags et metadata à la trace courante
    langfuse.update_current_trace(
        tags=["Partie 1", "Groupe_Natalène_Yacine"],
        metadata={"experiment": "temperature_variation"}
    )

    question = "J'ai récupéré des poireaux et des noix du marché ce matin. Qu'est-ce que je peux cuisiner avec pour ce soir ?"
    temps = [0.1, 0.7, 1.2]

    for t in temps:
        print(f"\n--- TEST DE TEMPERATURE : {t} ---")
        reponse = ask_chef(question, temperature=t)
        print(reponse)



# if __name__ == "__main__":
#     run_temperature_tests()

#     #  Envoi effectif des traces à Langfuse
#     langfuse.flush()
#     print("\nTraces envoyées à Langfuse.")


"""
On observe clairement l’effet de la température sur la variabilité des sorties :

À 0.1, la réponse est très structurée, déterministe et classique assez classique :
liste d’ingrédients précise, étapes numérotées, ton plutôt neutre et peu d’originalité. 
Le modèle reste dans un schéma “recette standard”, très stable.

À 0.7, on observe une légère diversification : 
les ingrédients varient (œufs, crème, muscade), modification des temps de cuisson et préparation
plus complexe. La structure reste similaire mais le contenu devient un peu plus élaboré et moins figé.

À 1.2, la réponse est plus synthétique et légèrement plus libre dans la formulation :
La recette est moins détaillée, plus narrative, avec une simplification des quantités. 
Toutefois, la créativité reste modérée : le plat proposé ne change pas.

Impression générale :
La température influence bien la formulation et le niveau de détail, mais pas le choix global 
du plat (toujours une tarte aux poireaux et noix). Cela suggère que le prompt contraint fortement 
la génération. Pour observer un effet plus marqué, il faudrait soit un prompt plus ouvert, 
soit comparer plusieurs générations par température pour constater les différences.

"""

# -----------------------------------------------------------------------
# --- PARTIE 2 : LE CHEF QUI RÉFLÉCHIT ---

@observe(name="get_plan")
def get_plan(constraints: str):
    """2.1 : Planificateur de menu (décomposition en étapes)"""
    prompt = f"""Analyse ces contraintes : {constraints}.
            Décompose la création d'un menu de semaine en 3 étapes distinctes.
            Réponds UNIQUEMENT en JSON avec le format suivant:
            {{"etapes": ["etape1", "etape2", "etape3"]}}"""

#            TEST ERREUR JSON :
#            Ne décompose pas le menu de la semaine.
#            Ne réponds pas en JSON, juste en texte brut selon ce format : etapes = 

    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            parsed = json.loads(completion.choices[0].message.content)

            # Validation métier
            if not isinstance(parsed, dict) or "etapes" not in parsed:
                raise ValueError("JSON structure invalid: missing 'etapes'")

            return parsed

        except Exception as e:   # capture TOUT
            try:
                langfuse.event(
                    name="json_parsing_error",
                    level="ERROR",
                    input=completion.choices[0].message.content if 'completion' in locals() else None,
                    metadata={
                        "error": str(e),
                        "attempt": attempt + 1,
                        "function": "get_plan"
                    }
                )
                langfuse.flush()
            except Exception:
                print("Langfuse logging failed")
                print("Original error:", e)

            if attempt == 1:
                raise

    # 
    raise RuntimeError("get_plan failed after retry — no valid JSON returned")



@observe(name="execute_step")
def execute_step(step_name: str, context: str):
    """2.2 : Planificateur de menu (exécution d'une étape)"""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"Contexte actuel : {context}"},
            {"role": "user", "content": f"Etape suivante : {step_name}"}
        ]
    )
    return completion.choices[0].message.content


@observe(name="plan_weekly_menu")
def plan_weekly_menu(constraints: str) -> str:
    """Fonction principale de la Partie 2"""

    # 1. Planification
    plan = get_plan(constraints)

    # 2. Exécution des étapes
    results = []
    current_context = f"Contraintes accumulées : {constraints}"

    for step in plan.get("etapes", []):
        res = execute_step(step, current_context)
        results.append(res)
        current_context += f"\n{res}"

    # 3. Synthèse finale
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Tu es un créateur de menus."},
            {"role": "user", "content": f"Assemble le tout en un menu cohérent : {' '.join(results)}"}
        ]
    )

    return completion.choices[0].message.content

@observe(name="run_partie_2")
def run_tests():
    """Test Partie 2"""

    # Tags spécifiques à la partie 2
    langfuse.update_current_trace(
        tags=["Partie 2", "Groupe_Natalène_Yacine"],
        metadata={"experiment": "menu_planner"}
    )

    print("\n--- TEST PLANIFICATION MENU ---")
    menu = plan_weekly_menu("Pour 6 personnes. Repas Végétariens. Produits d'été uniquement.")
    print(menu)

# if __name__ == "__main__":
#     run_tests()
#     langfuse.flush()
#     print("\nTraces envoyées à Langfuse.")

# -----------------------------------------------------------------------
# --- PARTIE 3 : EVALUATION ET QUALITE ---

@observe(name="rule_evaluator")
def rule_evaluator(**kwargs) -> dict:
    """3.2 - Évaluateur Programmatique"""

    output = kwargs.get("output", "")
    expected = kwargs.get("expected_output", {})

    output_lower = output.lower()
    scores = {}
    
    # Interdits
    must_avoid = expected.get("must_avoid", [])
    found_forbidden = [i for i in must_avoid if i.lower() in output_lower]
    scores["no_forbidden"] = 1.0 if not found_forbidden else 0.0
    
    # Requis
    must_include = expected.get("must_include", [])
    found_included = [i for i in must_include if i.lower() in output_lower]
    scores["included_ratio"] = len(found_included) / len(must_include) if must_include else 1.0
    
    return scores

@observe(name="llm_judge")
def llm_judge(**kwargs) -> dict:
    """# 3.3 - LLM Juge"""

    question = kwargs.get("input", "")
    output = kwargs.get("output", "")
    expected = kwargs.get("expected_output", {})

    prompt = f"""Note ce menu (0.0 à 1.0) selon :
    1. Pertinence (respect de {question})
    2. Créativité
    3. Praticité
    Réponds uniquement en JSON : {{"pertinence": 0, "creativite": 0, "praticite": 0}}
    Menu : {output}"""
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content)

@observe(name="run_partie_3")
def run_evaluation():
    """3.4 - Lancer l'expérience"""

    # Tags spécifiques à la partie 3
    langfuse.update_current_trace(
        tags=["Partie 3", "Groupe_Natalène_Yacine"],
        metadata={"experiment": "menu_evaluation"}
    )
    
    my_dataset = langfuse.get_dataset("chefbot-menu-eval-Natalène_Yacine")
    print("\n--- EVALUATION ---")

    
    # Liste des items
    langfuse.run_experiment(
        name="Partie 3 Natalène_Yacine",
        data=my_dataset.items,
        task=lambda input: plan_weekly_menu(input["constraints"]),
        evaluators=[
            rule_evaluator,
            llm_judge
        ]
    )

# if __name__ == "__main__":
#     run_evaluation()
#     langfuse.flush()
#     print("\nTraces envoyées à Langfuse.")

"""
Ce code renvoie :
--- EVALUATION ---
Item 0 failed: run_evaluation.<locals>.<lambda>() got an unexpected keyword argument 'item'
Item 1 failed: run_evaluation.<locals>.<lambda>() got an unexpected keyword argument 'item'
Item 2 failed: run_evaluation.<locals>.<lambda>() got an unexpected keyword argument 'item'

Nous avons une erreur d'appel de la fonction lambda dans run_experiment : 
elle reçoit un argument 'item' alors que nous avons défini 'input'.

Nous avons cherché ensemble la source de l'erreur, sans résultats le jour J.
"""





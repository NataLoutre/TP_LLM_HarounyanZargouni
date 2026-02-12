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


def run_temperature_tests():
    question = "J'ai récupéré des poireaux et des noix du marché ce matin. Qu'est-ce que je peux cuisiner avec pour ce soir ?"
    temps = [0.1, 0.7, 1.2]

    for t in temps:
        print(f"\n--- Test avec la Température : {t} ---")
        reponse = ask_chef(question, temperature=t)
        print(reponse)



if __name__ == "__main__":
    run_temperature_tests()

    #  Envoi effectif des traces à Langfuse
    langfuse.flush()
    print("\nTraces envoyées à Langfuse.")


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
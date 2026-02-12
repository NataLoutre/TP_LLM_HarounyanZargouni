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
from dotenv import load_dotenv
from langfuse import observe, get_client, Evaluation
from groq import Groq
import json
from datetime import datetime
from typing import Callable
import os

load_dotenv()

groq_client = Groq()
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com/"


# =============================================================================
# CREATING DATASETS
# =============================================================================

def create_chefbot_menu_eval():
    dataset_name = "chefbot-menu-eval-Natalène_Yacine"

    dataset = get_client().create_dataset(
        name=dataset_name,
        description="dataset for ChefBot menu",
        metadata={
            "created_by": "lecture_demo",
            "domain": "Restaurant_menu",
            "version": "1.0"
        }
    )

    test_cases = [
        {
            "input": {"constraints": "Plat végétarien"},
            "expected_output": {"must_avoid": ["viande", "poisson", "fruits de mer"],
                                 "must_include": ["légumes", "protéines végétales"],
                                 "max_calories_per_meal": 700}

        },
        {
            "input": {"constraints": "plat pour allergique aux fruits de mer"},
            "expected_output": {"must_avoid":  ["crevettes", "crabe", "homard", "moules", "huîtres", "calamar"],
                                 "must_include": ["protéines non marines", "légumes"],
                                 "max_calories_per_meal": 600}
        },
        {
            "input": {"constraints": "plat pour intolérant au gluten"},
            "expected_output": {"must_avoid":  ["blé", "orge", "seigle", "épeautre", "pâtes classiques", "pain classique"],
                                 "must_include": ["féculents sans gluten", "légumes"],
                                 "max_calories_per_meal": 650}
        }
    ]

    for i, case in enumerate(test_cases):
        get_client().create_dataset_item(
            dataset_name=dataset_name,
            input=case["input"],
            expected_output=case["expected_output"],
        )

    print(f"✓ Created dataset with {len(test_cases)} test cases")
    return dataset


if __name__ == "__main__":
    create_chefbot_menu_eval()
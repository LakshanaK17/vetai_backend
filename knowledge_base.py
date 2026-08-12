"""VetAI knowledge base: BSAVA-derived rule triggers + breed-aware diet facts + retrieval.
Shared by the FastAPI app and (optionally) the LLM decision layer."""

# ---- lesion model class -> rule category ----
LESION_CATEGORY_MAP = {
    "dermatitis": "dermatitis", "hypersensitivity": "dermatitis",
    "fungal": "dermatology", "ringworm": "dermatology", "demodicosis": "dermatology",
    "otology": "otology", "otitis": "otology", "ear": "otology",
    "healthy": "healthy",
}

def to_category(label: str) -> str:
    k = label.strip().lower().replace("_", " ")
    for key, val in LESION_CATEGORY_MAP.items():
        if key in k:
            return val
    return "dermatology"

def norm(x: str) -> str:
    return x.strip().lower().replace("_", " ")

# ---- BSAVA / Merck rule knowledge base (treatment triggers) ----
RULE_KB = {
    ("labrador retriever", "otology"): {"agent": "Marbofloxacin otic", "source": "Merck Veterinary Manual",
        "trigger": "Breed = Labrador Retriever AND lesion = otology"},
    ("german shepherd", "dermatitis"): {"agent": "Cephalexin (systemic)", "source": "BSAVA Small Animal Formulary",
        "trigger": "Breed = German Shepherd AND lesion = dermatitis"},
    ("chihuahua", "dermatitis"): {"agent": "Chlorhexidine 2-4% + hydrocortisone", "source": "BSAVA Small Animal Formulary",
        "trigger": "Breed = Chihuahua AND lesion = dermatitis"},
    ("french bulldog", "otology"): {"agent": "Fluconazole + gentamicin", "source": "Merck Veterinary Manual",
        "trigger": "Breed = French Bulldog AND lesion = otology"},
    ("rottweiler", "dermatology"): {"agent": "Mupirocin ointment", "source": "BSAVA Small Animal Formulary",
        "trigger": "Breed = Rottweiler AND lesion = dermatology"},
    ("siberian husky", "otology"): {"agent": "Marbofloxacin otic (no flushing)", "source": "Merck Veterinary Manual",
        "trigger": "Breed = Siberian Husky AND lesion = otology"},
}
GENERIC_KB = {
    "dermatitis":  {"agent": "Chlorhexidine bathing; systemic cephalexin if pyoderma confirmed", "source": "BSAVA Small Animal Formulary"},
    "dermatology": {"agent": "Topical antimicrobial (mupirocin/chlorhexidine); skin scrape to exclude mites/fungus", "source": "BSAVA Small Animal Formulary"},
    "otology":     {"agent": "Ear cytology + gentle cleaning; empirical marbofloxacin otic", "source": "Merck Veterinary Manual"},
    "healthy":     {"agent": "No pharmacological treatment indicated", "source": "-"},
}

# ---- breed-aware structured diet knowledge base ----
DIET_KB = {
    "labrador retriever": {"profile": "Obesity- and hip-dysplasia-prone; calorie-controlled, joint-supporting.",
        "recommended": ["Lean chicken/white fish", "Brown rice or sweet potato", "Fish oil (omega-3)", "Glucosamine + chondroitin", "Green beans/carrots"],
        "quantity": "~2.5-3 cups/day of controlled-calorie food, split into 2 meals.",
        "avoid": ["High-fat table scraps", "Free-feeding / overfeeding", "Excess treats"]},
    "german shepherd": {"profile": "Sensitive digestion, allergy-prone skin; highly digestible, skin-supporting.",
        "recommended": ["Lamb or salmon (novel protein)", "Digestible carbs (rice, oats)", "Omega-3 fish oil", "Probiotics", "Beet pulp"],
        "quantity": "~3-3.5 cups/day of high-quality food, split into 2 meals.",
        "avoid": ["Beef, wheat, soy (common allergens)", "Cheap fillers/by-products", "Sudden diet changes"]},
    "golden retriever": {"profile": "Allergy- and pancreatitis-aware, joint-sensitive; antioxidant-rich.",
        "recommended": ["Salmon or turkey", "Sweet potato/pumpkin", "Balanced omega-3/6", "Blueberries/antioxidants", "Glucosamine"],
        "quantity": "~2.5-3 cups/day, split into 2 meals; watch weight closely.",
        "avoid": ["Fatty foods (pancreatitis risk)", "Artificial additives", "Overfeeding"]},
    "chihuahua": {"profile": "Toy breed; dental-friendly, energy-dense, small frequent meals.",
        "recommended": ["Small-kibble nutrient-dense food", "Chicken/turkey", "Omega-3 for skin/coat", "Dental-support kibble"],
        "quantity": "~1/2-1 cup/day split into 3-4 small meals.",
        "avoid": ["Large kibble", "Sugary treats", "Overfeeding (obesity risk)"]},
    "french bulldog": {"profile": "Brachycephalic, allergy/gas-prone; limited-ingredient, easily digestible.",
        "recommended": ["Novel single protein (duck/fish)", "Limited-ingredient formula", "Omega-3 for skin & ears", "Probiotics"],
        "quantity": "~1.5-2 cups/day split into 2 meals; slow-feeder bowl.",
        "avoid": ["Dairy", "Gas-producing foods (beans, cabbage)", "Grains if allergic"]},
    "rottweiler": {"profile": "Giant breed; controlled growth, joint support, obesity prevention.",
        "recommended": ["Large-breed formula", "Lean beef/chicken", "Controlled calcium & phosphorus", "Glucosamine + omega-3", "Fish oil"],
        "quantity": "~4-6 cups/day of large-breed food, split into 2 meals.",
        "avoid": ["Calcium over-supplementation (puppies)", "High-fat diets", "Rapid overfeeding"]},
    "siberian husky": {"profile": "High-energy working breed; protein/fat-rich, zinc-supported coat.",
        "recommended": ["High-quality fish/meat protein", "Healthy fats for energy", "Zinc supplement", "Omega-3 for coat"],
        "quantity": "~2-3 cups/day of energy-dense food (adjust to activity), split into 2 meals.",
        "avoid": ["Low-zinc diets", "Excess fillers/grains", "Overfeeding despite lean look"]},
}
GENERIC_DIET = {"profile": "Balanced maintenance diet for an adult dog.",
    "recommended": ["Quality animal protein", "Digestible carbohydrate", "Omega-3 fish oil", "Vegetables"],
    "quantity": "Feed per body-weight guidance on packaging, split into 2 meals.",
    "avoid": ["Chocolate, grapes, onions, xylitol", "Excess table scraps", "Overfeeding"]}
DIET_MOD = {
    "otology": "Reduce yeast-feeding sugars/carbs; add omega-3 to calm ear inflammation.",
    "dermatitis": "Boost omega-3/6; consider a limited-ingredient trial to rule out food allergy.",
    "dermatology": "Add omega-3, zinc and vitamin E to support skin-barrier repair.",
    "healthy": "Maintain current balanced diet.",
}

def retrieve(breed_label: str, lesion_label: str) -> dict:
    """Retrieve the grounding rule + structured diet for the predicted breed and lesion."""
    b, cat = norm(breed_label), to_category(lesion_label)
    rule = RULE_KB.get((b, cat))
    if rule:
        rule = {**rule, "matched": True}
    else:
        g = GENERIC_KB.get(cat, GENERIC_KB["dermatology"])
        rule = {"agent": g["agent"], "source": g["source"],
                "trigger": f"No exact breed rule for ({breed_label}, {cat}); generic BSAVA {cat} guidance",
                "matched": False}
    diet = DIET_KB.get(b, GENERIC_DIET)
    return {"breed": breed_label, "lesion": lesion_label, "category": cat,
            "rule": rule, "diet": diet, "diet_modifier": DIET_MOD.get(cat, "")}

"""
Algions AI Labs - E-Commerce Product Recommendation Engine
Data Generator: Synthesizes realistic, power-law distributed e-commerce user-item interactions
with rich catalog metadata, session structures, and chronological timestamps.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

RANDOM_SEED = 42

CATEGORIES_TAXONOMY = {
    "Electronics": {
        "subcategories": ["Audio & Headphones", "Smartphones & Tablets", "Laptops & Computing", "Smart Home & IoT", "Accessories"],
        "brands": ["Sony", "Apple", "Samsung", "Bose", "Anker", "Dell", "Logitech", "Sennheiser", "Google", "Asus"],
        "price_range": (25.0, 1499.0),
        "tags": ["wireless", "bluetooth", "noise-cancelling", "usb-c", "portable", "premium", "fast-charging", "4k", "rechargeable", "smart"]
    },
    "Fashion": {
        "subcategories": ["Men's Apparel", "Women's Apparel", "Footwear", "Watches & Jewelry", "Bags & Luggage"],
        "brands": ["Nike", "Adidas", "Levi's", "Zara", "Fossil", "Samsonite", "Puma", "Tommy Hilfiger", "Calvin Klein", "Under Armour"],
        "price_range": (19.0, 399.0),
        "tags": ["cotton", "breathable", "stylish", "casual", "athletic", "formal", "waterproof", "leather", "classic", "sustainable"]
    },
    "Home & Living": {
        "subcategories": ["Kitchen Appliances", "Cookware", "Bedding & Bath", "Furniture & Decor", "Smart Lighting"],
        "brands": ["Instant Pot", "Philips", "Dyson", "Cuisinart", "IKEA", "T-fal", "KitchenAid", "Nespresso", "Breville", "Shark"],
        "price_range": (15.0, 699.0),
        "tags": ["stainless-steel", "ergonomic", "energy-efficient", "compact", "non-stick", "dishwasher-safe", "modern", "durable", "cozy", "automated"]
    },
    "Fitness & Sports": {
        "subcategories": ["Workout Equipment", "Running & Trail", "Yoga & Recovery", "Sports Nutrition", "Outdoor & Camping"],
        "brands": ["Garmin", "Lululemon", "Hydro Flask", "Bowflex", "Optimum Nutrition", "Therabody", "Coleman", "Manduka", "Fitbit", "The North Face"],
        "price_range": (12.0, 549.0),
        "tags": ["lightweight", "moisture-wicking", "high-protein", "insulated", "gps-enabled", "heavy-duty", "flexible", "recovery", "cardio", "hydration"]
    },
    "Books & Media": {
        "subcategories": ["Technology & AI", "Business & Finance", "Self-Improvement", "Fiction & Sci-Fi", "Science & Nature"],
        "brands": ["O'Reilly Media", "Penguin Random House", "HarperCollins", "MIT Press", "Simon & Schuster", "Hachette", "Wiley", "Macmillan"],
        "price_range": (9.99, 79.99),
        "tags": ["bestseller", "hardcover", "paperback", "illustrated", "handbook", "guide", "inspirational", "award-winning", "machine-learning", "mindset"]
    },
    "Beauty & Personal Care": {
        "subcategories": ["Skincare", "Haircare", "Fragrance & Grooming", "Bath & Body", "Wellness & Supplements"],
        "brands": ["CeraVe", "The Ordinary", "La Roche-Posay", "Dior", "Olaplex", "Neutrogena", "Kiehl's", "Aesop", "Glossier", "Laneige"],
        "price_range": (8.0, 220.0),
        "tags": ["hydrating", "fragrance-free", "dermatologist-tested", "organic", "cruelty-free", "anti-aging", "vitamin-c", "sulfate-free", "nourishing", "vegan"]
    }
}

SAMPLE_TITLES = {
    "Electronics": [
        "WH-1000XM5 Wireless Noise Canceling Headphones", "AirPods Pro 2nd Gen with USB-C", "Galaxy S24 Ultra 256GB Titanium",
        "MacBook Air 13-inch M3 16GB RAM", "QuietComfort 45 Bluetooth Headphones", "Soundcore Motion+ Hi-Res Bluetooth Speaker",
        "UltraWide 34-inch Curved Gaming Monitor", "MX Master 3S Advanced Wireless Mouse", "Pixel 9 Pro Hazel 128GB Smartphone",
        "Echo Show 10 Smart Display with Motion", "PowerCore 26800mAh Portable Fast Charger", "Momentum 4 Wireless Audiophile Headphones",
        "ROG Zephyrus G14 Gaming Laptop RTX 4070", "Nest Learning Thermostat 4th Gen", "Ergonomic Mechanical Keyboard RGB"
    ],
    "Fashion": [
        "Air Zoom Pegasus 40 Running Shoes", "Ultraboost Light Cushioned Athletic Shoes", "511 Slim Fit Stretch Denim Jeans",
        "Classic Minimalist Chronograph Watch", "Tech Fleece Full-Zip Windrunner Hoodie", "Free Rein Waterproof Hiking Boots",
        "Classic Leather Continental Wallet", "Omni-Shield Lightweight Packable Jacket", "All-Day Comfort Seamless Crew Tee 3-Pack",
        "Pro Training Duffel Bag 60L", "Heritage Stainless Steel Quartz Watch", "Everyday Canvas Messenger Bag Laptop Sleeve"
    ],
    "Home & Living": [
        "Duo Plus 9-in-1 Electric Pressure Cooker 6Qt", "Airfryer XXL with Digital Smart Sensing", "V15 Detect Cordless Vacuum Cleaner",
        "Custom Clad Tri-Ply Stainless Steel Cookware Set", "100% Organic Bamboo Deep Pocket Sheet Set", "VertuoPlus Coffee and Espresso Machine",
        "Ergonomic High-Back Office Chair with Lumbar Support", "Compact Countertop Ice Maker 26lbs/Day", "Smart WiFi Aroma Diffuser with Timer",
        "Cast Iron Enameled Dutch Oven 6-Quart", "Precision Chef Knife 8-inch German Steel", "True HEPA Air Purifier for Large Rooms"
    ],
    "Fitness & Sports": [
        "Forerunner 265 Running Smartwatch OLED", "Align High-Rise Yoga Pant 28-inch Weightless", "Wide Mouth Vacuum Insulated Stainless Steel Bottle 32oz",
        "SelectTech 552 Adjustable Dumbbells Pair", "Gold Standard 100% Whey Protein Powder 5lb", "Theragun Prime Deep Tissue Percussion Massager",
        "PRO Yoga Mat 6mm Non-Slip High Density", "Sundome 4-Person Camping Tent Weatherproof", "Hydro Flask Standard Mouth Flex Cap 24oz",
        "Speed Rope Pro Ball-Bearing Jump Rope", "Resistance Bands Set with Door Anchor and Handles", "Foam Roller High-Density Deep Tissue Massager"
    ],
    "Books & Media": [
        "Designing Machine Learning Systems", "Designing Data-Intensive Applications", "Atomic Habits: Tiny Changes, Remarkable Results",
        "Deep Learning with Python 2nd Edition", "The Psychology of Money: Timeless Lessons", "Hands-On Machine Learning with Scikit-Learn & PyTorch",
        "Thinking, Fast and Slow", "Clean Code: A Handbook of Agile Software Craftsmanship", "Zero to One: Notes on Startups",
        "System Design Interview: An Insider's Guide", "Storytelling with Data: A Data Visualization Guide", "Algorithms to Live By: Computer Science of Decisions"
    ],
    "Beauty & Personal Care": [
        "Hydrating Facial Cleanser for Normal to Dry Skin", "Niacinamide 10% + Zinc 1% Blemish Formula", "Anthelios Melt-in Milk Sunscreen SPF 60",
        "No. 7 Bonding Hair Oil Heat Protectant", "Hydro Boost Hyaluronic Acid Water Gel", "Lip Sleeping Mask Intense Moisture Berry",
        "Ultra Facial Cream Daily Squalane Moisturizer", "Resurrection Aromatique Hand Balm 75ml", "Vitamin C Suspension 30% in Silicone",
        "Milky Jelly Gentle Daily Cleanser Conditioning", "Purifying Clay Detox Face Mask", "Botanical Body Wash Geranium Leaf 500ml"
    ]
}


def generate_catalog(num_items: int = 600, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate rich, realistic e-commerce product catalog metadata."""
    random.seed(random_state)
    np.random.seed(random_state)

    categories = list(CATEGORIES_TAXONOMY.keys())
    items = []

    for i in range(1, num_items + 1):
        item_id = f"PROD_{i:04d}"
        category = categories[(i - 1) % len(categories)]
        cat_info = CATEGORIES_TAXONOMY[category]
        sub_category = random.choice(cat_info["subcategories"])
        brand = random.choice(cat_info["brands"])
        
        # Pick base title template or synthesize title
        titles_pool = SAMPLE_TITLES[category]
        base_title = titles_pool[(i - 1) % len(titles_pool)]
        model_suffix = f"Series {random.randint(2, 9)}" if random.random() > 0.4 else f"Gen {random.randint(1, 4)}"
        variant = random.choice(["Black", "Silver", "Navy", "Standard", "Pro Edition", "Edition Plus", "Midnight", "Matte"])
        title = f"{brand} {base_title} - {variant} ({model_suffix})"

        min_price, max_price = cat_info["price_range"]
        # Log-normal price distribution within category bounds
        price = round(float(np.clip(np.exp(np.random.normal(np.log(min_price * 1.5), 0.7)), min_price, max_price)), 2)

        # Select 3-5 tags
        item_tags = random.sample(cat_info["tags"], k=min(len(cat_info["tags"]), random.randint(3, 5)))
        tags_str = ", ".join(item_tags)

        # Description
        description = (
            f"High-quality {sub_category.lower()} by {brand}. Engineered with advanced {item_tags[0]} features "
            f"and crafted with durable, premium materials. Includes {item_tags[1]} design optimized for everyday performance "
            f"and reliable convenience. Highly recommended in {category}."
        )

        rating = round(float(np.clip(np.random.normal(4.35, 0.45), 2.5, 5.0)), 1)
        review_count = int(np.random.exponential(scale=180) + 5)

        items.append({
            "item_id": item_id,
            "title": title,
            "category": category,
            "sub_category": sub_category,
            "brand": brand,
            "price": price,
            "rating": rating,
            "review_count": review_count,
            "tags": tags_str,
            "description": description
        })

    return pd.DataFrame(items)


def generate_users(num_users: int = 2500, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate realistic user profiles with category affinity and demographic tiers."""
    random.seed(random_state)
    np.random.seed(random_state)

    categories = list(CATEGORIES_TAXONOMY.keys())
    users = []

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 8, 1)
    date_range_days = (end_date - start_date).days

    for u in range(1, num_users + 1):
        user_id = f"USER_{u:04d}"
        age_group = random.choices(["18-24", "25-34", "35-44", "45-54", "55+"], weights=[0.20, 0.38, 0.22, 0.12, 0.08])[0]
        gender = random.choices(["M", "F", "Other"], weights=[0.48, 0.48, 0.04])[0]
        primary_cat = random.choice(categories)
        secondary_cats = [c for c in categories if c != primary_cat]
        secondary_cat = random.choice(secondary_cats)
        price_tier = random.choices(["budget", "mid-range", "premium"], weights=[0.30, 0.50, 0.20])[0]

        signup_offset = random.randint(0, date_range_days)
        signup_date = (start_date + timedelta(days=signup_offset)).strftime("%Y-%m-%d")

        # Inherent user activity tier (power shopper vs casual)
        activity_tier = random.choices(["power", "moderate", "casual", "cold_start"], weights=[0.12, 0.48, 0.30, 0.10])[0]

        users.append({
            "user_id": user_id,
            "age_group": age_group,
            "gender": gender,
            "primary_category": primary_cat,
            "secondary_category": secondary_cat,
            "price_tier": price_tier,
            "activity_tier": activity_tier,
            "signup_date": signup_date
        })

    return pd.DataFrame(users)


def generate_interactions(
    users_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    start_time: datetime = datetime(2026, 8, 1, 0, 0, 0),
    duration_days: int = 90,
    random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate power-law distributed user-item implicit interactions across sessions.
    Simulates views, carts, and purchases with temporal order and realistic conversion funnels.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    # 1. Compute item popularity distribution via Zipf/Power-law
    num_items = len(catalog_df)
    item_ids = catalog_df["item_id"].tolist()
    
    # Power-law weights (alpha = 1.25)
    ranks = np.arange(1, num_items + 1)
    base_item_probs = 1.0 / (ranks ** 1.25)
    base_item_probs /= base_item_probs.sum()

    # Map item_id to its base probability and metadata
    item_prob_map = {item_ids[i]: base_item_probs[i] for i in range(num_items)}
    item_cat_map = catalog_df.set_index("item_id")["category"].to_dict()
    category_items_map = catalog_df.groupby("category")["item_id"].apply(list).to_dict()

    interactions = []
    session_counter = 1

    # Mapping activity tier to expected number of sessions
    tier_session_counts = {
        "power": (14, 28),
        "moderate": (5, 12),
        "casual": (2, 5),
        "cold_start": (0, 1)  # Pure cold start users have 0 sessions in train
    }

    # Conversion action weights
    # View (70%), Cart (20%), Purchase (10%)
    action_types = ["view", "cart", "purchase"]
    action_weights = [0.70, 0.20, 0.10]
    implicit_weight_map = {"view": 1.0, "cart": 2.5, "purchase": 5.0}

    for _, user in users_df.iterrows():
        user_id = user["user_id"]
        tier = user["activity_tier"]
        min_sess, max_sess = tier_session_counts[tier]
        num_sessions = random.randint(min_sess, max_sess)

        if num_sessions == 0:
            continue

        p_cat = user["primary_category"]
        s_cat = user["secondary_category"]

        # Spread sessions across duration_days
        session_day_offsets = sorted(np.random.uniform(0, duration_days, size=num_sessions))

        for offset in session_day_offsets:
            sess_id = f"SESS_{session_counter:07d}"
            session_counter += 1
            session_start = start_time + timedelta(days=float(offset), seconds=random.randint(0, 3600 * 12))

            # Items per session (1 to 5 items browsed)
            num_session_items = random.choices([1, 2, 3, 4, 6], weights=[0.25, 0.35, 0.22, 0.12, 0.06])[0]

            # In this session, decide focal category
            roll = random.random()
            if roll < 0.65:
                session_category = p_cat
            elif roll < 0.88:
                session_category = s_cat
            else:
                session_category = random.choice(list(CATEGORIES_TAXONOMY.keys()))

            # Filter items in session category and sample based on popularity
            candidate_items = category_items_map[session_category]
            candidate_probs = np.array([item_prob_map[iid] for iid in candidate_items])
            candidate_probs /= candidate_probs.sum()

            sampled_items = np.random.choice(
                candidate_items,
                size=min(len(candidate_items), num_session_items),
                replace=False,
                p=candidate_probs
            )

            current_timestamp = session_start
            for item_id in sampled_items:
                # Progression funnel: All items in session are viewed
                # Some are carted, some purchased
                action_choice = random.choices(action_types, weights=action_weights)[0]
                
                # If purchased, create preceding view and cart if natural
                event_seq = ["view"]
                if action_choice == "cart":
                    event_seq.append("cart")
                elif action_choice == "purchase":
                    event_seq.extend(["cart", "purchase"])

                for act in event_seq:
                    current_timestamp += timedelta(seconds=random.randint(15, 180))
                    interactions.append({
                        "session_id": sess_id,
                        "user_id": user_id,
                        "item_id": item_id,
                        "event_type": act,
                        "event_weight": implicit_weight_map[act],
                        "timestamp": current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    })

    df_interactions = pd.DataFrame(interactions)
    df_interactions["timestamp"] = pd.to_datetime(df_interactions["timestamp"])
    df_interactions = df_interactions.sort_values(by="timestamp").reset_index(drop=True)
    return df_interactions


def run_data_generation(output_dir: str = "data/raw", random_state: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete dataset generation pipeline and save to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic e-commerce catalog (random_seed={random_state})...")
    catalog_df = generate_catalog(num_items=600, random_state=random_state)
    catalog_path = os.path.join(output_dir, "products.csv")
    catalog_df.to_csv(catalog_path, index=False)
    print(f"Saved {len(catalog_df)} products to {catalog_path}")

    print("Generating user cohort...")
    users_df = generate_users(num_users=2500, random_state=random_state)
    users_path = os.path.join(output_dir, "users.csv")
    users_df.to_csv(users_path, index=False)
    print(f"Saved {len(users_df)} users to {users_path}")

    print("Generating user-item interaction streams with power-law dynamics...")
    interactions_df = generate_interactions(users_df, catalog_df, random_state=random_state)
    interactions_path = os.path.join(output_dir, "interactions.csv")
    interactions_df.to_csv(interactions_path, index=False)
    print(f"Saved {len(interactions_df)} interactions to {interactions_path}")

    return catalog_df, users_df, interactions_df


if __name__ == "__main__":
    run_data_generation()


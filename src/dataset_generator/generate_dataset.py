import random
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

NUM_RECORDS = 50000

# Project root:
# semantic_search_project/
# ├── data/
# └── src/
#     └── dataset_generator/
#         └── generate_dataset.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = PROJECT_ROOT / "data" / "NLP_50K_Upgraded.xlsx"


# ============================================================
# TOPICS
# ============================================================

topics = {
    "Python": [
        "Python variables and data types",
        "Python functions and parameters",
        "Python lists and dictionaries",
        "Python exception handling",
        "Python object oriented programming",
        "Python file handling",
        "Python modules and packages",
        "Python virtual environments",
        "Python decorators",
        "Python generators"
    ],

    "SQL": [
        "SQL SELECT queries",
        "SQL JOIN operations",
        "SQL GROUP BY and aggregation",
        "SQL subqueries",
        "SQL indexes",
        "SQL constraints",
        "SQL transactions",
        "SQL views",
        "SQL stored procedures",
        "SQL database normalization"
    ],

    "NLP": [
        "natural language processing",
        "text preprocessing",
        "tokenization",
        "stemming and lemmatization",
        "named entity recognition",
        "sentiment analysis",
        "text classification",
        "language models",
        "word embeddings",
        "semantic similarity"
    ],

    "Machine Learning": [
        "supervised learning",
        "unsupervised learning",
        "linear regression",
        "logistic regression",
        "decision trees",
        "random forests",
        "support vector machines",
        "K means clustering",
        "model evaluation",
        "feature engineering"
    ],

    "Deep Learning": [
        "neural networks",
        "convolutional neural networks",
        "recurrent neural networks",
        "LSTM networks",
        "transformers",
        "backpropagation",
        "activation functions",
        "loss functions",
        "optimization algorithms",
        "deep learning training"
    ],

    "Generative AI": [
        "large language models",
        "prompt engineering",
        "AI assistants",
        "text generation",
        "foundation models",
        "generative AI applications",
        "LLM inference",
        "AI agents",
        "model evaluation",
        "responsible AI"
    ],

    "RAG": [
        "retrieval augmented generation",
        "document retrieval",
        "RAG pipelines",
        "query processing",
        "context retrieval",
        "hybrid search",
        "RAG evaluation",
        "document chunking",
        "retrieval ranking",
        "question answering"
    ],

    "Vector Database": [
        "vector embeddings",
        "vector similarity search",
        "FAISS",
        "vector indexes",
        "embedding storage",
        "nearest neighbor search",
        "semantic search",
        "vector database architecture",
        "metadata filtering",
        "embedding retrieval"
    ],

    "Data Science": [
        "data cleaning",
        "exploratory data analysis",
        "data visualization",
        "feature selection",
        "statistical analysis",
        "data preprocessing",
        "data quality",
        "predictive analytics",
        "data science workflow",
        "data preparation"
    ],

    "Web Development": [
        "HTML fundamentals",
        "CSS styling",
        "JavaScript programming",
        "responsive web design",
        "frontend development",
        "backend development",
        "REST APIs",
        "web authentication",
        "web application architecture",
        "browser development"
    ],

    "Software Engineering": [
        "software development lifecycle",
        "object oriented design",
        "version control",
        "Git workflows",
        "software testing",
        "unit testing",
        "debugging",
        "clean code",
        "design patterns",
        "software architecture"
    ],

    "Cloud Computing": [
        "cloud computing fundamentals",
        "cloud storage",
        "virtual machines",
        "cloud databases",
        "serverless computing",
        "cloud security",
        "cloud networking",
        "container services",
        "cloud monitoring",
        "cloud architecture"
    ],

    "Cybersecurity": [
        "network security",
        "authentication",
        "authorization",
        "encryption",
        "secure coding",
        "security monitoring",
        "identity management",
        "web security",
        "data protection",
        "cybersecurity fundamentals"
    ],

    "DevOps": [
        "continuous integration",
        "continuous deployment",
        "Docker containers",
        "Kubernetes",
        "CI CD pipelines",
        "infrastructure automation",
        "DevOps monitoring",
        "deployment strategies",
        "source control",
        "DevOps practices"
    ],

    "Database": [
        "relational databases",
        "database design",
        "database normalization",
        "database indexing",
        "database transactions",
        "database security",
        "NoSQL databases",
        "database performance",
        "data integrity",
        "database administration"
    ],

    "APIs": [
        "REST API design",
        "API authentication",
        "HTTP methods",
        "API requests",
        "API responses",
        "JSON APIs",
        "API error handling",
        "API versioning",
        "API security",
        "API integration"
    ],

    "Computer Vision": [
        "image classification",
        "object detection",
        "image preprocessing",
        "OpenCV",
        "image segmentation",
        "face detection",
        "computer vision models",
        "image recognition",
        "visual feature extraction",
        "computer vision applications"
    ],

    "Artificial Intelligence": [
        "artificial intelligence fundamentals",
        "AI problem solving",
        "AI agents",
        "knowledge representation",
        "machine reasoning",
        "AI planning",
        "intelligent systems",
        "AI applications",
        "AI model evaluation",
        "responsible artificial intelligence"
    ],

    "Programming Concepts": [
        "algorithms",
        "data structures",
        "arrays",
        "linked lists",
        "stacks and queues",
        "hash tables",
        "recursion",
        "sorting algorithms",
        "search algorithms",
        "algorithm complexity"
    ]
}


# ============================================================
# CONTENT TEMPLATES
# ============================================================

templates = [
    "{topic} is an important concept in {category}. It is commonly used when building practical software systems.",

    "Understanding {topic} helps developers design efficient and maintainable applications. The concept is widely used in real-world projects.",

    "In {category}, {topic} is useful for solving technical problems and improving application reliability.",

    "A developer working with {category} should understand {topic} because it can affect application performance, scalability, and maintainability.",

    "{topic} is commonly studied as part of {category}. Practical implementation requires understanding the underlying concepts and appropriate use cases.",

    "Modern applications often use {topic} as part of a broader {category} workflow. Developers should understand both its advantages and limitations.",

    "When implementing {topic}, developers should consider performance, security, maintainability, and the requirements of the application.",

    "{topic} can be combined with other {category} techniques to build more reliable and scalable systems.",

    "Learning {topic} provides practical knowledge that can be applied to projects involving {category}.",

    "The implementation of {topic} may vary depending on the application requirements, available resources, and expected system behavior."
]


# ============================================================
# PREPARE TOPIC LIST
# ============================================================

topic_items = []

for category, items in topics.items():
    for item in items:
        topic_items.append((category, item))


# ============================================================
# GENERATE DATA
# ============================================================

print("=" * 60)
print("STARTING 50K DATASET GENERATION")
print("=" * 60)

records = []

for i in range(1, NUM_RECORDS + 1):

    # Select category and topic
    category, topic = random.choice(topic_items)

    # Create title
    title = topic.title()

    # Select different sentences
    selected_templates = random.sample(
        templates,
        k=random.randint(3, 6)
    )

    content_parts = []

    for template in selected_templates:

        sentence = template.format(
            topic=topic,
            category=category
        )

        content_parts.append(sentence)

    # Combine sentences
    content = " ".join(content_parts)

    # Create keywords
    keywords = [
        word.lower()
        for word in topic.replace("-", " ").split()
        if len(word) > 2
    ]

    keyword = ", ".join(keywords)

    # Create document
    record = {
        "document_id": f"DOC_{i:05d}",
        "category": category,
        "title": title,
        "content": content,
        "keyword": keyword
    }

    records.append(record)

    # Progress
    if i % 5000 == 0:
        print(f"Generated {i:,} records...")


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# CREATE DATA DIRECTORY IF NEEDED
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_excel(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("50K DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Records       : {len(df):,}")
print(f"Columns       : {list(df.columns)}")
print(f"Categories    : {df['category'].nunique()}")
print(f"Unique titles : {df['title'].nunique()}")
print(f"Output file   : {OUTPUT_FILE}")

print()
print("Category distribution:")
print(df["category"].value_counts())

print("=" * 60)
print("STAGE 1 DATASET GENERATION COMPLETE")
print("=" * 60)
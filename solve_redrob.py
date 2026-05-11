import argparse
import math
from collections import Counter


SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",
    # Web — Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",
    # Web — Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android",
    "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}


RESUMES = [
    ("Arjun Sharma", "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair", "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta", "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel", "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh", "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta", "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao", "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar", "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer", "python, R, statistics, ML, regression, clustering, Power-BI"),
]


JDS = [
    (
        "JD-1",
        "Kakao (ML Engineer)",
        "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
        "NLP, BERT, Feature Engineering, Statistics",
    ),
    (
        "JD-2",
        "Naver (Backend Engineer)",
        "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
        "REST API, CI/CD, Redis",
    ),
    (
        "JD-3",
        "Line (Frontend Engineer)",
        "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
        "Node.js, GraphQL, Redux, Jest, AWS",
    ),
]


MULTI_WORD_ALIASES = sorted(
    [k for k in SKILL_ALIASES if " " in k], key=len, reverse=True
)


def normalize_token(raw_token):
    token = raw_token.strip().lower()
    if not token:
        return None

    for phrase in MULTI_WORD_ALIASES:
        if token == phrase:
            return SKILL_ALIASES[phrase]

    return SKILL_ALIASES.get(token)


def normalize_skills(raw_skills):
    canonical = []
    for raw_token in raw_skills.split(","):
        mapped = normalize_token(raw_token)
        if mapped is not None:
            canonical.append(mapped)

    deduped = []
    seen = set()
    for skill in canonical:
        if skill not in seen:
            deduped.append(skill)
            seen.add(skill)
    return deduped


def build_vocabulary(normalized_resumes):
    vocab_set = set()
    for _, skills in normalized_resumes:
        vocab_set.update(skills)
    return sorted(vocab_set)


def compute_idf(normalized_resumes, vocabulary):
    num_docs = len(normalized_resumes)
    df = Counter()
    for _, skills in normalized_resumes:
        for skill in set(skills):
            df[skill] += 1
    return {skill: math.log(num_docs / df[skill]) for skill in vocabulary}


def compute_resume_tfidf_vectors(normalized_resumes, vocabulary, idf):
    vectors = {}
    for name, skills in normalized_resumes:
        n = len(skills)
        tf = 1 / n if n else 0.0
        skill_set = set(skills)
        vector = [tf * idf[skill] if skill in skill_set else 0.0 for skill in vocabulary]
        vectors[name] = vector
    return vectors


def build_jd_binary_vector(jd_skills, vocabulary):
    jd_set = set(jd_skills)
    return [1.0 if skill in jd_set else 0.0 for skill in vocabulary]


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_top3(jd_vector, resume_vectors):
    scored = []
    for name, vec in resume_vectors.items():
        score = cosine_similarity(vec, jd_vector)
        scored.append((name, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:3]


def verify_pipeline(normalized_resumes, vocabulary, idf, resume_vectors):
    assert len(normalized_resumes) == 10, "Expected exactly 10 resumes."
    assert vocabulary == sorted(vocabulary), "Vocabulary must be alphabetically sorted."

    for name, skills in normalized_resumes:
        assert len(skills) == len(set(skills)), f"Duplicate skills found after dedupe: {name}"
        assert skills, f"Resume has zero recognized skills after normalization: {name}"

    for skill, idf_value in idf.items():
        assert idf_value >= 0.0, f"Unexpected negative IDF for skill: {skill}"

    dim = len(vocabulary)
    for name, vec in resume_vectors.items():
        assert len(vec) == dim, f"Vector length mismatch for resume: {name}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redrob hackathon resume matching engine (TF-IDF + cosine)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print normalized resumes and vocabulary before final ranking output.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run pipeline validation assertions before printing final output.",
    )
    args, _ = parser.parse_known_args()  # ignore Jupyter's kernel args
    return args


def main():
    args = parse_args()
    normalized_resumes = [(name, normalize_skills(raw)) for name, raw in RESUMES]
    vocabulary = build_vocabulary(normalized_resumes)
    idf = compute_idf(normalized_resumes, vocabulary)
    resume_vectors = compute_resume_tfidf_vectors(normalized_resumes, vocabulary, idf)

    if args.verify:
        verify_pipeline(normalized_resumes, vocabulary, idf, resume_vectors)

    if args.verbose:
        print("=== Normalized + Deduplicated Resume Skills ===")
        for name, skills in normalized_resumes:
            print(f"{name}: {', '.join(skills)}")
        print("\n=== Shared Vocabulary ===")
        print(", ".join(vocabulary))
        print()

    for jd_id, jd_label, required, preferred in JDS:
        jd_skills = normalize_skills(f"{required}, {preferred}")
        jd_vector = build_jd_binary_vector(jd_skills, vocabulary)
        top3 = rank_top3(jd_vector, resume_vectors)
        formatted = ", ".join(f"{name}({score:.2f})" for name, score in top3)
        print(f"{jd_id} — {jd_label}")
        print(formatted)


if __name__ == "__main__":
    main()

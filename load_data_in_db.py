import os
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import CONN_CRED

if __name__ == "__main__":
    conn = psycopg2.connect(**CONN_CRED)

    input = [os.path.join("input", f) for f in os.listdir("input") if os.path.isfile(os.path.join("input", f)) and f[-3:] == 'txt']

    with conn.cursor() as cursor:

        cursor.execute("CREATE EXTENSION vector;")
        cursor.execute("CREATE TABLE IF NOT EXISTS docs(id serial PRIMARY KEY, path text);")
        cursor.execute("""CREATE TABLE IF NOT EXISTS paragraphs(
                            id bigserial PRIMARY KEY, 
                            paragraph text, 
                            embedding vector(384), 
                            doc serial REFERENCES docs(id));""")
        cursor.execute("CREATE INDEX ON paragraphs USING hnsw (embedding vector_cosine_ops);")

        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

        for id, file in enumerate(input[:1]):
            cursor.execute("INSERT INTO docs (id, path) VALUES (%s, %s)", (id, file))
            with open(file, "r", encoding = "utf8") as f:
                pars = f.read().split("\n\n")
            print(f"Inserting paragraphs from {file}")
            for par in tqdm(pars):
                if par:
                    cursor.execute(F"INSERT INTO paragraphs (paragraph, embedding, doc) VALUES (%s, %s, %s);", (par, model.encode(par).reshape(-1,).tolist(), id))
    conn.commit()
    conn.close()
    

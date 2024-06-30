import psycopg2
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer

from config import CONN_CRED

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-s", "--search", help = "search query", type = str, default="huge creature")

    args = parser.parse_args()

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

    conn = psycopg2.connect(**CONN_CRED)

    with conn.cursor() as cursor:
        query = model.encode(args.search).reshape(-1,).tolist()
        #cursor.execute("SELECT p.paragraph, d.path FROM paragraphs AS p LEFT JOIN docs AS d ON p.doc = d.id ORDER BY p.embedding <=> %s::vector(384) LIMIT 10;", (query,))
        cursor.execute("SELECT p.paragraph FROM paragraphs AS p ORDER BY p.embedding <=> %s::vector(384) LIMIT 10;", (query,))
        results = cursor.fetchall()
    for text in results:
        print(f"Text: {text}")
        #print(f"Path: {path}")
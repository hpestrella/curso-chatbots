# -------------------------------------------------------------
# knowledge_base_aws.py
#
# Este script muestra cómo conectarse a un Knowledge Base (base de conocimiento)
# en AWS Bedrock, recuperar información relevante usando una consulta, y procesar
# los resultados. Está diseñado para ser claro y didáctico para estudiantes que
# están comenzando a trabajar con APIs, AWS y recuperación de información.
# -------------------------------------------------------------

import os  # Para interactuar con el sistema operativo y variables de entorno
import pprint  # Para imprimir resultados de forma legible

# -----------------------------
# Importaciones de terceros
# -----------------------------
import boto3  # SDK de AWS para Python, permite interactuar con servicios de AWS

# -----------------------------
# Importaciones estándar
# -----------------------------
from dotenv import load_dotenv  # Para cargar variables de entorno desde un archivo .env

# -----------------------------
# Cargar variables de entorno
# -----------------------------
# Esto busca un archivo .env en el directorio actual y carga las variables definidas allí.
# Es útil para mantener tus credenciales y configuraciones fuera del código fuente.
load_dotenv()

# -----------------------------
# Obtener credenciales de AWS
# -----------------------------
# Las credenciales se almacenan en variables de entorno para mayor seguridad.
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# -----------------------------
# Definir parámetros de búsqueda
# -----------------------------
# knowledge_id: ID de la base de conocimiento en AWS Bedrock
# retrieval_setting: Configuración de la búsqueda (top_k: cuántos resultados, score_threshold: umbral de relevancia)
# query: La pregunta o consulta que queremos responder usando la base de conocimiento
knowledge_id = "EIB66RF7WA"
retrieval_setting = {"top_k": 2, "score_threshold": 0.0}
query = "Quién es Alicia?"

# -----------------------------
# Crear cliente de AWS Bedrock
# -----------------------------
# Aquí se configura el cliente para conectarse al servicio 'bedrock-agent-runtime' en la región deseada.
client = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="eu-west-3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# --------------------------------------------------------------------------------
# Recuperar información de la base de conocimiento
# --------------------------------------------------------------------------------
# Se realiza una consulta a la base de conocimiento usando el método 'retrieve'.
# Se especifica el tipo de búsqueda (en este caso, 'HYBRID' para combinar diferentes métodos de búsqueda).
response = client.retrieve(
    knowledgeBaseId=knowledge_id,
    retrievalConfiguration={
        "vectorSearchConfiguration": {
            "numberOfResults": retrieval_setting.get("top_k"),
            "overrideSearchType": "HYBRID",
        }
    },
    retrievalQuery={"text": query},
)

# --------------------------------------------------------------------------------
# Procesar la respuesta
# --------------------------------------------------------------------------------
# Se extraen los resultados relevantes de la respuesta, filtrando por el umbral de score.
results = []
if (
    response.get("ResponseMetadata")
    and response.get("ResponseMetadata").get("HTTPStatusCode") == 200
):
    if response.get("retrievalResults"):
        retrieval_results = response.get("retrievalResults")
        for retrieval_result in retrieval_results:
            # Filtrar resultados con score menor al umbral
            if retrieval_result.get("score") < retrieval_setting.get("score_threshold", 0.0):
                continue
            result = {
                "metadata": retrieval_result.get("metadata"),
                "score": retrieval_result.get("score"),
                "title": retrieval_result.get("metadata").get("x-amz-bedrock-kb-source-uri"),
                "content": retrieval_result.get("content").get("text"),
            }
            results.append(result)

# --------------------------------------------------------------------------------
# Ahora que tienes el conocimiento recuperado y listo en la variable 'results',
# puedes enviarlo como contexto a tu modelo de lenguaje favorito (por ejemplo,
# OpenAI GPT, AWS Bedrock LLM, etc.) usando la API correspondiente para generar
# respuestas más informadas y personalizadas.
# --------------------------------------------------------------------------------

# -----------------------------
# Mostrar resultados
# -----------------------------
# Se imprime la lista de resultados de forma legible para el usuario.
pprint.pprint(results)

# El resultado es el siguiente, y viene de la base de conocimiento que construimos durante el curso:
"""
>>> pprint.pprint(results)
[{'content': '¡Cuidado con esta\r'
             ' teja suelta!... ¡Eh, que se cae! ¡Cuidado con la cabeza!»\r'
             ' Aquí se oyó una fuerte caída. «Vaya, ¿quién ha si-\r'
             ' \r'
             ' \r'
             ' 36\r'
             ' \r'
             ' \r'
             ' \r'
             ' do?... Creo que ha sido Bill... ¿Quién va a bajar por la\r'
             ' chimenea?... ¿Yo? Nanay. ¡Baja tú!... ¡Ni hablar! Tiene\r'
             ' que bajar Bill... ¡Ven aquí, Bill! ¡El amo dice que tienes\r'
             ' que bajar por la chimenea!»\r'
             ' \r'
             ' —¡Vaya! ¿Conque es Bill el que tiene que bajar por\r'
             ' la chimenea? se dijo Alicia—. ¡Parece que todo se lo\r'
             ' cargan a Bill! No me gustaría estar en su pellejo; desde luego '
             'esta chimenea es estrecha, pero me parece\r'
             ' que podré dar algún puntapié por ella.\r'
             ' \r'
             ' Alicia hundió el pie todo lo que pudo dentro de la\r'
             ' chimenea, y esperó hasta oír que la bestezuela (no\r'
             ' podía saber de qué tipo de animal se trataba) escarbaba y '
             'arañaba dentro de la chimenea, justo encima\r'
             ' de ella.',
  'metadata': {'x-amz-bedrock-kb-chunk-id': '1%3A0%3AUajG45YB1R0bQCa3F_fv',
               'x-amz-bedrock-kb-data-source-id': '95R3MMFNLK',
               'x-amz-bedrock-kb-source-uri': 's3://curso-gpt/Alicia en el '
                                              'país de las maravillas.txt'},
  'score': 0.489817,
  'title': 's3://curso-gpt/Alicia en el país de las maravillas.txt'},
 {'content': 'En el momento\r'
             ' en que apareció Alicia, todos se abalanzaron sobre\r'
             ' ella. Pero Alicia echó a correr con todas sus fuerzas,\r'
             ' y pronto se encontró a salvo en un espeso bosque.\r'
             ' \r'
             ' —Lo primero que ahora tengo que hacer —se dijo\r'
             ' Alicia, mientras vagaba por el bosque —es crecer hasta volver a '
             'recuperar mi estatura. Y lo segundo es\r'
             ' encontrar la manera de entrar en aquel precioso jardín. Me '
             'parece que éste es el mejor plan de acción.\r'
             ' \r'
             ' Parecía, desde luego, un plan excelente, y expuesto de un modo '
             'muy claro y muy simple. La única dificultad radicaba en que no '
             'tenía la menor idea de\r'
             ' cómo llevarlo a cabo. Y, mientras miraba ansiosamente por entre '
             'los árboles, un pequeño ladrido que\r'
             ' sonó justo encima de su cabeza la hizo mirar hacia\r'
             ' arriba sobresaltada.\r'
             ' \r'
             ' Un enorme perrito la miraba desde arriba con sus grandes ojos '
             'muy abiertos y alargaba tímidamente una patita\r'
             ' para tocarla.\r'
             ' \r'
             ' —¡Qué cosa tan bonita!',
  'metadata': {'x-amz-bedrock-kb-chunk-id': '1%3A0%3AVqjG45YB1R0bQCa3F_fv',
               'x-amz-bedrock-kb-data-source-id': '95R3MMFNLK',
               'x-amz-bedrock-kb-source-uri': 's3://curso-gpt/Alicia en el '
                                              'país de las maravillas.txt'},
  'score': 0.489817,
  'title': 's3://curso-gpt/Alicia en el país de las maravillas.txt'}]
"""

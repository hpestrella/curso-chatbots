import boto3

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime", region_name="eu-west-3"  # e.g., us-east-1, us-west-2, etc.
)

response = bedrock_runtime.retrieve_and_generate(
    knowledgeBaseId="EIB66RF7WA", input={"text": "Quién es Alicia?"}
)

# TODO

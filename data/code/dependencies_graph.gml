graph [
  directed 1
  node [
    id 0
    label "C:\Users\ecepeda\PycharmProjects\curso-chatbots\src\models_ia\call_model.py::get_embeddings"
    type "function"
  ]
  node [
    id 1
    label "C:\Users\ecepeda\PycharmProjects\curso-chatbots\src\models_ia\call_model.py::generate_answer"
    type "function"
  ]
  node [
    id 2
    label "C:\Users\ecepeda\PycharmProjects\curso-chatbots\src\models_ia\call_model.py::<module>"
    type "module"
  ]
  node [
    id 3
    label "loguru"
    type "import"
  ]
  node [
    id 4
    label "openai"
    type "import"
  ]
  node [
    id 5
    label "src.config.settings"
    type "import"
  ]
  node [
    id 6
    label "OpenAI"
    type "unknown"
  ]
  node [
    id 7
    label "client.embeddings.create"
    type "unknown"
  ]
  node [
    id 8
    label "logger.info"
    type "unknown"
  ]
  node [
    id 9
    label "KeyError"
    type "unknown"
  ]
  node [
    id 10
    label "client.chat.completions.create"
    type "unknown"
  ]
  node [
    id 11
    label "response.model_dump"
    type "unknown"
  ]
  node [
    id 12
    label "C:\Users\ecepeda\PycharmProjects\curso-chatbots\src\models_ia\__init__.py::<module>"
    type "module"
  ]
  edge [
    source 0
    target 7
    type "call"
  ]
  edge [
    source 0
    target 8
    type "call"
  ]
  edge [
    source 0
    target 9
    type "call"
  ]
  edge [
    source 1
    target 10
    type "call"
  ]
  edge [
    source 1
    target 11
    type "call"
  ]
  edge [
    source 2
    target 6
    type "call"
  ]
]

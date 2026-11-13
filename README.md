# DocAgent AI

Plataforma de análise de documentos PDF desenvolvida para explorar conceitos de **RAG (Retrieval-Augmented Generation), agentes de IA, Tool Calling e busca semântica**.

O projeto foi desenvolvido como exercício prático para revisar e consolidar conhecimentos adquiridos em cursos da Alura.

## Funcionalidades

- Upload, visualização e gerenciamento de documentos PDF.
- Busca semântica utilizando embeddings locais.
- Chat com respostas fundamentadas nos documentos e citações das fontes.
- Agente de IA com LangGraph e ferramentas de busca.
- Memória conversacional persistida no PostgreSQL.
- Cadastro, autenticação JWT e isolamento de documentos por usuário.
- Proteções básicas de LLMSecOps, incluindo testes contra prompt injection.

## Tecnologias

| Área | Tecnologias |
|---|---|
| Backend | Python, FastAPI, SQLAlchemy |
| Frontend | React, TypeScript, Vite |
| Banco de dados | PostgreSQL, pgvector |
| Embeddings | MiniLM, ONNX Runtime |
| IA generativa | Groq |
| Agentes | LangGraph, LangChain Core |
| Infraestrutura | Docker Compose |
| Testes | Pytest, Playwright |

## Arquitetura

O processamento dos documentos segue o fluxo:

```text
PDF → Extração de texto → Chunking → Embeddings
                                      ↓
                              PostgreSQL + pgvector
                                      ↓
Pergunta → Busca semântica → RAG / Agente LangGraph
                                      ↓
                              Groq → Resposta com fontes
```

## Execução local com Docker

**Pré-requisitos:** Docker Desktop com Docker Compose e uma chave de API da Groq.

1. Crie `infra/.env.docker` a partir de `infra/.env.example` e configure as variáveis necessárias, incluindo senhas, segredo JWT e chave da Groq.

2. Na raiz do projeto, execute:

```bash
docker compose --env-file infra/.env.docker -f infra/compose.yaml build

docker compose --env-file infra/.env.docker -f infra/compose.yaml up -d postgres

docker compose --env-file infra/.env.docker -f infra/compose.yaml --profile setup run --rm migrate

docker compose --env-file infra/.env.docker -f infra/compose.yaml --profile setup run --rm prepare-model

docker compose --env-file infra/.env.docker -f infra/compose.yaml up -d api frontend
```

3. Acesse a aplicação em **http://localhost:5173**.

Para criar uma conta pela interface, habilite `REGISTRATION_ENABLED=true` no arquivo de configuração Docker.

## Segurança

O projeto inclui autenticação JWT, controle de acesso aos documentos, validação de citações, limites de execução do agente e medidas básicas contra prompt injection.

## Limitações

- Aceita PDFs digitais de até 5 MiB e 20 páginas.
- Não possui OCR para documentos digitalizados.
- A geração de respostas depende da API da Groq.
- Desenvolvido para execução local, não para exposição pública em produção.

## Objetivo

O DocAgent AI é um projeto pessoal de estudo, voltado à aplicação e revisão de conceitos de LLMs, RAG, LangChain, LangGraph e engenharia de software.

O frontend foi desenvolvido com auxílio de LLMs, enquanto o foco do projeto esteve na integração dos componentes de IA e backend.

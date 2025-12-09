# Setting up Qdrant

1. Spin up qdrant
```
docker compose up 
```
2. Add all cosmic chunks to the data/cosmic_chunks directory

3. Run the chunk ingesting 

```
python ingest_chunks.py
```
4. Run the cosmic and ticket agent

```
python ticket_creation_main.py
```
# Setup the environment

1. Create a virtual env of your choosing and then 
```
pip install -r requirements.txt
```
# Setting up Qdrant

1. Spin up qdrant
```
docker compose up 
```
2. Add all cosmic chunks to the data/cosmic_chunks directory (Add a few if you are just testing things)

3. Run the chunk ingesting 

```
python ingest_chunks.py
```

4. Run the cosmic and ticket agent

```
python ticket_creation_main.py
```

* Start with a cosmic query and then say 'i want to create a ticket from this' or something similar to start the ticket creation process. 
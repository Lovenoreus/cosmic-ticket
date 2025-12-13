# RUN INSTRUCTIONS
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# powershell -ExecutionPolicy Bypass
# .\cycle.ps1
docker compose down
docker compose build
docker compose up

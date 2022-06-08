@echo off & setlocal enabledelayedexpansion

for /f "tokens=1,2 delims==" %%a in (config.ini) do (
	if "%%a"=="TOTAL_CLIENTS" (
		set "TOTAL_CLIENTS=%%b"
		echo TOTAL_CLIENTS: !TOTAL_CLIENTS!
		start cmd.exe /k python server.py
		for /l %%x in (1, 1, !TOTAL_CLIENTS!: ) do (
			start cmd.exe /k python client.py -i %%x
		)
	)
)
# Prompt — Implementazione DataAgent

## Contesto del progetto

Sto costruendo un sistema multi-agent in Python per anomaly detection su dati di transiti aeroportuali. Il progetto confronta una pipeline classica (già implementata) con una pipeline multi-agent. L'architettura usa LangGraph come orchestratore e LangChain per i singoli agenti.

La struttura della repo è:
```
classical-vs-multiagent/
├── classical_pipeline/
│   └── preprocessing.py          ← già implementato, NON modificare
├── multiagent_pipeline/
│   ├── src/
│   │   └── features.py           ← già implementato, NON modificare
│   ├── state.py                  ← già implementato, NON modificare
│   └── agents/                   ← qui devi creare data_agent.py
├── data/
│   └── processed/
│       ├── allarmi_clean.csv
│       ├── viaggiatori_clean.csv
│       └── dataset_merged.csv
└── requirements.txt
```

## File esistenti rilevanti

### multiagent_pipeline/state.py (già scritto — usalo come riferimento)
```python
class AgentState(TypedDict):
    perimeter: dict
    df_raw: Optional[Any]          # output di DataAgent
    data_meta: Optional[dict]      # output di DataAgent
    df_features: Optional[Any]
    feature_meta: Optional[dict]
    df_baseline: Optional[Any]
    baseline_meta: Optional[dict]
    df_anomalies: Optional[Any]
    anomaly_meta: Optional[dict]
    report: Optional[dict]
    report_path: Optional[str]

class Perimeter(BaseModel):
    anno: Optional[int]
    aeroporto_arrivo: Optional[str]
    aeroporto_partenza: Optional[str]
    paese_partenza: Optional[str]
    zona: Optional[int]

PATHS = {
    "dataset_merged": "data/processed/dataset_merged.csv",
    ...
}
```

### data/processed/dataset_merged.csv — colonne principali
```
AREOPORTO_ARRIVO, AREOPORTO_PARTENZA, ANNO_PARTENZA, MESE_PARTENZA,
PAESE_PART, ZONA, TOT, MOTIVO_ALLARME, flag_rischio,
tot_entrati, tot_allarmati, tot_investigati,
tasso_allarme_volo, tasso_inv_volo, n_respinti, n_fermati, n_segnalati
```

---

## Cosa devi implementare

Crea il file `multiagent_pipeline/agents/data_agent.py`.

### Responsabilità del DataAgent
Il DataAgent è il primo nodo del grafo LangGraph. Riceve un perimetro di filtro dall'utente, carica il dataset merged già pulito dal preprocessing, applica i filtri e restituisce il DataFrame filtrato con le sue statistiche nello stato condiviso.

È un agente **deterministico**: non usa un LLM. Esegue sempre gli stessi step in sequenza. Questo è intenzionale: il DataAgent sa esattamente cosa fare, un LLM sarebbe spreco di risorse e latenza.

### I 3 tool da implementare

Ogni tool deve avere:
- Decorator `@tool` di LangChain
- Docstring chiaro (l'LLM lo legge per capire quando usarlo)
- Type hints corretti
- Gestione degli errori con messaggi leggibili

**Tool 1 — `load_dataset`**
- Signature: `load_dataset(path: str) -> str`
- Carica il CSV dal percorso indicato con `pd.read_csv`
- Restituisce il DataFrame serializzato come JSON string (orient="records")
- Se il file non esiste, restituisce un messaggio di errore chiaro

**Tool 2 — `filter_by_perimeter`**
- Signature: `filter_by_perimeter(data_json: str, anno: Optional[int], aeroporto_arrivo: Optional[str], aeroporto_partenza: Optional[str], paese_partenza: Optional[str], zona: Optional[int]) -> str`
- Deserializza il JSON, applica i filtri solo se il parametro non è None
- I filtri sono: ANNO_PARTENZA, AREOPORTO_ARRIVO, AREOPORTO_PARTENZA, PAESE_PART, ZONA
- I confronti stringa devono essere case-insensitive con `.str.upper()`
- Se dopo il filtro il DataFrame è vuoto, restituisce un messaggio di errore chiaro
- Restituisce il DataFrame filtrato come JSON string

**Tool 3 — `get_dataset_stats`**
- Signature: `get_dataset_stats(data_json: str) -> str`
- Deserializza il JSON e calcola le seguenti statistiche:
  - `n_righe`: numero di righe
  - `n_rotte_uniche`: numero di rotte distinte (AREOPORTO_PARTENZA + "-" + AREOPORTO_ARRIVO)
  - `anni_presenti`: lista degli anni unici in ANNO_PARTENZA
  - `paesi_partenza_top5`: top 5 paesi per frequenza in PAESE_PART
  - `n_con_allarmi`: numero di righe con TOT > 0
- Restituisce le statistiche come JSON string

### La funzione nodo per LangGraph

Dopo i tool, implementa la funzione `data_agent_node(state: AgentState) -> AgentState` che:
1. Legge `state["perimeter"]` e lo converte in un oggetto `Perimeter`
2. Chiama i 3 tool in sequenza (deterministico, non LLM-driven):
   - `load_dataset` con il path da `PATHS["dataset_merged"]`
   - `filter_by_perimeter` con i parametri del perimetro
   - `get_dataset_stats` sul risultato filtrato
3. Deserializza il JSON finale in un DataFrame pandas
4. Popola `state["df_raw"]` con il DataFrame
5. Popola `state["data_meta"]` con il dizionario delle statistiche
6. Stampa un log sintetico: n_righe, n_rotte, filtri applicati
7. Restituisce lo stato aggiornato

### Gestione errori
Se uno dei tool fallisce, `data_agent_node` deve loggare l'errore e restituire lo stato con `df_raw = None` e `data_meta = {"error": "messaggio"}`, senza lanciare eccezioni che bloccherebbero il grafo.

---

## Requisiti tecnici

- Usa `from langchain_core.tools import tool` per il decorator
- Usa `from multiagent_pipeline.state import AgentState, Perimeter, PATHS`
- Il file deve essere eseguibile standalone per testarlo: aggiungi un blocco `if __name__ == "__main__":` che esegue un test con perimetro `{"anno": 2024, "paese_partenza": "Algeria"}` e stampa le statistiche
- Non usare variabili globali
- Tutti i import devono essere in cima al file

## Import necessari
```python
import json
import pandas as pd
from typing import Optional
from langchain_core.tools import tool
from multiagent_pipeline.state import AgentState, Perimeter, PATHS
from pathlib import Path
```

## Output atteso dal test standalone
```
[DataAgent] Caricamento dataset: data/processed/dataset_merged.csv
[DataAgent] Filtri applicati: anno=2024, paese_partenza=Algeria
[DataAgent] Risultato: 143 righe, 28 rotte uniche
[DataAgent] Top 5 paesi: ['Algeria']
[DataAgent] ✓ Completato
```

# Classical vs Multi-Agent — Riepilogo Progetto
**Reply × LUISS 2026**

---

## Contesto

Il progetto è commissionato da Reply per LUISS. Il cliente reale è rappresentato da autorità di controllo frontaliero e operatori aeroportuali che gestiscono grandi volumi di transiti passeggeri. Ogni transito ha metadati associati: timestamp, gate, rotta, nazionalità, tipo documento, esito del controllo e allarmi di sicurezza.

Il problema attuale è che il rilevamento delle anomalie è reattivo. L'obiettivo è costruire un sistema analitico proattivo capace di identificare pattern sospetti prima che diventino incidenti operativi o di sicurezza.

Il progetto richiede di implementare lo stesso sistema di anomaly detection **due volte**: una con una pipeline classica e una con un'architettura multi-agent. Il risultato finale deve essere un'analisi comparativa che argomenta quale approccio è più conveniente e in quali condizioni operative.

---

## Dataset

Il dataset di input è un file CSV con dati di transiti aeroportuali italiani (fonte: dataset ALLARMI e TIPOLOGIA_VIAGGIATORE). Le colonne principali sono rotta, paese di partenza, zona geografica, tipo di occorrenza, motivo dell'allarme, esito del controllo, numero di passeggeri entrati, allarmati e investigati.

Il preprocessing è **condiviso** tra pipeline classica e multi-agent, garantendo un confronto onesto. Produce tre file puliti: `allarmi_clean.csv`, `viaggiatori_clean.csv`, `dataset_merged.csv`.

---

## Pipeline Classica (già implementata)

La pipeline classica segue un flusso lineare in 5 step implementati in notebook Jupyter:

1. **Feature Engineering** — 54 feature numeriche aggregate per rotta, costruite da pivot delle occorrenze, percentuali per motivo allarme, tassi di chiusura/rilevanza e profilo demografico dei viaggiatori. Feature chiave: `tot_allarmi_log`, `pct_interpol`, `score_rischio_esiti`, `tasso_rilevanza`.

2. **Baseline Construction** — baseline storica per 11 feature chiave, con z-score per misurare la deviazione di ogni rotta dalla norma.

3. **Anomaly Detection** — ensemble di 4 modelli: IsolationForest (peso 35%), LOF (30%), Z-score (15%), Autoencoder (20%). Score composito finale da 0 a 1.

4. **Post-Processing** — soglie data-driven (p97/p90) per classificare le rotte in ALTA, MEDIA, NORMALE. 17 rotte ALTA, 40 MEDIA, 510 NORMALE su 567 totali.

5. **Output** — `final_report.csv` con rotte rankate, score e metriche di rischio. Report statico.

---

## Architettura Multi-Agent (in sviluppo)

### Perché un'architettura a agenti

Un agente è un LLM combinato con un loop Think → Act → Observe → Reply. I tool sono funzioni Python che l'LLM può decidere di chiamare. Il valore aggiunto rispetto al classico non è negli algoritmi (sono gli stessi) ma in tre aspetti: flessibilità del perimetro di analisi, autonomia decisionale su quale algoritmo usare, e report narrativo generato dinamicamente da un LLM invece che statico.

### I 5 agenti

L'architettura scelta è **Supervisor**: un orchestratore LangGraph coordina 5 agenti specializzati in sequenza.

**Agent 1 — DataAgent**
Carica il dataset e filtra per il perimetro definito dall'utente (anno, aeroporto, paese, zona). È un agente deterministico: non usa un LLM, esegue sempre gli stessi step. Tool: `load_dataset`, `filter_by_perimeter`, `get_dataset_stats`. Output: DataFrame filtrato + metadati.

**Agent 2 — FeatureAgent**
Ricalcola le stesse 54 feature della pipeline classica usando le classi in `src/features.py`. Garantisce che il confronto sia onesto: stessa feature engineering, architettura diversa. Tool: `build_features`, `get_feature_cols`. Output: DataFrame aggregato per rotta.

**Agent 3 — BaselineAgent**
Carica la baseline precomputata o la ricalcola dinamicamente. Calcola z-score per ogni rotta rispetto alla baseline storica. Tool: `load_baseline`, `compute_zscore`. Output: DataFrame con deviazioni dalla norma.

**Agent 4 — OutlierAgent**
Applica IsolationForest, LOF, Z-score e Autoencoder con gli stessi pesi del classico (35/30/15/20). Produce lo stesso formato di output del classico, permettendo il confronto diretto. Tool: `run_isolation_forest`, `run_lof`, `run_zscore`, `run_autoencoder`, `ensemble_scores`. Output: rotte con score, label ALTA/MEDIA/NORMALE e rank.

**Agent 5 — ReportAgent**
L'unico agente con un LLM vero. Prende le rotte anomale e genera una spiegazione narrativa per ciascuna in linguaggio naturale. Questo è il differenziatore principale rispetto al classico, che produce solo numeri. Tool: `format_route_for_llm`, `generate_explanation`, `export_report`. Output: JSON con spiegazioni testuali per ogni anomalia.

### Nota importante sull'uso dell'LLM

Nel sistema multi-agent non tutti gli agenti usano un LLM. DataAgent, FeatureAgent, BaselineAgent e OutlierAgent sono deterministici: sanno esattamente cosa fare e non hanno bisogno di ragionamento autonomo. Solo il ReportAgent usa un LLM, perché la generazione di spiegazioni narrative è un compito ambiguo che beneficia del linguaggio naturale. Questa distinzione è una scelta architetturale deliberata, rilevante da mostrare nella presentazione finale.

### Stack tecnologico

- **LangGraph** — framework per il grafo di agenti e il flusso di stato
- **LangChain** — wrapping dei tool e gestione del singolo agente
- **OpenAI / Anthropic API** — LLM per il ReportAgent
- **scikit-learn, pyod** — stessi algoritmi ML del classico
- **pandas, numpy, statsmodels** — feature engineering e baseline
- **Streamlit** — interfaccia utente finale

---

## Struttura della repo

```
classical-vs-multiagent/
├── classical_pipeline/
│   └── preprocessing.py          ← condiviso con multi-agent
├── multiagent_pipeline/
│   ├── src/
│   │   ├── __init__.py
│   │   └── features.py           ← 6 classi di feature engineering
│   ├── agents/                   ← un file per agente (da implementare)
│   ├── state.py                  ← contratto dati condiviso (AgentState)
│   └── orchestrator.py           ← grafo LangGraph (da implementare)
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/                    ← pipeline classica
├── streamlit_app/                ← UI finale
└── requirements.txt
```

---

## Ordine di sviluppo (best practice reale)

1. **Contratto dei dati** ✅ — `state.py` con `AgentState`, schemi Pydantic, costanti condivise
2. **Classi src** ✅ — `src/features.py` con 6 classi, verificato match perfetto con il classico (567/567 rotte, diff = 0.000000)
3. **Tool Python** — funzioni pure testabili in isolamento, senza LLM
4. **Agenti in isolamento** — ogni agente gira e viene testato da solo
5. **Orchestratore** — i nodi vengono collegati nel grafo LangGraph
6. **ReportAgent con LLM** — aggiunto per ultimo, il più costoso da testare
7. **Streamlit UI** — costruita sopra quando la pipeline è stabile

---

## Interfaccia Streamlit

L'utente seleziona i filtri (anno, aeroporto, paese) e vede tre sezioni:

**Tabella rotte anomale** — ordinata per score, con colonne rotta, paese, livello di rischio (ALTA/MEDIA/NORMALE), score numerico e numero di modelli che l'hanno flaggata.

**Report narrativo** — per ogni rotta ALTA, una spiegazione in linguaggio naturale generata dal ReportAgent che descrive il pattern anomalo, i segnali rilevati e le possibili cause.

**Confronto classico vs multi-agent** — le stesse rotte con i due score affiancati e un grafico scatter che mostra la concordanza tra i due approcci. Questa è la sezione più importante per la presentazione con Reply.

Il report finale viene salvato su disco come `multiagent_report.json`.

---

## Domande per il checkpoint con il professore e Reply

- È necessario che ogni agente abbia un LLM, o basta che il sistema nel complesso sia agentico?
- Come volete che valutiamo la qualità delle anomalie rilevate, visto che non c'è una ground truth?
- Per "tool realistici" intendete chiamate a API/database esterni, o bastano funzioni Python ben definite?
- La UI Streamlit è richiesta per il checkpoint o solo per la presentazione finale?
- Nella vostra esperienza in Reply, in quali condizioni operative il classico batte il multi-agent?

---

## Metriche di confronto

Per argomentare quale approccio è più conveniente, le metriche da misurare sono:

- **Concordanza** tra i due approcci sulle rotte flaggate (% di accordo)
- **Tempo di esecuzione** end-to-end su stesso dataset
- **Flessibilità** del perimetro di analisi (il multi-agent è più adattabile)
- **Qualità del report** (valutazione qualitativa: numeri statici vs spiegazioni narrative)
- **Costo operativo** (il multi-agent chiama un LLM API a pagamento)

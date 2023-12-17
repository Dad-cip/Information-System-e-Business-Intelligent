Il seguente progetto riporta 4 esercizi:
1. Notbook Jupiter con analisi e modelli addestrati dei dataset forniti -> "Male Parasites Under The Lens.ipynb"
2. Dashboard streamlit con analisi e addestramenti interattivi -> "Dashboard.py"
3. Modello Power BI con analisi esplorativa dei dati -> "EDA_ExploratoryDataAnalysis.pbix"
4. Workflow Bonita con analisi di un flusso di processo aziendale -> "Shipments.bos"

Per l'avvio del 2 esercizio è necessario avere installato un ambiente virtuale 'venv' all'interno della cartella di progetto, 
che nello specifico dovrà contenere una versione di py 3.11.7 (o altre omologate per l'installazione di tensorflow).
Dopo averlo attivato, da terminale (in ambiente virtuale) bisogna lanciare il comando:
**pip install -r .\requirements.txt**
**streamlit run .\Dashboard.py**

Per il corretto utilizzo dell'4 sono necessarie delle estensioni, quali:
- Groovy
- Bonita Admin Application
- Email
- REST
- Database
In particolare, bisogna configurare opportunamente il provider fittizio,
seguendo la relativa documentazione: 'https://documentation.bonitasoft.com/bonita/2021.1/configure-email-connector'.

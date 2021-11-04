# Machine-Learning-Project
Implementation of a neural network from scratch


# Appunti Mike
### Inserirò qui quello che penso.

1 - Userei il formato tutto minuscolo e _ per i nomi. A parte per le
sigle, che servono tutte maiuscole.  

2 - Ora mi metto a modificare le cartelle

         2.1 - Farò una cartella di modello, dove metteremo tutte le informazioni del
modello, un file di classe, ed un file che allena minimo. Poggerei li tutte le
cose relative al modello, e poi un file contenenti tutte le funzioni non
relative al modello in modo intrinseco. (aka non faranno parte della classe)  

	 2.2 - Cartella contenente le configurazioni, scritte in json, ci sarà
	 un file di esempio.
	 
	 2.3 - Cartella contenente i dati. Non credo che avremo bisogno di
	 preprocessing, però nel caso divideremo entrambi.
	 
	 2.4 - Ho mentito, se vogliamo fare k-fold cross validation ha senso
	 preprocessare i dati e creare cartelle ad hoc per ogni fold.  
	 

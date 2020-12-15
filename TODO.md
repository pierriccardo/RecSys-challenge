# TODO

- prova p3alpha con implicit=True






- normalization on similarity matrix
- round robin
- tuning with different similarity (cosine, jaccard)
- user KNN CF CB

- check for cold users OK
- data analysis
    - average interaction per user
    - min interaction per user
    - max interaction per user
    - same for ICM
    - how many cold users # IMPORTANTE

- mettere metriche su utils ?
- importare le metriche su recommender ?
- spostare la evaluation su recommender ?

- trasformazione okapi
- traformazione TF iDF da una sim -> ibridala 


- dataset con dicts
- deterministic shuffle on dataset
- get_r_hat on each model
- save and load a model
- normalize each similarity matrix before the alpha 



- normalizzazione qunado ha senso farla (urm) (rhat)
- pipeline ha senso passare la r hat
- ...


- user KNN
urm normalizzo per riga
urm moltiplica per icm -> user feature
user feature x user feature T -> w
w x urm


Buonasera prof.,
le mandiamo questa mail per chiederle una mano a chiarire alcuni dubbi che abbiamo in merito alla recsys challenge.
In particolare i nostri dubbi riguardano:
- Algoritmi di matrix factorization:
    purtroppo le nostre implementazioni degli algoritmi di matrix factorization (sia quelli visti a lezione che quelli della sua repository github) non riescono a
    performare neanche lontanamente quanto gli algoritmi di similarity matrix. Perciò ci è venuto il dubbio che, più che gli algoritmi siano inefficienti in sè,
    siamo noi che probabilmente abbiamo capito male come implementarli.
    Il nostro metodo consiste nel creare i nparray user*factors e item*factors, e poi costruire il nostro modello facendo un dot product dei due.
    Sfortunatamente i risultati (in termini di MAP) sono molto scarsi, perciò volevamo capire se stiamo percorrendo la strada giusta (a livello di implementazione)
    oppure se ci sfugge qualcosa;

    secondo lei conviene concentrare gli sforzi su questi o provare altro? è possibile che questi tipi citati non siano ottimali per il problema affrontando? oppure stiamo sbagliando qualcosa nell'implementazione?
    
- Normalizzazioni
    su questa parte invece abbiamo provato a utilizzare le normalizzazioni, sia per quanto riguarda le urm (prima di utilizzarle nei nostri algoritmi) sia per 
    quanto riguarda il modello finale. Alcuni algoritmi hanno risposto bene a questa normalizzazione, mentre altri meno. Il nostro dubbio riguarda proprio il fatto
    di capire in che contesto usare le normalizzazioni, onde evitare di fare confusione e di utilizzarle inappropriatamente.

    soprattutto perchè quando facciamo un hybrido con le similarity oppure un hybrido ottenuto unendo in percentuale la matrice degli scores stranamente normalizzare ci porta ad un risultato peggiore, questo non riusciamo a spiegarcelo a livello teorico in quanto la normalizzazione dovrebbe solamente servire a equilibrare i valori delle due matrici. Le normalizzazioni che abbiamo provato sono quelle di sklearn.preprocessing normalize e di qeuste avbbiamo provato con l1, l2, max alcune le migliori sono solitamente l1, l2 ma i risultati migliori sono ottenuti sempre senza normalizzare

    
    
- ...

La ringraziamo in anticipo per l'aiuto e ci scusi per il disturbo.
Buona serata,
Daniele e Pierriccardo.


var = [0, 0, 0]
len_recs = len(var)
granularity = 5
temp = 0

u = 1 / ( len_recs * granularity ) 
i = granularity * len_recs

print(2.33 % 2)

'''
for r in range(0, len_recs):
    for r2 in range(0, len_recs):
        for it in range(0, i**(r+1)):
            var[r2] += u if it % i**r == 0 else 0
            #temp+=1
            print('|{:.3f}|{:.3f}|{:.3f}|'.format(var[0], var[1], var[2]))
'''
from math import factorial # python math library

i = 5               # i is the lexicographic index (counting starts from 0)
n = 3               # n is the length of the permutation
p = range(1, n + 1) # p is a list from 1 to n

for k in range(1, n + 1): # k goes from 1 to n
    f = factorial(n - k)  # compute factorial once per iteration
    d = i // f            # use integer division (like division + floor)
    print(p[d]),          # print permuted number with trailing space
    p.remove(p[d])        # delete p[d] from p
    i = i % f             # reduce i to its remainder
        
  

    

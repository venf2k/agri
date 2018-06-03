# agri
DL Net Image Classification


Sample.py: legge due file contenenti rispettivamente traing e test set di immagini 64x64 pixel in formato HDF5 - relative a 6 possibili posizioni 
di una mano che indica i numeri da 0 (pugno chiuso) a 5 (pugno aperto) - e li trasforma in matrici a due dimensioni (numero di byte dell'immagine, 
numero di casi) pronti per essere dati in input auna rete neurale.
In questo caso la dimensione totale di ciascuna immagine è 12288 pari al numero di pixel (64x64=4096) dell'immmagine moltiplicato per 3 livelli di RGB.
Inoltre ad ogni immagine della matrice dei casi X è associata la label corrispondente nel vettore Y, 
ovvero il numero da 0 a 5 che mostra la mano nell'immmagine.

In ./datasets trovi i due dataset di esempio.

Nel nostro caso invece, in cui abbiamo deciso di suddividere ogni sample (apppezzamento di terreno appartenente ad una determinata categoria) 
in una lista di mattonelle 20x20, possiamo rappresentare il training set come una matrice di dimensione (numero di sample, 20, 20, x, 3), 
dove x è il numero di mattonelle che abbiamo deciso di prendere per ogni sample.

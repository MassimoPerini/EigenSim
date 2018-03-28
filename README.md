# LambdaProj

Parametri:
lam = Lambda_HO(URM_train, recompile_cython=False, sgd_mode="adagrad", pseudoInv=False, rcond = 0.18, check_stability=False, save_lambda=False, save_eval=False).  <br />
URM_train : la URM per il train.  <br />
recompile_cython : a True ri-compila con Cython (serve un compilatore di C++).  <br /> 
sgd_mode : "sgd": usa sgd, "adagrad": usa adagrad.  <br />
pseudoInv : a True esegue l'algoritmo con la pseudo inversa, a False con la trasposta.  <br />
rcond : viene utilizzato solo se pseudoInv = True. E' il parametro rcond per il calcolo della pseudoinversa con NumPy: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.pinv.html.  <br />

check_stability: se True esegue la valutazione della stabilità durante i test (aggiungendo 2 elementi al profilo e calcolando le variazioni delle misurazioni).  <br />
save_lambda : se True salva i lambda ad ogni valutazione in out/lambdas (deve essere creata la cartella).  <br />
save_eval : se True salva i risultati del test in out/evaluations (deve essere creata la cartella).  <br />
 <br />
lam.fit(epochs=12,URM_test=URM_test,sgd_mode="adagrad", learning_rate=0.0003, alpha=0, batch_size=URM_train.nnz, validate_every_N_epochs=1, start_validation_after_N_epochs=0, initialize = "zero") . 
 <br />
epochs : numero di epoche.  <br />
URM_test : la matrice di test.  <br />
sgd_mode : "sgd": usa sgd, "adagrad": usa adagrad.  <br />
learning_rate : learning rate.  <br />
alpha : coefficiente di regolarizzazione (l'ho sempre lasciato a 0).  <br />
batch_size : per la trasposta: quanti campionamenti eseguire in "batch" (il pre-calcolo facendo il prodotto con la trasposta).  Per la pseudoinversa deve essere messo ad 1.  <br />
validate_every_N_epochs = ogni quante epoche eseguire la valutazione.  <br />
start_validation_after_N_epochs = dopo quante epoche iniziare con la valutazione.  <br />
initialize : "zero": inizializza i lambda a 0, "random" : inizializza i lambda a random, "one" : inizializza i lambda a 1.  <br />

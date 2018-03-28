# LambdaProj

Parametri:
lam = Lambda_HO(URM_train, recompile_cython=False, sgd_mode="adagrad", pseudoInv=False, rcond = 0.18, check_stability=False, save_lambda=False, save_eval=False)
URM_train : la URM per il train
recompile_cython : a True ri-compila con Cython (serve un compilatore di C++)
sgd_mode : "sgd": usa sgd, "adagrad": usa adagrad
pseudoInv : a True esegue l'algoritmo con la pseudo inversa, a False con la trasposta
rcond : viene utilizzato solo se pseudoInv = True. E' il parametro rcond per il calcolo della pseudoinversa con NumPy: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.pinv.html

check_stability: se True esegue la valutazione della stabilità durante i test (aggiungendo 2 elementi al profilo e calcolando le variazioni delle misurazioni)
save_lambda : se True salva i lambda ad ogni valutazione in out/lambdas (deve essere creata la cartella)
save_eval : se True salva i risultati del test in out/evaluations (deve essere creata la cartella)

lam.fit(epochs=12,URM_test=URM_test,sgd_mode="adagrad", learning_rate=0.0003, alpha=0, batch_size=URM_train.nnz, validate_every_N_epochs=1, start_validation_after_N_epochs=0, initialize = "zero")

epochs : numero di epoche
URM_test : la matrice di test
sgd_mode : "sgd": usa sgd, "adagrad": usa adagrad
learning_rate : learning rate
alpha : coefficiente di regolarizzazione (l'ho sempre lasciato a 0)
batch_size : per la trasposta: quanti campionamenti eseguire in "batch" (il pre-calcolo facendo il prodotto con la trasposta). Per la pseudoinversa deve essere messo ad 1
validate_every_N_epochs = ogni quante epoche eseguire la valutazione
start_validation_after_N_epochs = dopo quante epoche iniziare con la valutazione
initialize : "zero": inizializza i lambda a 0, "random" : inizializza i lambda a random, "one" : inizializza i lambda a 1

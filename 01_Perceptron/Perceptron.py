import numpy as np


class Perceptron:

    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None  # defined in method fit

    def fit(self, X, y):

        """Fit training dat.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        """
        self.w_ = np.zeros(1 + X.shape[1])  # First position corresponds to threshold
        # TODO: Put your code (fit algorithm)
        # por cada iteración
        for _ in range(self.n_iter):
            #obtener predicción de cada sample
            for j in range(X.shape[0]):  # recorrer cada muestra
                print("- Bucle de samples " + str(j))
                aux_pred = self.predict(X[j])  # predicción del ejemplo j
                error = self.eta * (y[j] - aux_pred)  # diferencia entre etiqueta real y predicción
                self.w_[1:] += error * X[j]  # actualizar pesos de las características
                self.w_[0] += error  # actualizar bias/umbral


            """"
            for j in range(X.shape[0]):
                aux_pred = self.predict(X[j])
                print("- Bucle de samples " + str(j))
                # y luego ir actualizando valor de pesos
                for z in range(X.shape[1]):
                    print("   -- Bucle de features " + str(z))
                    error = self.eta*(y[z]-aux_pred[z])*X[j, z]
                    self.w_[z] = self.w_[z] + error
            """



        # TODO: Put your code
    def predict(self, X):
            """Return class label.
                First calculate the output: (X * weights) + threshold
                Second apply the step function
                Return a list with classes
            """
            prd = []
            prd_aux = self.w_[0]

            # Creo que es innecesario
            if len(X.shape) > 1:
                print("Samples and features")
                for i in range(X.shape[0]):
                    prd_aux = self.w_[0]
                    for j in range(X.shape[1]):
                        prd_aux += self.w_[j] * X[i][j]

                    if prd_aux >= 0:
                        prd.append(1)
                    else:
                        prd.append(-1)


            else:
                print("Features of only one sample")
                for i in range(X.shape[0]):
                    prd_aux += self.w_[i] * X[i]

                if prd_aux >= 0:
                        prd.append(1)
                else:
                        prd.append(-1)

            return prd

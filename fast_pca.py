import time
import numpy as np
import os
import scipy.linalg


'''
    Faster method to compute PCs of X from eigendecomposition of C=XX^T
    when num rows of X is much greater than num columns

    Please read through the comments to understand how this works


    TODO:
        - Add explained variance ratio calculation and plots
'''

class FastPCA:
    def __init__(self, quiet=False):
        self.calculated = False # used to check if calculate() method has been called
        if not quiet:
            self.warn()

    def warn(self):
        print("This program is made to calculate the principal components and loading vectors of a data matrix.")
        print("The data matrix must meet 2 conditions for this to be both accurate and effective:")
        print("\t- The number of rows is greater than the number of columns")
        print("\t- The rows of the data matrix represent features and the columns represent samples.")
        print("If either of these conditions is not met, it is not reccomended that you use this method.\n")
        print("What can be returned:")
        print("\t- The EOFs which are the COLUMNS of E.")
        print("\t- The principal components of X, where each ROW is a principal component timeseries.")
        print("Note that for both the data matrix X and the principal components, rows contain spatial features and columns contain temporal samples.")

    def calculate(self, X):

        self.calculated = True

        # Step 1: calculate C'
        self.Cprime = np.matmul(X.T, X)

        # Step 2: solve eigenvalue problem for C' to get its eigenvectors and eigenvalues
        self.delta, self.D = np.linalg.eig(self.Cprime) # delta is eigenvals, D is eigenvectors of C'
        self.delta = self.delta.real.astype(np.float32)

        # Step 3: project D onto X to get eigenvectors of C (these will be different than those for C')
        self.E = np.matmul(X, self.D).real.astype(np.float32) # get weird error if not casted back to real float32
        # The COLUMNS of E are the EOFs

        # step 4: project X onto the transpose of the EOFs (principal axes) to get the principal components
        self.pc = np.matmul( (self.E).T, X )
        # Each ROW in pc is a timeseries. Where temporal information is stored in the columns (i.e. num columns = num. years)

    def getPCs(self):
        if not self.calculated:
            raise ValueError("Whoops! You need to call the calculate() method before you can do this.")
        return self.pc

    def getEOFs(self, eigenvals=False):
        if not self.calculated:
            raise ValueError("Whoops! You need to call the calculate() method before you can do this.")
        if eigenvals:
            return self.E, self.delta
        return self.E

    def getExpVar(self):
        if not self.calculated:
            raise ValueError("Whoops! You need to call the calculate() method before you can do this.")

        total = np.sum(self.delta)
        evr = [(i/total) for i in sorted(self.delta, reverse=True)]
        return evr

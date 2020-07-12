import numpy as np

class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, k, x, y):
        if len(x)!=k or len(y)!=k:
            assert False,'Data generator: Length of x or y is not equal to k.'
        self.k=k
        self.x_list=x
        self.y_list=y

    # Get i-th fold
    def getFold(self, i):
        isFirst=True
        for p in range(self.k): 
            if p!=i:
                if isFirst:
                    train_data       = self.x_list[p]
                    train_targets    = self.y_list[p]
                    isFirst = False
                else:
                    train_data      = np.concatenate((train_data, self.x_list[p]))
                    train_targets   = np.concatenate((train_targets, self.y_list[p]))
            else:
                val_data    = self.x_list[p]
                val_targets = self.y_list[p]
        return train_data,train_targets,val_data,val_targets

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1,self.k):
            All_X = np.append(All_X,self.x_list[i], axis=0)
        return All_X

    # Get all label y
    def getY(self):
        All_Y = self.y_list[0]
        for i in range(1,self.k):
            All_Y = np.append(All_Y,self.y_list[i], axis=0)
        return All_Y

    # Get all label y
    def getY_one_hot(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)
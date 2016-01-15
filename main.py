#Matheus Eduardo Rodrigeus Freitag

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt


def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset =   genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
    target  =   [x[0] for x in dataset] #Target = The actual numbers that people have written
    train   =   [x[1:] for x in dataset] #Train = The information about the Pixels forming the numbers
    test    =   genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:] #The dataset to be predicted

    #create random forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2) #My computer has a multicore CPU so I'm setting n_jobs=2
    #training the random forest
    rf.fit(train, target)

    #Saving the results of the prediction with the test dataset and formatting the output
    savetxt('submission2.csv', rf.predict(test), delimiter=',', fmt='%1.0f')


if __name__=="__main__":
    main()

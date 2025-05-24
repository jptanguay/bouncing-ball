'''

 à partir des données prégénérées
 du déplacement d'une balle dans un espace carré
 (avec potentiellement des collisions 100% élastiques contre les parois)

 entrainer un MLPRegressor à prédire la position finale et le vecteur vélocité

 comme les coordonnées peuvent avoir une grandeur importante (de 0 à 224)
 par rapport à la vélocité (de -4 à 4)
 on utilise un scaller min max

 
'''

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


import pickle
    

def LoadData():
    with open("x.pickle", "rb") as f:
        X = pickle.load(f)
        #print(X)
            
    with open("y.pickle", "rb") as f:
        y = pickle.load(f)
        #print(y)

    return X,y

    
def MakeRegr():
    regr = MLPRegressor(
        random_state=None,        
        hidden_layer_sizes=tuple([100]*4),
        learning_rate="adaptive",
        max_iter=1000,
        batch_size='auto',
        early_stopping=True,
        n_iter_no_change=200,        
        tol=10e-4,
        verbose=True
    )
    return regr

##############################
# MAIN
##############################

X,y = LoadData()
X_train, X_test, y_train, y_test = train_test_split(X, y) 

pipe = make_pipeline(MinMaxScaler(), MakeRegr() )

print("fitting...")
pipe.fit(X_train, y_train)

with open('model.pickle','wb') as f:
    pickle.dump(pipe,f)



score = pipe.score(X_test, y_test)
print(score)

print("X:", X_test[:2])
print("y:", y_test[:2])


print("preds:", pipe.predict(X_test[:2]))




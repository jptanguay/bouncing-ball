'''
    2025-05-24


    Generer des données de déplacement d'une balle     
    pour entrainer un NN a prédire la prochaine position et la vélocité 
    
    La balle de rayon r se déplace à pratir d'un point x,y dans un espace carré de dimension BOX_WIDTH x BOX_HEIGHT.
    
       
        Les données d'entraitement "X" sont:
        
            la position de départ et 2 positions prises à interval de temps régulier (en fonction e de la vélocité)
            
        Les données "y" à prédire sont:
        
            la position après 3 intervals de temps, et la vélocité finale (sous forme d'un vecteur vx, vy)
                (note: s'il y a collision avec un ou deux côtés de la boîte, alors vx et/ou vy seront inversé)
            
        
         Ex.: Données d'entrainement générées:
         
             X                  y
             ---------------------------------
             x,y,x1,y1,x2,y2    x3,y3,vxf,vyf  
             (...)              (...)
             
    
'''


import random
import math
import pickle
 
 
BOX_WIDTH = 224
BOX_HEIGHT = 224




##########################
# functions
##########################
'''
    calculer la nouvelle position de la balle (selon un axe x ou y)
    
        à partir de la position initiales "position" pour une durée "delta_t" à unw vélocité "velocity"
        size = l'espace disponible entre les bornes
        radius = le rayon de la balle
        
        on utilise une fonction triangulaire pour modéliser l'environnement et faire les calcul
    
        https://www.desmos.com/calculator/
        https://en.wikipedia.org/wiki/Triangle_wave
            p * abs(t/p - math.floor(t/p+1.0/2))
            
        pour w = 3, et r = 1 => ww = 3, utiliser p = 6
        # px(4,.5,2,8)
'''
def CalcNewPos(size, radius, position, velocity, delta_t):
    # position rescalée dans nouveau système de coordonnées + déplacement durant delta_t
    xx = (position - radius) + delta_t*velocity
    # calculer la période
    period = (size - radius * 2 ) * 2 
    
    temp_position = period * abs( (xx)/period - math.floor( (xx)/period + 1/2) ) 
    # direction
    #print("xx", xx)
    if (xx % period) > period/2.0:
        velocity = - abs(velocity)
    else:
        velocity = velocity

    if velocity == 0 or math.floor(xx / period/2)  % 2 == 0:
        pass       
    else:        
        velocity = -velocity
        
    # position dans système de coordonnées d'origine ( + radius)
    new_position = temp_position + radius
    return [new_position, velocity]



'''
    générer n_samples
    retourne X et y

'''
def GenData(n_samples=10, w=BOX_WIDTH, h=BOX_HEIGHT, r=2, delta_t = 5, debug=False):

    MIN_VELOCITY = -4
    MAX_VELOCITY = 4
    
    NB_FEATURE_POINTS = 3
    
    samples = []
    
    for i in range(n_samples):
        
        x, y = random.randint(r,w-r), random.randint(r,h-r)
        vx, vy = random.randint(MIN_VELOCITY,MAX_VELOCITY), random.randint(MIN_VELOCITY,MAX_VELOCITY)
        sample = [int(round(x)), int(round(y)), vx, vy]
            
        for j in range(NB_FEATURE_POINTS):                        
            if debug:
                print(f"delta_t {delta_t}, vx {vx}, vy {vy}")
            x, vx = CalcNewPos(w,r,x,vx,delta_t)
            y, vy = CalcNewPos(h,r,y,vy,delta_t)
            sample.extend([int(round(x)), int(round(y)), vx, vy])

        if debug:
            print(f"delta_t {delta_t}, vx {vx}, vy {vy}")
            
        samples.append(sample)
        
        
    
    X = []
    y = []
    for line in samples:
        
        # position de départ + 2 positions suivantes
        row_x = [line[0], line[1], line[4], line[5], line[8], line[9] ]        
        X.append(row_x)

        # la dernière position + vélocité finale
        row_y = line[12:16]
        y.append(row_y)

    return [X,y]
    


'''
    enregistrer les données X et y dans des fichiers séparés
'''
def Save(x, y):
    print("Saving X")
    with open("x.pickle", "wb") as fx:
        pickle.dump(X, fx)
        #print(X)
            
    print("Saving y")
    with open("y.pickle", "wb") as fy:
        pickle.dump(y, fy)
        #print(y)
    

################################################
# MAIN
################################################
 
X,y = GenData(10000,delta_t = 10 )
Save(X,y)


    


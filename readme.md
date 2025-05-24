## Bouncing ball - MLPRegressor

### Génération des données


    Generer des données de déplacement d'une balle     
    pour entrainer un NN a prédire la prochaine position et la vélocité 
    
    La balle de rayon r se déplace à partir d'un point x,y dans un espace carré de dimension BOX_WIDTH x BOX_HEIGHT.
    
       
        Les données d'entrainement "X" sont:
        
            la position de départ et 2 positions prises à intervalle de temps régulier (en fonction e de la vélocité)
            
        Les données "y" à prédire sont:
        
            la position après le 3ieme intervalle de temps, et la vélocité finale (sous forme d'un vecteur vx, vy)
                (note: s'il y a collision avec un ou deux côtés de la boîte, alors vx et/ou vy seront inversé)
            
        
         Ex.: Données d'entrainement générées:
         
             X                  y
             ---------------------------------
             x,y,x1,y1,x2,y2    x3,y3,vxf,vyf  
             (...)              (...)
             
    
### Training

Entrainer un MLPRegressor à prédire la position finale et le vecteur vélocité
à partir des données prégénérées du déplacement d'une balle dans un espace carré
(avec potentiellement des collisions 100% élastiques contre les parois)

comme les coordonnées peuvent avoir une grandeur importante (jusqu'à 224)
par rapport à la vélocité (de -4 à 4)
on utilise un scaller min max

---

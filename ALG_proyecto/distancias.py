import numpy as np
import math

def levenshtein_matriz(x, y, threshold=None):
    # esta versión no utiliza threshold, se pone porque se puede
    # invocar con él, en cuyo caso se ignora
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]

def levenshtein_edicion(x, y, threshold=None):
    # a partir de la versión levenshtein_matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    camino = []
    indX = lenX
    indY = lenY

    while indX>0 or indY >0:
        
        xi = indX -1
        yi = indY
        c = D[xi,yi]
        op = (x[xi], "")

        if D[indX, indY -1] <= c:
            xi = indX
            yi = indY-1
            c = D[xi,yi]
            op = ("", y[yi])

        if D[indX-1, indY-1] <= c:
            xi = indX-1
            yi = indY-1
            c = D[xi,yi]
            op = (x[xi], y[yi])    

        camino.append(op)
        indX = xi
        indY = yi      

    camino.reverse()

    return D[lenX, lenY], camino


def levenshtein_reduccion(x, y, threshold=None):
    # completar versión con reducción coste espacial
    #2. Implementar Levenshtein con reducción de coste espacial y con un
    #parámetro umbral o threshold de modo que se pueda dejar de calcular
    #cualquier distancia mayor a dicho umbral.
    # COMPLETAR

    lenX, lenY = len(x), len(y)
    vcurrent = np.zeros(lenX + 1, dtype=np.int)
    vnext = np.zeros(lenX + 1, dtype=np.int)
    for i in range(1, lenX + 1):
        vcurrent[i] = vcurrent[i - 1] + 1
    for j in range(1, lenY + 1):
        vnext[0] = vcurrent[0] + 1
        for i in range(1, lenX + 1):
            vnext[i] = min(vcurrent[i] + 1, 
                            vnext[i - 1] + 1, 
                            vcurrent[i - 1] + (x[i - 1] != y[j - 1]),
            )
        vnext, vcurrent = vcurrent, vnext
    return vcurrent[lenX] 


def levenshtein(x, y, threshold):
    # completar versión reducción coste espacial y parada por threshold
    #2. Implementar Levenshtein con reducción de coste espacial y con un
    #parámetro umbral o threshold de modo que se pueda dejar de calcular
    #cualquier distancia mayor a dicho umbral.
     # COMPLETAR
    lenX, lenY = len(x), len(y)
    vcurrent = np.zeros(lenX + 1, dtype=np.int)
    vnext = np.zeros(lenX + 1, dtype=np.int)
    for i in range(1, lenX + 1):
        vcurrent[i] = vcurrent[i - 1] + 1
    for j in range(1, lenY + 1):
        vnext[0] = vcurrent[0] + 1
        paradaPorThreshold = True
        if(vnext[0] <= threshold): paradaPorThreshold = False
        elif(vnext[0] == threshold and lenX - i == lenY - j): paradaPorThreshold = False
        for i in range(1, lenX + 1):
            vnext[i] = min(vnext[i - 1] + 1, 
                            vcurrent[i] + 1,
                            vcurrent[i - 1] + (x[i - 1] != y[j - 1]))
            if(vnext[i] < threshold): paradaPorThreshold = False
            elif(vnext[i] == threshold and lenX - i == lenY - j): paradaPorThreshold = False
        if(paradaPorThreshold): return threshold+1
        vnext, vcurrent = vcurrent, vnext
    return vcurrent[lenX]

def levenshtein_cota_optimista(x, y, threshold):
    # COMPLETAR Y REEMPLAZAR ESTA PARTE
    dic = set(x)
    dic.update(set(y))

    res = { 1: 0,-1: 0}

    for letra in dic:
        dif = x.count(letra) - y.count(letra)
        if dif < 0:
            res[1] += abs(dif)
        else:
            res[-1] += abs(dif)


    res = max(res[1], res[-1])
    if res > threshold:
        return threshold + 1
    else:
        return levenshtein(x, y,  threshold)

def damerau_restricted_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein restringida con matriz
    #2EXTRA. Implementar la versión restringida de Damerau-Levenstein
    #(también con un parámetro umbral o threshold de modo que se pueda
    #dejar de calcular cualquier distancia mayor a dicho umbral). Es
    #automático que quede integrado en el recuperador.
    # COMPLETAR
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1))
    for i in range(1, lenX + 1):
        D[i, 0] = i
    for j in range(1, lenY + 1):
        D[0, j] = j
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            if i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:
                if x[i - 1] == y[j - 1]:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i-1][j-1], D[i-2][j-2] + 1)
                else:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i-1][j-1] + 1, D[i-2][j-2] + 1)
            else:
                if x[i - 1] == y[j - 1]:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i-1][j-1])
                else:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i-1][j-1] + 1)
    return D[lenX, lenY]
    


def damerau_restricted_edicion(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1))
    for i in range(1, lenX + 1):
        D[i, 0] = i
    for j in range(1, lenY + 1):
        D[0, j] = j
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            if i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:
                if x[i - 1] == y[j - 1]:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i-1][j-1], D[i-2][j-2] + 1)
                else:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i-1][j-1] + 1, D[i-2][j-2] + 1)
            else:
                if x[i - 1] == y[j - 1]:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i-1][j-1])
                else:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i-1][j-1] + 1)
    
    camino = []
    indX = lenX
    indY = lenY

    while indX>0 or indY >0:
        
        xi = indX -1
        yi = indY
        c = D[xi,yi]
        op = (x[xi], "")

        if D[indX, indY -1] <= c:
            xi = indX
            yi = indY-1
            c = D[xi,yi]
            op = ("", y[yi])

        if D[indX-1, indY-1] <= c:
            xi = indX-1
            yi = indY-1
            c = D[xi,yi]
            op = (x[xi], y[yi])  

        if D[indX-2, indY-2] <= c and x[indX-2]==y[indY-1] and x[indX-1]==y[indY-2]:
            xi = indX-2
            yi = indY-2
            c = D[xi,yi]
            op = (x[xi]+x[xi+1], y[yi]+y[yi+1])   

        camino.append(op)
        indX = xi
        indY = yi      

    camino.reverse()

    return D[lenX, lenY], camino

def damerau_restricted(x, y, threshold):
    # versión con reducción coste espacial y parada por threshold
    #2EXTRA.Implementar la versión restringida de Damerau-Levenstein
    #(también con un parámetro umbral o threshold de modo que se pueda
    #dejar de calcular cualquier distancia mayor a dicho umbral). Es
    #automático que quede integrado en el recuperador.
     # COMPLETAR Y REEMPLAZAR ESTA PARTE
    lenX, lenY = len(x), len(y)
    vec1 = np.zeros(lenX + 1, dtype=np.int) #(la fila anterior de las distancias)
    vec2 = np.zeros(lenX + 1, dtype=np.int)
    vec3 = np.zeros(lenX + 1, dtype=np.int) #(distancias de fila actuales) la calculamos con las filas previas vec0 y vec1
    for i in range(1, lenX + 1):
        vec1[i] = vec1[i - 1] + 1
    for j in range(1, lenY + 1):
        vec2[0] = vec1[0] + 1
        paradaPorThreshold = True
        if(vec2[0] <= threshold): paradaPorThreshold = False
        elif(vec2[0] == threshold and lenX - i == lenY - j): paradaPorThreshold = False
        for i in range(1, lenX + 1):
            if(i > 1 and j > 1 and (x[i - 2] == y[j - 1]) and (x[i - 1] == y[j - 2])):
            #usamos la formula para completar vec2
                vec2[i] = min(vec1[i] + 1,
                            vec2[i-1] + 1,
                            vec1[i-1] + (x[i-1] != y[j - 1]),
                            vec3[i - 2] + 1)
            else:
                vec2[i] = min(vec1[i] + 1,
                            vec2[i-1] + 1,
                            vec1[i-1] + (x[i-1] != y[j - 1]))
            if(vec2[i] < threshold): paradaPorThreshold = False
            elif(vec2[i] == threshold and lenX - i == lenY - j): paradaPorThreshold = False
        if(paradaPorThreshold): return threshold+1  
        vec1, vec2, vec3 = vec2, vec3, vec1 
    return vec1[lenX]


def damerau_intermediate_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein intermedia con matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1))
    for i in range(1, lenX + 1):
        D[i, 0] = i
    for j in range(1, lenY + 1):
        D[0, j] = j
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            minInit = 0
            if x[i - 1] == y[j - 1]:
                minInit = min(D[i-1, j] + 1, D[i, j-1] + 1, D[i-1][j-1])
            else:
                minInit = min(D[i-1, j] + 1, D[i, j-1] + 1, D[i-1][j-1] + 1)

            if j > 1 and i > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:
                D[i,j] = min(minInit, D[i-2][j-2] + 1)
            elif j > 2 and i > 1 and x[i-2] == y[j-1] and x[i-1] == y[j-3]:
                D[i,j] = min(minInit, D[i-2][j-3] + 2)
            elif i > 2 and j > 1 and x[i - 3] == y[j-1] and x[i-1] == y[j-2]:
                D[i,j] = min(minInit, D[i-3][j-2] + 2)
            else:
                D[i,j] = minInit
    return D[lenX, lenY]

def damerau_intermediate_edicion(x, y, threshold=None):
    # partiendo de matrix_intermediate_damerau añadir recuperar
    # secuencia de operaciones de edición
    # completar versión Damerau-Levenstein intermedia con matriz
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE
    
def damerau_intermediate(x, y, threshold):
    # versión con reducción coste espacial y parada por threshold
    # COMPLETAR Y REEMPLAZAR ESTA PARTE
    lenX, lenY = len(x), len(y)
    vec1 = np.zeros(lenX + 1, dtype=np.int)
    vec2 = np.zeros(lenX + 1, dtype=np.int)
    vec3 = np.zeros(lenX + 1, dtype=np.int)
    vec4 = np.zeros(lenX + 1, dtype=np.int)
    reglaNum = 0
    regla2Num = 0
    regla3Num = 0
    for i in range(1, lenX + 1):
        vec1[i] = vec1[i - 1] + 1
    for j in range(1, lenY + 1):
        vec2[0] = vec1[0] + 1
        paradaPorThreshold = True
        if(vec2[0] <= threshold): paradaPorThreshold = False
        elif(vec2[0] == threshold and lenX - i == lenY - j): paradaPorThreshold = False
        for i in range(1, lenX + 1):
            if(i > 1 and j > 1 and (x[i - 2] == y[j - 1]) and (x[i - 1] == y[j - 2])):
                reglaNum = vec3[i - 2] + 1
            else:
                reglaNum =  vec1[i] + 10
            if(i > 2 and j > 1 and (x[i - 3] == y[j - 1]) and (x[i - 1] == y[j - 2])):
                regla2Num = vec3[i - 3] + 2
            else:
                regla2Num =  vec1[i] + 10 #nunca es el minimo
            if(i > 1 and j > 2 and (x[i - 1] == y[j - 3]) and (x[i - 2] == y[j - 1])):
                regla3Num = vec4[i - 2] + 2
            else:
                regla3Num =  vec1[i] + 10 #nunca es el minimo
            #usamos la formula para completar vec2
            vec2[i] = min(
                vec1[i] + 1,
                vec2[i - 1] + 1,
                vec1[i - 1] + (x[i - 1] != y[j - 1]),
                reglaNum,
                regla2Num,
                regla3Num,
            )

            if(vec2[i] < threshold): paradaPorThreshold = False
            elif(vec2[i] == threshold and lenX - i == lenY - j): paradaPorThreshold = False
        if(paradaPorThreshold): return threshold+1  
        vec1, vec2, vec3, vec4 = vec2, vec4, vec1, vec3
    return vec1[lenX]



opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'levenshtein_r': levenshtein_reduccion,
    'levenshtein':   levenshtein,
    'levenshtein_o': levenshtein_cota_optimista,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_r':     damerau_restricted,
    'damerau_im':    damerau_intermediate_matriz,
    'damerau_i':     damerau_intermediate
}

opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}


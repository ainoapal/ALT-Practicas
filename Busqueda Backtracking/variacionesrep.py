# AUTOR:
# Ainoa Palomino Pérez

def variacionesRepeticion(elementos, cantidad):
    sol = [None]*cantidad
    def backtracking(longSol):
        if longSol == cantidad:
            yield sol.copy()
        else:
            for child in elementos:
                sol[longSol] = child
                yield from backtracking(longSol+1)
    yield from backtracking(0)



#COMPLETAR  --> Actividad 1
def permutaciones(elementos):
    N = len(elementos)
    sol = [None]*N

    def prometedor(child, longSol):
        if longSol == 0: return True
        else:
            for i in range(1, longSol+1):
                if sol[i-1] == child: return False
        return True

    def backtracking(longSol):
        if longSol == N:    #cambiamos cantidad por N
            yield sol.copy()
        else:
            for child in elementos:
                if prometedor(child, longSol): #simplemente le añadimos esta linea al codigo anterior
                    sol[longSol] = child
                    yield from backtracking(longSol+1)
    yield from backtracking(0)



#COMPLETAR  --> Actividad 2
def combinaciones(elementos, cantidad):
    sol = [None]*cantidad # vamos a tomar N elementos

    def prometedor(child, longSol):
        if longSol==0:
            return True
        if(child not in sol[:longSol] and sol[longSol-1] not in elementos[elementos.index(child):]): 
            return True
        else:
            return False

    def backtracking(longSol):
        if longSol == cantidad: #esta completo
            yield sol.copy()
        else:
            for child in elementos:
                if prometedor(child,longSol):
                    sol[longSol] = child
                    yield from backtracking(longSol+1)
    yield from backtracking(0)
    
    ''' Es el de combinaciones pero en vez de imprimir las cadenas, imprime números
    def combinaciones(elementos, cantidad):
    N = len(elementos)
    sol = [None]*cantidad # vamos a tomar N elementos
    def backtracking(longSol):
        if longSol == cantidad: #esta completo
            yield sol.copy()
        else:
            # tenemos que rellenar desde longSol hasta N-1, son N-longSol+1 valores
            desde = 1 if longSol==0 else sol[longSol-1]+1
            hasta = N-(cantidad-longSol)+1
            for i in range(desde,hasta+1):
                sol[longSol]=i
                yield from backtracking(longSol+1)
    yield from backtracking(0)
    '''
    


if __name__ == "__main__":    
    for x in variacionesRepeticion(['tomate','queso','anchoas'],3):
        print(x)
    print('---------------------------------------------')
    for x in permutaciones(['tomate','queso','anchoas']):
        print(x)
    print('---------------------------------------------')
    for x in combinaciones(['tomate','queso','anchoas', 'aceitunas'], 3):
        print(x)


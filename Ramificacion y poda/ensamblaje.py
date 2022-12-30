import numpy as np
import heapq
import collections
from collections import namedtuple

######################################################################
#
#                     GENERACIÓN DE INSTANCIAS
#
######################################################################

def genera_instancia(M, low=1, high=1000):
    return np.random.randint(low=low,high=high,
                             size=(M,M),dtype=int)

######################################################################
#
#                       ALGORITMOS VORACES
#
######################################################################

# Función que calcula el coste de una solución
def compute_score(costMatrix, solution):
    return sum(costMatrix[pieza,instante]
               for pieza,instante in enumerate(solution))

def naive_solution(costMatrix):
    solution = list(range(costMatrix.shape[0]))
    return compute_score(costMatrix,solution), solution

def voraz_x_pieza(costMatrix):
    # costMatrix[i,j] el coste de situar pieza i en instante j
    M = costMatrix.shape[0] # nº piezas

    # COMPLETAR

    # Inicializar la solución con -1
    solution = [-1] * M
    # Iterar a través de las tareas
    for i in range(M):
        min=100000
        for j in range(M):
            # Encuentra el intervalo de tiempo con el menor coste para la tarea actual
            if(costMatrix[i, j] < min and j not in solution):
                min = costMatrix[i, j]
                col = j
            # Asignar la tarea actual al intervalo de tiempo con el menor coste
            solution[i] = col
    # Calcular el coste total de la solución
    score = compute_score(costMatrix, solution)
    
    return score,solution

def voraz_x_instante(costMatrix):
    # costMatrix[i,j] el coste de situar pieza i en instante j
    M = costMatrix.shape[0] # nº piezas

    # COMPLETAR

    # Inicializar la solución con -1
    solution = [-1] * M
    solution_aux = [-1] * M
    # Iterar a través de los intervalos de tiempo
    for i in range(M):
        min=100000
        for j in range(M):
            # Encuentra la tarea con el menor coste para el intervalo de tiempo actual
            if(costMatrix[j,i]<min and i not in solution):
                min=costMatrix[j,i]
                col=i
                fila=j
        # Asigna la tarea con el menor coste al intervalo de tiempo actual
        solution_aux[i] = (fila,col)
    solution_aux=sorted(solution_aux, key=lambda i: i[0])
    for i in range(M):
        solution[i]=solution_aux[i][1]
    # Calcular el coste total de la solución
    score = compute_score(costMatrix, solution)
    
    return score,solution

def voraz_x_coste(costMatrix):
    # costMatrix[i,j] el coste de situar pieza i en instante j
    M = costMatrix.shape[0] # nº piezas

    # COMPLETAR

    # Inicializar la solución con -1
    solution = [-1] * M
    solution_aux = [-1] * M
    filas = [-1] * M
    cols = [-1] * M
    min=100000
    # Ordenar las tareas por coste
    #sorted_tasks = sorted(range(M), key=lambda i: costMatrix[i,0])
    # Iterar a través de las tareas ordenadas por coste
    for k in range(M):
        for i in range(M):
            for j in range(M):
                # Encuentra el intervalo de tiempo con el menor coste para la tarea actual
                if(costMatrix[i,j]<min and i not in filas and j not in cols):
                    min=costMatrix[i,j]
                    col=j
                    fila=i
        min=10000
        filas[k]=fila
        cols[k]=col 
        # Asignar la tarea actual al intervalo de tiempo con el menor coste
        solution_aux[k]=(fila,col)
    solution_aux=sorted(solution_aux, key=lambda i: i[0])
    for i in range(M):
        solution[i]=solution_aux[i][1]
    # Calcular el coste total de la solución
    score = compute_score(costMatrix, solution)
   
    return score,solution

def voraz_combina(costMatrix):
    
    # COMPLETAR

    # costMatrix[i,j] el coste de situar pieza i en instante j
    M = costMatrix.shape[0] # nº piezas
    # Obtener solución voraz x pieza
    score1, solution1 = voraz_x_pieza(costMatrix)
    # Obtener solución voraz x instante
    score2, solution2 = voraz_x_instante(costMatrix)
    # Obtener solución voraz x coste
    score3, solution3 = voraz_x_coste(costMatrix)

    # Devolver la solución con el menor coste
    if score1 < score2 and score1 < score3:
        return score1, solution1
    elif score2 < score1 and score2 < score3:
        return score2, solution2
    else:
        return score3, solution3
    #return score,solution
        
######################################################################
#
#                       RAMIFICACIÓN Y PODA
#
######################################################################

class Ensamblaje:

    def __init__(self, costMatrix, initial_sol = None):
        '''
        costMatrix es una matriz numpy MxM con valores positivos
        costMatrix[i,j] es el coste de ensamblar la pieza i cuando ya
        se han ensamblado j piezas.
        '''
        # no haría falta pero por si acaso comprobamos que costMatrix
        # es una matriz numpy cuadrada y de costMatrix positivos
        assert(type(costMatrix) is np.ndarray and len(costMatrix.shape) == 2
               and costMatrix.shape[0] == costMatrix.shape[1]
               and costMatrix.dtype == int and costMatrix.min()>=0)
        self.costMatrix = costMatrix
        self.M = costMatrix.shape[0]
        # la forma más barata de ensamblar la pieza i si podemos
        # elegir el momento de ensamblaje que más nos convenga:
        self.minPieza = [costMatrix[i,:].min() for i in range(self.M)]
        self.x = initial_sol
        if initial_sol is None:
            self.fx = np.inf
        else:
            self.fx = compute_score(costMatrix,initial_sol)
        
    def branch(self, s_score, s):
        '''
        s_score es el score de s
        s es una solución parcial
        '''
        i = len(s) # i es la siguiente pieza a montar, i<M
        
        # costMatrix[i,j] coste ensamblar objeto i en instante j
        for j in range(self.M): # todos los instantes
            # si j no ha sido utilizado en s
            if j not in s: # NO es la forma más eficiente
                           # al ser lineal con len(s)
                new_score = s_score - self.minPieza[i] + self.costMatrix[i,j]
                yield (new_score, s + [j])

    def is_complete(self, s):
        '''
        s es una solución parcial
        '''
        return len(s) == self.M

    def initial_solution(self):
        return (sum(self.minPieza),[])

    def solve(self):
        A = [ self.initial_solution() ] # cola de prioridad
        iterations = 0 # nº iteraciones
        gen_states = 0 # nº estados generados
        podas_opt  = 0 # nº podas por cota optimista
        maxA       = 0 # tamaño máximo alzanzado por A
        # bucle principal ramificacion y poda (PODA IMPLICITA)
        while len(A)>0 and A[0][0] < self.fx:
            iterations += 1
            lenA = len(A)
            maxA = max(maxA, lenA)
            s_score, s = heapq.heappop(A)
            for child_score, child in self.branch(s_score, s):
                gen_states += 1
                if self.is_complete(child): # si es terminal
                    # es factible (pq branch solo genera factibles)
                    # falta ver si mejora la mejor solucion en curso
                    if child_score < self.fx:
                        self.fx, self.x = child_score, child
                else: # no es terminal
                    # lo metemos en el cjt de estados activos si
                    # supera la poda por cota optimista:
                    if child_score < self.fx:
                        heapq.heappush(A, (child_score, child) )
                    else:
                        podas_opt += 1
                        
        stats = { 'iterations':iterations,
                  'gen_states':gen_states,
                  'podas_opt':podas_opt,
                  'maxA':maxA}
        return self.fx, self.x, stats

def functionRyP(costMatrix):
    e = Ensamblaje(costMatrix)
    fx,x,stats = e.solve()
    return fx,x

######################################################################
#
#                        EXPERIMENTACIÓN
#
######################################################################

cjtAlgoritmos = {'naif': naive_solution,
                 'x_pieza': voraz_x_pieza,
                 'x_instante': voraz_x_instante,
                 'x_coste': voraz_x_coste,
                 'combina': voraz_combina,
                 'RyP': functionRyP}
# Se le da a los algoritmos el mismo nombre que en las diapositivas (Diapo 19/19)
cjtAlgoritmosRyP = {'naif+RyP': naive_solution,
                 'x_pieza+RyP': voraz_x_pieza,
                 'x_instante+RyP': voraz_x_instante,
                 'x_coste+RyP': voraz_x_coste,
                 'combina+RyP': voraz_combina,
                 'RyP': functionRyP}

def probar_ejemplo():
    ejemplo = np.array([[7, 3, 7, 2],
                        [9, 9, 4, 1],
                        [9, 4, 8, 1],
                        [3, 4, 8, 4]], dtype=int)
    
    for label,function in cjtAlgoritmos.items():
        score,solution = function(ejemplo)
        print(f'Algoritmo {label:10}', solution, score)


def comparar_algoritmos():
    print('talla',end=' ')
    for label in cjtAlgoritmos:
        print(f'{label:>10}',end=' ')
    print()
    numInstancias = 10
    for talla in range(5,15+1):
        dtalla = collections.defaultdict(float)
        for instancia in range(numInstancias):
            cM = genera_instancia(talla)
            for label,function in cjtAlgoritmos.items():
                score,solution = function(cM)
                dtalla[label] += score
        print(f'{talla:>5}',end=' ')
        for label in cjtAlgoritmos:
            media = dtalla[label]/numInstancias
            print(f'{media:10.2f}', end=' ')
        print()

def comparar_sol_inicial():
    ejemplo = np.array([[7, 3, 7, 2],
                        [9, 9, 4, 1],
                        [9, 4, 8, 1],
                        [3, 4, 8, 4]], dtype=int)
    bb = Ensamblaje(ejemplo)
    fx, x, stats = bb.solve()
    numInstancias = 10
    for st in stats:
        print(st.center(70,"-"))
        print('talla',end=' ')
        for label in cjtAlgoritmosRyP:
            print(f'{label:>10}',end=' ')
        print()
        for talla in range(5,15+1):
            dtalla = collections.defaultdict(float)
            for instancia in range(numInstancias):
                cM = genera_instancia(talla)
                for lab in cjtAlgoritmosRyP:
                    score,sol = cjtAlgoritmosRyP[lab](cM)
                    e = Ensamblaje(cM,sol)
                    fx,x,stats = e.solve()
                    dtalla[lab] += stats[st]
            print(f'{talla:>5}',end=' ')
            for label in cjtAlgoritmosRyP:
                media = dtalla[label]/numInstancias
                print(f'{media:10.2f}', end=' ')
            print()
        

def probar_ryp():
    ejemplo = np.array([[7, 3, 7, 2],
                        [9, 9, 4, 1],
                        [9, 4, 8, 1],
                        [3, 4, 8, 4]], dtype=int)
    # scorevoraz, solvoraz = voraz_combina(ejemplo)
    # bb = Ensamblaje(ejemplo, solvoraz)
    bb = Ensamblaje(ejemplo)
    fx, x, stats = bb.solve()
    print(x,fx,compute_score(ejemplo,x))
    print(stats)
    
######################################################################
#
#                             PRUEBAS
#
######################################################################


if __name__ == '__main__':
    probar_ejemplo()
    print('-'*70)
    probar_ryp()
    print('-'*70)
    comparar_algoritmos()
    comparar_sol_inicial()
    


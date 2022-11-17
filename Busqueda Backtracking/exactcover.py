# AUTOR:
# Ainoa Palomino Pérez

def exact_cover(listaConjuntos):
    U = set().union(*listaConjuntos) # para saber qué universo tenemos
    N = len(listaConjuntos)
    solucion = []
    
    def backtracking(longSol, cjtAcumulado):
        # COMPLETAR
        # consulta los métodos isdisjoint y union de la clase set,
        # podrías necesitarlos
        if longSol == N:    #si es terminal
            if set().union(*solucion)==U:   #si es factible
                yield solucion.copy()
        else: # ramificar
            cjt = listaConjuntos[longSol]
            if set(cjt).isdisjoint(cjtAcumulado):   #si cjt y cjtAcumulado son disjuntos:
                solucion.append(cjt)    #añadir cjt en solucion (append)
                yield from backtracking(longSol+1,set().union(cjtAcumulado,cjt))    #yield from backtracking pasándole cjtAcumulado|cjt
                # donde | es el operador union
                solucion.pop(-1)    #quitar cjt de solucion (pop)
            # en cualquier caso probar a saltarse cjt
            yield from backtracking(longSol+1,cjtAcumulado) #yield from backtracking sin añadir cjt al cjtAcumulado
    yield from backtracking(0, set())   # empezamos con cjt vacío

if __name__ == "__main__":
    cjtdcjts = [{"casa","coche","gato"},
                {"casa","bici"},
                {"bici","perro"},
                {"boli","gato"},
                {"coche","gato","bici"},
                {"casa", "moto"},
                {"perro", "boli"},
                {"coche","moto"},
                {"casa"}]
    for solucion in exact_cover(cjtdcjts):
        print(solucion)

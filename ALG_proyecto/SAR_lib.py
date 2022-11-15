# coding=utf-8
import json
import math
from operator import itemgetter
from nltk.stem.snowball import SnowballStemmer
import os
import re

## Equipo SAR compuesto por:
## Daniil Antsyferov
## Diego Garcia
## Matthieu Cabrera
## Ricardo Carrascosa

class SAR_Project:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de noticias
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm + ranking de resultado

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [("title", True), ("date", False),
              ("keywords", True), ("article", True),
              ("summary", True)]
    
    
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.index = {
            'title': {},
            'date': {},
            'keywords':{},
            'article':{},
            'summary':{}
        } # hash para el indice invertido de terminos --> clave: termino, valor: posting list.
                        # Si se hace la implementacion multifield, se pude hacer un segundo nivel de hashing de tal forma que:
                        # self.index['title'] seria el indice invertido del campo 'title'.
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {
            'title': {},
            'date': {},
            'keywords': {},
            'article': {},
            'summary': {}
        } # hash para el indice permuterm.
        self.docs = {} # diccionario de documentos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados. puede no utilizarse
        self.news = {} # hash de noticias --> clave entero (newid), valor: la info necesaria para diferenciar la noticia dentro de su fichero (doc_id y posición dentro del documento)
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()
        self.newsNum = -1
        self.terminosSnippet = []


    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_snippet(self, v):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v


    def set_stemming(self, v):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v


    def set_ranking(self, v):
        """

        Cambia el modo de ranking por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON RANKING DE NOTICIAS

        si self.use_ranking es True las consultas se mostraran ordenadas, no aplicable a la opcion -C

        """
        self.use_ranking = v




    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################


    def index_dir(self, root, **args):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root" e indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """

        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        docID = 0
        for dir, subdirs, files in os.walk(root):
            for filename in files:
                if filename.endswith('.json'):
                    fullname = os.path.join(dir, filename)
                    docID += 1                        
                    self.docs[docID] = fullname            #Crea entrada en el diccionario de DOCUMENTOS
                    self.index_file(fullname,docID)
                    if (self.stemming) : 
                      self.make_stemming()
                    if (self.permuterm):
                        self.make_permuterm()

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        

    def index_file(self, filename, docID):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Indexa el contenido de un fichero.

        Para tokenizar la noticia se debe llamar a "self.tokenize"

        Dependiendo del valor de "self.multifield" y "self.positional" se debe ampliar el indexado.
        En estos casos, se recomienda crear nuevos metodos para hacer mas sencilla la implementacion

        input: "filename" es el nombre de un fichero en formato JSON Arrays (https://www.w3schools.com/js/js_json_arrays.asp).
                Una vez parseado con json.load tendremos una lista de diccionarios, cada diccionario se corresponde a una noticia

        """

        with open(filename) as fh:
            jlist = json.load(fh)

        #
        # "jlist" es una lista con tantos elementos como noticias hay en el fichero,
        # cada noticia es un diccionario con los campos:
        #      "title", "date", "keywords", "article", "summary"
        #
        # En la version basica solo se debe indexar el contenido "article"
        #
        #
        #
        #jlist es una lista de noticias/diccionarios

        ########################################################################
        ## 1. Iteramos sobre la lista de noticias                             ##
        ## 2. Tokenizamos la noticia                                          ##
        ## 3. Creamos/actualizamos la entrada en el diccionario de documentos ##
        ## 4. Creamos/actualizamos la entrada en el diccionario de tokens     ##
        ########################################################################
        newsID = -1
        for new in jlist:
          newsID += 1  
          self.newsNum += 1
          if docID not in self.news:
            self.news[docID] = []
          self.news[docID].append(newsID)
          for (field, shouldTokenize) in self.fields:
            if shouldTokenize:
              tokens = self.tokenize(new[field])
              for t in tokens:
                if t not in self.index.get(field,{}):
                  self.index[field][t] = [(docID,newsID)]
                elif (docID,newsID) not in self.index[field][t]:
                  self.index[field][t].append((docID,newsID))
                else:
                  continue #Si la tupla (docID,newsID) está en el diccionario, no hacemos nada
            else:
              dateToken = new[field]  
              if dateToken not in self.index.get(field,{}):
                self.index[field][dateToken] = [(docID,newsID)]
              elif (docID,newsID) not in self.index[field][dateToken]:
                self.index[field][dateToken].append((docID,newsID))      

    def tokenize(self, text):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()


    def make_stemming(self):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING.
        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.
        self.stemmer.stem(token) devuelve el stem del token
        """
        #we open the file
        docId = len(self.docs)
        newsID = -1
        with open(self.docs[docId]) as fh:
          jlist = json.load(fh)
        for new in jlist : 
          newsID += 1  
          self.newsNum += 1
          for (field, shouldTokenize) in self.fields:
            if shouldTokenize:
              tokens = self.tokenize(new[field])
              for t in tokens:
                #we retrieve the stemming of the current word and we fill the dictionary sindex
                stem = self.stemmer.stem(t)
                if stem not in self.sindex.get(stem, {}):
                  self.sindex[stem] = [t]
                elif t not in self.sindex[stem] : 
                  self.sindex[stem].append(t)
                else :
                  #the index exists already, we have nothing to do 
                  continue
              else:
                dateToken = new[field]  
                if dateToken not in self.sindex.get(dateToken,{}):
                  self.index[dateToken] = [dateToken]
        #print(self.sindex)
        pass

    
    def make_permuterm(self):
    
    
        
        # Si se activa la función multifield
        if self.multifield:
            multifield = ['title', 'date', 'keywords', 'article', 'summary']
        else:
            multifield = ['article']
        for field in multifield:
            # Se crea la lista de permuterms de un token
            # En este caso solo se guarda la noticia, no la posición
            for token in self.index[field]:
                token_p = token + '$'
                permuterm = []
                for _ in range(len(token_p)):
                    token_p = token_p[1:] + token_p[0]
                    permuterm += [token_p]

                for permut in permuterm:
                    if permut not in self.ptindex[field]:
                        self.ptindex[field][permut] = [token]
                    else:
                        if token not in self.ptindex[field][permut]:
                            self.ptindex[field][permut] += [token]


    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        print('==========================================')
        print('Number of indexed days: ', len(self.docs))
        print('------------------------------------------')
        print('Number of indexed news: ', self.newsNum)
        print('------------------------------------------')
        print('TOKENS:')
        if (self.permuterm):
            print('PERMUTERMS:')
            for field in self.ptindex.keys():
                 print("\t# of tokens in '{}': {}".format(field, len(self.ptindex[field])))
            print('----------------------------------------')
        length = 0
        for (field, b) in self.fields:
            length += len(self.index.get(field,{}))
        print('        # of tokens in ''article'': ', length)
        print('------------------------------------------')
        print('Positional queries are not allowed.')
        print('==========================================')


    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################
      
    def part_solve_query(self, query):
      '''
      Método auxiliar de solve_query
      param: query a resolver en formato lista en el que sus componentes serán strings y listas en caso de que hubiese habido paréntesis
      output: lista con las tuplas 
      '''

      ## Al ser notación infija necesitamos ambos lados del operador
      ## 1. Si no es un operador:
      ## 1.1. Si es una lista, ya viene resuelta y solo la copiamos
      ## 1.2. Si es una cadena, es un término que tenemos que extraer del índice
      ## 1.2.1. En caso de que el término anterior fuese un NOT, ejecutamos el método reverse_posting
      ## 1.3. Una vez tenemos dos operandos y un operador, llamamos al método and/or_posting
      ## 2. Si es un operador:
      ## 2.1. Almacenamos el tipo para hacer la operación cuando tengamos dos operandos (uno en caso de operador unario)
      pl1 = pl2 = []
      oper = ''
      operNot = False
      print("query in method: " + str(query))
      for t in query:
        if t not in ['AND','OR','NOT']:
          if len(pl1) == 0:
            if isinstance(t, list):
              pl1 = t
            elif ':' in t:
              pair = t.split(':')
              pl1 = self.get_posting(pair[1].lower(),pair[0])
            else:
              pl1 = self.get_posting(t.lower())
            if operNot:
              pl1 = self.reverse_posting(pl1)
              operNot = False
          else:
            if isinstance(t, list):
              pl2 = t
            elif ':' in t:
              pair = t.split(':')
              pl2 = self.get_posting(pair[1].lower(),pair[0])
            else:
              pl2 = self.get_posting(t.lower())
            if operNot:
              pl2 = self.reverse_posting(pl2)
              operNot = False
            if oper == 'AND':
              pl1 = self.and_posting(pl1, pl2)
            elif oper == 'OR':
              pl1 = self.or_posting(pl1, pl2)
        elif t == 'NOT':
          operNot = True
        else:
          oper = t
      return pl1

    def solve_query(self, query, prev={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen

        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.
        return: posting list con el resultado de la query
        """
        #############################################################################
        ##                           PROCESO DEL METODO                            ##
        ## 1. Se divide la cadena en divisiones por blanco                         ##
        ## 2. Se buscan paréntesis y se crea una nueva lista                       ##
        ## 3. Se resuelven los paréntesis primero y después de izquierda a derecha ##
        #############################################################################
      
        if query is None or len(query) == 0:
            return []

        ## 1. Split de la cadena
        query_parse = query.split()

        ## 2. Búsqueda de paréntesis
        ##    Si la cadena es alfanum se añade a la lista auxiliar y si es el final se termina
        ##    Si no es alfanum el primer carácter o el último, se separa del término y se añaden a la lista
        query_list = []
        i = 0
        term = query_parse[i]
        eof = False
        while not eof:
          if term.isalnum() or ':' or '*' or '?' in term:
            query_list.append(term) 
            if i < len(query_parse) - 1:
              i += 1
              term = query_parse[i]
            else:
              eof = True
          elif not term[0].isalnum():
            query_list.append(term[0])
            term = term[1:]
          elif not term[-1].isalnum():
            j = term.find(r')')
            query_list.append(term[0:j])
            query_list = query_list + list(term[j:])
            if i < len(query_parse) - 1:
              i += 1
              term = query_parse[i]
            else:
              eof = True
          
        ## 3. Se resuelven los paréntesis interiores primero
        ##    Buscamos el paréntesis más interior y se manda al método auxiliar para su resolución
        ##    Se añade el resultado a la lista y se eliminan los componentes de la sub-query, incluidos paréntesis, por slice
        ##    Cuando se resuelve un paréntesis, se busca desde el principo de nuevo hasta que no encuentra ninguno más
        ##    Ya sin paréntesis, se manda al método auxiliar para su resolución
        eof = False
        i = 0
        res = []
        while not eof:
          #Buscamos un (
          if i < len(query_list) and query_list[i] == '(':
            j = i + 1
            #Buscamos un )
            while query_list[j] != ')':
              #Si encontramos otro (, pasamos a buscar su cierre
              if query_list[j] == '(':
                i = j
                break
              else:
                j += 1  
            #cuando encontramos un par () (que está en las posiciones i y j)
            #extraemos la subconsulta y la pasamos al método que la resuelve 
            #para posteriormente eliminar lo que ya se ha resuelto
            else:
              partial_query = query_list[i+1:j]
              res = self.part_solve_query(partial_query)
              query_list.insert(i, res)
              query_list = query_list[:i+1] + query_list[j+2:]
              i = 0
          elif i < len(query_list):
            i += 1
          else:
            eof = True
        
        return self.part_solve_query(query_list)
  
    def get_posting(self, term, field='article'):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        ## Se devuelve la posting list del témino proporcionado o la lista vacia en caso de no existir
        self.terminosSnippet.append(term)
        res = []
        if (self.stemming) :
          listRes = self.get_stemming(term, field)
          for it in listRes : 
            res.extend(self.index.get(field).get(it,[]))
        elif '*' in term or '?' in term:
            print('* get_posting')
            res = self.get_permuterm(term,field)
        else : 
          res = self.index.get(field).get(term,[])
        return res



    def get_positionals(self, terms, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        pass
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################


    def get_stemming(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING

        Devuelve la posting list asociada al stem de un termino.

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        stem = self.stemmer.stem(term)
        if stem in self.sindex.get(stem, {}):
          field = self.sindex[stem]
        return field
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################


    def get_permuterm(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        term = term + '$'
        res = []
        
        #recorremos term para crear la query 
        while term[-1] != '*' and term[-1] != '?': 
            term = term[1:] + term[0]
        #guardamos si alguno de los carácteres es comodin
        com = term[-1]
        #guardamos el termino de busqueda
        term = term[:-1]

        #Si com = '*' : buscamos TODOS sus permutaciones que comiencen por query
        #Si com = '?': buscamos TODAS sus permutaciones que comiencen por query + longitud = al de term
        for perm in (x for x in list(self.ptindex[field].keys()) if x.startswith(term) and (com == '*' or len(x) == len(term) + 1)):
            #Buscamos para cada token en ptindex
            for token in self.ptindex[field][perm]:
                res = self.or_posting(res, self.get_posting(token, field))

        return res
 




    def reverse_posting(self, p):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los newid exceptos los contenidos en p
        """
        ## Se hace una lista con todos los pares (doc,new) y luego se eliminan los que se pasan como argumento
        listaNot = [(i,x) for i in self.news for x in self.news[i]]
        res = [x for x in listaNot if x not in p]
      
        return res



    def and_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos en p1 y p2

        """
        #Implementación usando comprehensive lists
        return [x for x in p1 if x in p2]
      
        ## Implementación del algoritmo visto en clase
        res = []
        i = j = 0
        while (i < len(p1)) & (j < len(p2)):
            if p1[i] == p2[j]:
                res.append(p2[j])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1

        return res


    def or_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos de p1 o p2

        """
        #Implementación utilizando comprhensive lists
        return p1 + [x for x in p2 if x not in p1 ]
      
        ## Implementación del algoritmo visto en clase
        res = []
        i = j = 0
        while (i < len(p1)) & (j < len(p2)):
            if p1[i] == p2[j]:
                res.append(p2[j])
                i +=  1
                j +=  1
            elif p1[i] < p2[j]:
                res.append(p1[i])
                i += 1
            else:
                res.append(p2[j])
                j += 1
        
        while i < len(p1):
            res.append(p1[i])
            i += 1

        while j < len(p2):
            res.append(p2[j])
            j += 1

        return res


    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se propone por si os es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos de p1 y no en p2

        """
        res = [x for x in p1 if x not in p2]
        return res


    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################


    def solve_and_count(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T

        """
        result = self.solve_query(query)
        print("Query: " + query + "\n")
        print("Number of results: " + str(len(result)) + "\n")
        return len(result)  # para verificar los resultados (op: -T)


    def solve_and_show(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra informacion de las noticias recuperadas.
        Consideraciones:

        - En funcion del valor de "self.show_snippet" se mostrara una informacion u otra.
        - Si se implementa la opcion de ranking y en funcion del valor de self.use_ranking debera llamar a self.rank_result

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T
        
        """
        result = self.solve_query(query)
        if self.use_ranking:
            result = self.rank_result(result, query)   

        #mostarar consulta y numero de resultados
        print("========================\n")    
        print("Query: " + query + "\n")
        if (result):
          print("Number of results: " + str(len(result)) + "\n")
          i=0
          if (not self.show_all):
            result = result[:10]
          for (docId, newId) in result:
              filename = self.docs[docId]
              with open(filename) as fh:
                  jlist = json.load(fh)
              i+=1
              #mostarar informacion para cada articulo
              print("#" + str(i) + "\n")
              print("Score: 0\n")
              print(str(newId) + "\n")  
              print("Date: " + jlist[newId]["date"] + "\n")
              print("Title: " + jlist[newId]["title"] + "\n")
              print("Keywords: " + jlist[newId]["keywords"] + "\n")
              if self.show_snippet:
                  # sacar snippet sigiendo la primera technica propuesta
                  # enter todos terminos de consulta calcular min index y max index
                  # snippet es = articulo[min:max]
                  min = -1
                  max = -1
                  terms = self.tokenizer.sub(' ', jlist[newId]["article"].lower()).split()
                  for term in self.terminosSnippet:
                      first = terms.index(term)
                      last = terms[::-1].index(term)
                      if min < 0 or first < min:
                          min = first
                      if max < 0 or last > max:
                          max = last
                  print("Snippet: " + jlist[newId]["article"][min:max] + "\n")
          return len(result)
        else :
          print("Number of results: 0 \n")
          return 0




    def rank_result(self, result, query):
        """
        NECESARIO PARA LA AMPLIACION DE RANKING

        Ordena los resultados de una query.

        param:  "result": lista de resultados sin ordenar
                "query": query, puede ser la query original, la query procesada o una lista de terminos


        return: la lista de resultados ordenada

        """
        temp = {}
        #result : list of the result
        for iterator in result : 
          #we retrieve the occurrence of the query in each element of result
          f_td = iterator[1]
          if f_td > 0 :
            tf_td = 1 + math.log10(f_td)
          else : 
            tf_td = 0
          temp[iterator[0]] = tf_td
        temp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
        listIndex = []
        for iterator in temp : 
          listIndex.append(iterator[0])
        listRes = []
        for l in listIndex:
          for index, value in result : 
            if (index == l) : 
              listRes.append((index, value))
        #print(listRes)
        return listRes
          
        #pass
        
        ###################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE RANKING ##
        ###################################################

# -*- coding: utf-8 -*-
import re
import distancias
#import numpy as np


class SpellSuggester:

    """
    Clase que implementa el método suggest para la búsqueda de términos.
    """

    def __init__(self,
                 dist_functions,
                 vocab = [],
                 default_distance = None,
                 default_threshold = None):
        
        """Método constructor de la clase SpellSuggester

        Construye una lista de términos únicos (vocabulario),

        Args:
           dist_functions es un diccionario nombre->funcion_distancia
           vocab es una lista de palabras o la ruta de un fichero
           default_distance debe ser una clave de dist_functions
           default_threshold un entero positivo

        """
        self.distance_functions = dist_functions
        self.set_vocabulary(vocab)
        if default_distance is None:
            default_distance = 'levenstein'
        if default_threshold is None:
            default_threshold = 3
        self.default_distance = default_distance
        self.default_threshold = default_threshold

    def build_vocabulary(self, vocab_file_path):
        """Método auxiliar para crear el vocabulario.

        Se tokeniza por palabras el fichero de texto,
        se eliminan palabras duplicadas y se ordena
        lexicográficamente.

        Args:
            vocab_file (str): ruta del fichero de texto para cargar el vocabulario.
            tokenizer (re.Pattern): expresión regular para la tokenización.
        """
        tokenizer=re.compile("\W+")
        with open(vocab_file_path, "r", encoding="utf-8") as fr:
            vocab = set(tokenizer.split(fr.read().lower()))
            vocab.discard("")  # por si acaso
            return sorted(vocab)

    def set_vocabulary(self, vocabulary):
        if isinstance(vocabulary,list):
            self.vocabulary = vocabulary # atención! nos quedamos una referencia, a tener en cuenta
        elif isinstance(vocabulary,str):
            self.vocabulary = self.build_vocabulary(vocabulary)
        else:
            raise Exception("SpellSuggester incorrect vocabulary value")

    def suggest(self, term, distance=None, threshold=None, flatten=True):
        """

        Args:
            term (str): término de búsqueda.
            distance (str): nombre del algoritmo de búsqueda a utilizar
            threshold (int): threshold para limitar la búsqueda
        """
        if distance is None:
            distance = self.default_distance
        if threshold is None:
            threshold = self.default_threshold

        ########################################
        # COMPLETAR
        ########################################

        resul =[]
        newresul =[]
        #mismos nombres que en resultado_test_spellsuggester
        #nombres cogidos de levensthein_m
        for palabra in self.vocabulary: #devuelve una lista de lista de palabras
            #la lista i-ésima contiene las palabras a distancia i 
            # (para i hasta threshold incluido)
            if distance=="levenshtein_m":
                if distancias.levenshtein_matriz(term,palabra,threshold)==threshold:
                    newresul .append(palabra)
            if distance=="levenshtein_r":
                if distancias.levenshtein_reduccion(term,palabra,threshold)==threshold:
                    newresul .append(palabra)
            if distance=="levenshtein":
                if distancias.levenshtein(term,palabra,threshold)==threshold:
                    newresul .append(palabra)
            if distance=="levenshtein_o":
                if distancias.levenshtein_cota_optimista(term,palabra,threshold)==threshold:
                    newresul .append(palabra)
            if distance=="damerau_rm":
                if distancias.damerau_restricted_matriz(term,palabra,threshold)==threshold:
                    newresul .append(palabra)
            if distance=="damerau_r":
                if distancias.damerau_restricted(term,palabra,threshold)==threshold:
                    newresul .append(palabra)
            if distance=="damerau_im":
                if distancias.damerau_intermediate_matriz(term,palabra,threshold)==threshold:
                    newresul .append(palabra)
            if distance=="damerau_i":
                if distancias.damerau_intermediate(term,palabra,threshold)==threshold:
                    newresul .append(palabra)
            
        if flatten: #si flatten=True (opcion por defecto) --> devuelve lista
            newresul  = [word for wlist in resul  for word in wlist]
        #devuelve un diccionario donde para cada distancia de edición tiene ligada 
        #la lista de palabras que estan a esa distancia
        return newresul 


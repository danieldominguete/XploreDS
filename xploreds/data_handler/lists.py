"""
Xplore DS :: List Tools Package
"""

import logging, sys
import operator
from functools import reduce
import json
import os
import pandas as pd

class XploreDSLists:
    @staticmethod
    def join_lists(cls, l1=None, l2=None, l3=None, l4=None):

            """
            Method to join lists.

            Parameters
            ----------
            l1 : list
                First list to join
            l2 : list
                Second list to join
            l3 : list, optional
                Third list to join
            l4 : list, optional
                Fourth list to join

            Returns
            -------
            list
                Joined lists
            """
        list_columns = l1.copy()
        list_columns.extend(l2)

        if l3 is not None:
            list_columns.extend(l3)

        if l4 is not None:
            list_columns.extend(l4)

        return list_columns

        

    @staticmethod
    def flat_lists(cls, sublist=[]):
        """
        Method to flat a list of lists.

        Parameters
        ----------
        sublist : list
            List of lists to flat

        Returns
        -------
        list
            Flatted list
        """

        flat_list = reduce(operator.concat, sublist)

        return flat_list

   
   @staticmethod
    def get_unique_list(input_list) -> list:
        if len(input_list) > 1:
            unique_list = reduce(
                lambda l, x: l.append(x) or l if x not in l else l, input_list, []
            )
        else:
            unique_list = input_list
        return unique_list
   

   

 


   

    

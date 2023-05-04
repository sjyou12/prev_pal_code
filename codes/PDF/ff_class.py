import math
from codes.PDF import ffcoef  # form factor coefficient table
import copy
import numpy as np

class Atom:
    def __init__(self, type):
        # assign index(order of atom) 0에서 부터 시작
        # assign type of atom
        # if type == "O":
        #     # type = "O"
        index_in_e_list = ffcoef.element_list.index(type)
        self.type = ffcoef.atom_type_list[index_in_e_list]
        # elif type == "H":
        #     pass
        # if type in ffcoef.element_list:  # ElementList에 type이 들어있어야함
        #     index_in_e_list = ffcoef.element_list.index(type)
        #     self.type = ffcoef.atom_type_list[index_in_e_list]
        # else:
        #     raise NameError("Unknown Atom type")
        # self.pos = Pos3D
        # self.posArr = [Pos3D.x, Pos3D.y, Pos3D.z]

    @staticmethod
    def form_factor(Atom, q):
        sum_of_ff = 0
        s = q / (4 * math.pi)
        for m in range(0, 5):
            sum_of_ff += Atom.type.a_list[m] * (np.exp((-(Atom.type.b_list[m] * s * s)))) #교수님 code의 f랑 같음.
        sum_of_ff += Atom.type.c

        # remove normalize function
        '''
        if sum_of_ff > self.maxFF[Atom.a_index]:
            self.maxFF[Atom.a_index] = sum_of_ff
        sum_of_ff = sum_of_ff / self.maxFF[Atom.a_index] #normalize
        '''
        return sum_of_ff
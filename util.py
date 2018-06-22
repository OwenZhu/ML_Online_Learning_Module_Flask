# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 下午8:58
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : util.py
# @Software: PyCharm Community Edition

import random


def convert_word_to_embedding_index(word, voc):
    if word in voc:
        return voc[word]
    else:
        return random.randint(0, len(voc) - 1)

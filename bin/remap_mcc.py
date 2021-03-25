#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:33:51 2021

@author: matt
"""

def maprange( a, b, s):
	(a1, a2), (b1, b2) = a, b
	return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))


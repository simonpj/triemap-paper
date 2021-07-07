#!/usr/bin/env python3
#
# Copied and modified from crit_csv_to_latex.py, so better copy from there if you want something like this.

import csv
import datetime
import math
import sys
from decimal import Decimal
from collections import defaultdict


def quantize_n_digits(n: int, val: Decimal):
  """
  quantize_n_digits(3, 00.0054499) ==> 0.00545
  quantize_n_digits(3, 54.9934455) ==> 55.0
  quantize_n_digits(3, 99.9934455) ==> 100
  quantize_n_digits(3, 9999.34455) ==> 10000
  """
  shift_amount = val.adjusted() - (n-1)
  shifted = val / Decimal(10)**shift_amount
  rounded = Decimal(round(shifted))
  if rounded.adjusted() > shifted.adjusted(): # The 99.999 case
    assert(rounded.adjusted() == shifted.adjusted() + 1)
    shift_amount = shift_amount+1
    rounded = Decimal(round(shifted/10))
  return rounded * Decimal(10)**shift_amount

n_digits=3

def mark_digit_insignificant(dig):
  return '\insigdig{'+dig+'}'

def mark_insignificant_digits(val: str, significant_digs: int):
  """
  mark_insignificant_digits("2.43", 0) ==> (2).(4)(3)
  mark_insignificant_digits("2.43", 1) ==> 2.(4)(3)
  mark_insignificant_digits("2.43", 2) ==> 2.4(3)
  mark_insignificant_digits("2.43", 3) ==> 2.43
  mark_insignificant_digits("2.43", 4) ==> 2.43
  """
  rev_s = val[::-1]
  s = ''
  insignificant_digs = max(0, n_digits - significant_digs)
  for i, c in enumerate(rev_s):
    if c == '.':
      s = c+s
      continue
    assert c.isdigit(), c
    if insignificant_digs > 0:
      insignificant_digs = insignificant_digs-1
      s = mark_digit_insignificant(c)+s
    else:
      s = c+s
  return s

sis = ['B', 'KB', 'MB', 'GB']

def format_absolute(mean: Decimal):
  mean_qz = quantize_n_digits(n_digits, mean)
  si_idx = math.floor(math.log(mean_qz, 2) / 10)
  mag_si = si_idx * 10
  mean_n_digs = mean_qz/(2**mag_si)

  s = f"{mean_n_digs:.{n_digits}g}"
  return s+sis[si_idx]

def format_relative(baseline: Decimal, mean: Decimal):
  size_factor = quantize_n_digits(n_digits, mean / baseline)
  return f"{size_factor:.{n_digits-1}f}"

def csv_to_cleaned_table(filename):
  table = defaultdict(lambda: defaultdict(lambda: {}))
  with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      descr = row['Name']
      size = Decimal(row['Size'])

      parts=descr.split('/')
      (name, map_size, expr_size, data_structure) = parts
      map_size=int(map_size)
      expr_size=int(expr_size)

      entry = ''
      if data_structure == 'ExprMap':
        # baseline
        entry = format_absolute(size)
      else:
        (baseline, _) = table[name][(map_size, expr_size)]['ExprMap']
        entry = format_relative(baseline, size)

      # print (name, map_size, expr_size, data_structure)
      # print (size, entry)
      table[name][(map_size, expr_size)][data_structure] = (size, entry)
    return table

table = csv_to_cleaned_table(sys.argv[1])

variants=['ExprMap','Map','HashMap']

# Format E x M table for space
space_names=['space', 'space_app1', 'space_app2', 'space_lam']
for name in space_names:
  s = ''
  inputs=table[name]
  for expr_size in [10,100,1000,10000]:
    s = s + '& \\textbf{'+str(expr_size)+'} & '
    for map_size in [10,100,1000,10000]:
      measurements = inputs[(map_size, expr_size)]
      winner=''
      if measurements:
        (winner,_) = min([(v, measurements[v][0]) for v in variants if v in measurements], key=lambda p: p[1])
      print(measurements)
      print(winner)
      for v in variants:
        #print(name,size,v)
        entry = measurements[v][1] if v in measurements else '\\dag'
        if v == winner:
          s = s+'\\textbf{'+entry+'} & '
        else:
          s = s+entry+' & '
    # the weird slice is to strip the trailing '& '
    s = s[:-2]+'\\\\\n'

  print(s)
  with open(f'../paper/bench-{name}.tex-incl', 'w') as f: f.write(s)



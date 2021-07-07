#!/usr/bin/env python3

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

sis = ['ps', 'ns', '\\textmu{}s', 'ms', 's']

def format_absolute(mean: Decimal, confidence: Decimal):
  mean_qz = quantize_n_digits(n_digits, mean)
  mag_ps = mean_qz.adjusted() + 12 # for picoseconds
  si_idx = math.floor(mag_ps / 3)
  mag_si = si_idx * 3
  mean_n_digs = mean_qz*(10**(12-mag_si))

  # Encode confidence in string s
  # ex.: mean_qz = 55.3, confidence = 0.6, n_digits = 3
  # mag_delta = 2 ==> insignificant_digs = 1
  s = mark_insignificant_digits(f"{mean_n_digs:.{n_digits}g}", mean_qz.adjusted() - confidence.adjusted())
  return s+sis[si_idx]

def format_relative(baseline: Decimal, mean: Decimal, confidence: Decimal):
  speedup = quantize_n_digits(n_digits, baseline / mean)
  s = mark_insignificant_digits(f"{speedup:.{n_digits-1}f}", mean.adjusted() - confidence.adjusted())
  return s

def csv_to_cleaned_table(filename):
  table = defaultdict(lambda: defaultdict(lambda: {}))
  with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      descr = row['Name']
      mean = Decimal(row['Mean'])
      stddev2 = 2*Decimal(row['Stddev'])

      parts=descr.split('/')
      name=''
      map_size=''
      expr_size=''
      data_structure=''
      if len(parts) == 5:
        (name, map_size, expr_size, data_structure, _) = parts
        map_size=int(map_size)
        expr_size=int(expr_size)
      elif len(parts) == 4:
        (name, size, data_structure, _) = parts
        expr_size = map_size = int(size)
      else:
        assert False, len(parts)

      entry = ''
      if data_structure == 'ExprMap':
        # baseline
        entry = format_absolute(mean, stddev2)
      else:
        (baseline, _) = table[name][(map_size, expr_size)]['ExprMap']
        entry = format_relative(baseline, mean, stddev2)

      # print (name, map_size, expr_size, data_structure)
      # print (mean, entry)
      table[name][(map_size, expr_size)][data_structure] = (mean, entry)
    return table

table = csv_to_cleaned_table(sys.argv[1])
#print(table['lookup_all'][(100, 100)])

variants=['ExprMap','Map','HashMap']
# Format overview table for all benchmarks
s = ''
overview_names=['lookup_all','lookup_all_app1','lookup_all_app2','lookup_all_lam','lookup_one','insert_lookup_one','fromList','fromList_app1','fold']
for name in overview_names:
  inputs=table[name]
  s = s + '\\benchname{'+name.replace('_','\\_')+'} & '
  for size in [10,100,1000]:
    measurements = inputs[(size, size)]
    (winner,_) = min([(v, measurements[v][0]) for v in variants], key=lambda p: p[1])
    for v in variants:
      #print(name,size,v)
      (_, entry) = measurements[v]
      if v == winner:
        s = s+'\\textbf{'+entry+'} & '
      else:
        s = s+entry+' & '
  # the weird slice is to strip the trailing '& '
  s = s[:-2]+'\\\\\n'

print(s)
with open('../paper/bench-overview.tex-incl', 'w') as f: f.write(s)

# Format E x M table for lookup_all
s = ''
overview_names=['lookup_all']
for name in overview_names:
  inputs=table[name]
  for expr_size in [10,100,1000,10000]:
    s = s + '& \\textbf{'+str(expr_size)+'} & '
    for map_size in [10,100,1000,10000]:
      measurements = inputs[(map_size, expr_size)]
      (winner,_) = min([(v, measurements[v][0]) for v in variants], key=lambda p: p[1])
      for v in variants:
        #print(name,size,v)
        (_, entry) = measurements[v]
        if v == winner:
          s = s+'\\textbf{'+entry+'} & '
        else:
          s = s+entry+' & '
    # the weird slice is to strip the trailing '& '
    s = s[:-2]+'\\\\\n'

print(s)
with open('../paper/bench-lookup.tex-incl', 'w') as f: f.write(s)

# Format E x M table for insert_lookup_one
s = ''
overview_names=['insert_lookup_one']
for name in overview_names:
  inputs=table[name]
  for expr_size in [10,100,1000,10000]:
    s = s + '& \\textbf{'+str(expr_size)+'} & '
    for map_size in [10,100,1000,10000]:
      measurements = inputs[(map_size, expr_size)]
      (winner,_) = min([(v, measurements[v][0]) for v in variants], key=lambda p: p[1])
      for v in variants:
        #print(name,size,v)
        (_, entry) = measurements[v]
        if v == winner:
          s = s+'\\textbf{'+entry+'} & '
        else:
          s = s+entry+' & '
    # the weird slice is to strip the trailing '& '
    s = s[:-2]+'\\\\\n'

print(s)
with open('../paper/bench-insert.tex-incl', 'w') as f: f.write(s)



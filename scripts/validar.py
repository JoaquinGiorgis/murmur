"""Valida que entrega/ cumpla el formato oficial contra los datos de entrada.
Uso: python scripts/validar.py datos/final entrega
"""
import sys, pandas as pd
from datetime import datetime
datos, ent = sys.argv[1], sys.argv[2]
r = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
ci, cu = r(f"{datos}/comentarios.csv.gz"), r(f"{datos}/cuentas.csv.gz")
co, cs, na = r(f"{ent}/comentarios.csv"), r(f"{ent}/cuentas.csv"), r(f"{ent}/narrativas.csv")
def chk(n, ok, det=""): print(f"{'OK   ' if ok else 'FALLA'} {n} {det}")
chk("comentarios: mismos ids", set(co.comentario_id)==set(ci.comentario_id), f"({len(co)} filas, entrada {ci.comentario_id.nunique()})")
chk("comentarios: sin duplicados", not co.comentario_id.duplicated().any())
chk("cuentas: mismos ids", set(cs.cuenta_id)==set(cu.cuenta_id), f"({len(cs)} filas, entrada {cu.cuenta_id.nunique()})")
chk("cuentas: sin duplicados", not cs.cuenta_id.duplicated().any())
ning = co.narrativa.str.lower().isin(["ninguna",""])
chk("postura vacía si narrativa=ninguna", (co.postura[ning]=="").all(), f"({ning.sum()} ninguna, {(co.postura[ning]!='').sum()} con postura)")
chk("postura válida si hay narrativa", co.postura[~ning].isin(["apoya","rechaza"]).all(), str(co.postura[~ning].value_counts().to_dict()))
tz = [datetime.fromisoformat(x).tzinfo is not None if x else False for x in na.fecha_origen]
chk("fecha_origen con zona horaria", all(tz), f"({sum(tz)}/{len(na)}) ej: {na.fecha_origen.iloc[0] if len(na) else '(sin narrativas)'}")
chk("narrativas de comentarios existen en narrativas.csv", set(co.narrativa[~ning]) <= set(na.narrativa))

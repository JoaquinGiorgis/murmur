#!/usr/bin/env bash
# Corrida sobre el set final. Uso:  bash scripts/viernes.sh sin-ia   |   bash scripts/viernes.sh con-ia
# No hace commit ni push: valida la entrega e imprime los comandos para correrlos después de mirarla.
set -euo pipefail
MODO=${1:?"sin-ia o con-ia"}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO"
source .venv/bin/activate
test -f datos/final/comentarios.csv.gz || { echo "FALTA datos/final/comentarios.csv.gz"; exit 1; }

# caffeinate (macOS): la máquina no se duerme mientras corre
CAF=""; command -v caffeinate >/dev/null && CAF="caffeinate -i"
RC=0
if [ "$MODO" = "sin-ia" ]; then
  time $CAF env -u ANTHROPIC_API_KEY HF_HUB_OFFLINE=1 python -m murmur --datos datos/final --entrega entrega --sin-ia || RC=$?
else
  : "${ANTHROPIC_API_KEY:?exportá ANTHROPIC_API_KEY antes}"
  time $CAF env HF_HUB_OFFLINE=1 python -m murmur --datos datos/final --entrega entrega || RC=$?
fi
# 0 = todo bien; 2 = entrega completa pero una etapa falló (ver ATENCIÓN en el log); otro = se cayó
[ "$RC" = 0 ] || [ "$RC" = 2 ] || { echo "murmur terminó con código $RC: NO hay entrega válida"; exit 1; }
[ "$RC" = 2 ] && echo "ATENCIÓN: entrega con una etapa fallida (ver arriba). Igual se valida."

echo; echo "== validación"
python scripts/validar.py datos/final entrega | tee /dev/stderr | grep -q FALLA && { echo "HAY FALLAS: no pushear"; exit 1; }
echo; echo "== claves: no debe haber ninguna en entrega/"
if grep -rqE "sk-ant-|ANTHROPIC_API_KEY" entrega; then echo "HAY UNA CLAVE EN entrega/: no pushear"; exit 1; else echo "OK   sin claves"; fi
echo; echo "== archivos en entrega/ (deben ser 4 + .gitkeep)"; ls -la entrega
echo; echo "== costo.json"; cat entrega/costo.json
echo; echo "Si todo está OK:"
echo "  git add entrega && git status --short && git commit -m 'Entrega set final ($MODO)' && git push"

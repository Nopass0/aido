#!/usr/bin/env sh
# Needle-in-a-haystack test for long-context KV modes.
#   sh scripts/needle.sh models/Llama-3.2-1B-Instruct-Q4_K_M.gguf [chars]
MODEL=${1:-models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}
CHARS=${2:-60000}
python - "$CHARS" <<'PY'
import sys
t = open('data/alice_en.txt', encoding='utf-8', errors='ignore').read()
n = int(sys.argv[1]); h = (t * (n // len(t) + 1))[:n]
pos = int(len(h) * 0.15)
doc = h[:pos] + ' The secret code for the vault is BLUE-FALCON-4721. ' + h[pos:]
q = 'Below is a long document. Read it and answer the question at the end.\n\n<document>\n' + doc + '\n</document>\n\nQuestion: What is the secret code for the vault? Answer with the code only.'
open('needle_prompt.txt', 'w', encoding='utf-8').write(q)
PY
for cfg in "--kv f16 --kv-budget 0" "--kv q8 --kv-budget 0" "--kv q8 --kv-budget 4096 --kv-recent 1024" "--kv q4 --kv-budget 4096 --kv-recent 1024"; do
  echo "== $cfg"
  aido run -m "$MODEL" -p @needle_prompt.txt -n 24 --ctx $((CHARS / 3)) --spec off $cfg 2>&1 | tail -3
done
rm -f needle_prompt.txt

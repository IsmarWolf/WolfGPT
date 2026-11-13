"""WolfGPT Phase 2 - step 1: train our own BPE tokenizer on FineWeb-Edu.

BPE (byte-pair encoding): start from single bytes, repeatedly merge the most frequent
neighbouring pair into a new token. Common words/pieces become ONE token; rare words are
split into pieces. Byte-level means ANY text can be encoded (no unknown characters).
Output: data_fw/tokenizer.json
"""
import sys, time
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from common import *

VOCAB = int(sys.argv[1]) if len(sys.argv) > 1 else 32768

t0 = time.time()
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOT], show_progress=True,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
print(f"streaming {TOK_DOCS:,} docs to train a {VOCAB}-token vocabulary...")
tok.train_from_iterator(doc_stream(0, TOK_DOCS), trainer=trainer, length=TOK_DOCS)
tok.save(f"{OUT}/tokenizer.json")
print(f"trained in {time.time()-t0:.0f}s, vocab size {tok.get_vocab_size()}")

# --- sanity checks on documents the tokenizer never saw -----------------------------------
test = list(doc_stream(TOK_DOCS, 500))
chars = sum(map(len, test))
ntok = sum(len(e.ids) for e in tok.encode_batch(test))
print(f"characters per token on unseen text: {chars/ntok:.2f}   (char-level model = 1.00)")
s = "Brazilian Portuguese is spoken in São Paulo. Tokenization works!"
e = tok.encode(s)
assert tok.decode(e.ids) == s, "round-trip failed"
print("round trip OK:", e.tokens)

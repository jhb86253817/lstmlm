import json

last_ppl = 603
best_ppl = 603
lr_current = 0.1
pre_params = (last_ppl, best_ppl, lr_current)
with open('../model/pre_params.json', 'wb') as f:
    f.write(json.dumps(pre_params))

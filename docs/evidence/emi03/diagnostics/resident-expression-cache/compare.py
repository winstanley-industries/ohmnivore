from pathlib import Path
import sys,json
from reference.emi01 import circuits,metrics,signals
from reference.emi03 import ensemble
p=Path(sys.argv[1]);tables={};records={}
for lane in ('cpu','gpu'):
 table=signals.parse_raw((p/(lane+'.raw')).read_bytes(),circuits.DPT_NAMES,16e-6,2e-9)
 tables[lane]=table
 records[lane]={'fixture':'dpt','id':lane,'sample_step_s':1e-9,'metrics':metrics.dpt(table,1e-9)}
 records[lane]['status']='qualified' if records[lane]['metrics']['pass'] else 'accuracy_failure'
result=ensemble.compare_pair(tables['gpu'],records['gpu'],tables['cpu'],records['cpu'])
(p/'comparison.json').write_text(json.dumps({'scope':'full q0 DPT differential only; not complete qualification','comparison':result,'metrics':records},indent=2)+'\n')
print(json.dumps(result,indent=2))

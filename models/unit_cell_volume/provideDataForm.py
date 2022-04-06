#!/usr/bin/env python3
# coding: utf-8
import pandas as pd

def main(paramDic):
	MaterialMappingDf   = pd.read_csv(paramDic['MaterialMappingPath'],sep='\t',compression='gzip').set_index('matSet')
	ICSDParamsDf        = pd.DataFrame({'icsd-id': MaterialMappingDf['icsd-id'].values, 'volume_ICSD':None, 'TInK_ICSD':None}).set_index('icsd-id')
	ICSDParamsDf.to_csv(paramDic['ICSDParamsDfPath'], index=True, header=True, sep='\t')

if __name__ == '__main__':
	main({
		'MaterialMappingPath': 'features/materialsMapping.csv.gz',  
		'ICSDParamsDfPath':    'external/ICSDParams.csv',
	})


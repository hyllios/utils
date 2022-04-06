#!/usr/bin/env python3
# coding: utf-8
import sys, json, bz2
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
import pymatgen as pm
import numpy as np
import pandas as pd

def latticeDicOf(DBLbl,struc):
	anglesInRad = np.array(struc.lattice.angles)*(np.pi/180.)
	return {
		'volume_'+DBLbl:       struc.volume,
		'latticeA_'+DBLbl:     struc.lattice.abc[0],
		'latticeB_'+DBLbl:     struc.lattice.abc[1],
		'latticeC_'+DBLbl:     struc.lattice.abc[2],
		'latticeAlpha_'+DBLbl: anglesInRad[0],
		'latticeBeta_'+DBLbl:  anglesInRad[1],
		'latticeGamma_'+DBLbl: anglesInRad[2]
	}

def main(paramDic):
	# readin DB-mapping and lattice parameters
	MaterialMappingDf   = pd.read_csv(paramDic['MaterialMappingPath'],sep='\t',compression='gzip').set_index('matSet')
	PBEsolParamsDf      = pd.read_csv(paramDic['PBEsolParamsDfPath'], sep='\t',compression='gzip').set_index('matSet')
	ICSDParamsDf        = pd.read_csv(paramDic['ICSDParamsDfPath'],sep='\t').set_index('icsd-id')

	# compile composition features
	featureCalcs        = MultipleFeaturizer([
		cf.Stoichiometry(), cf.ElementProperty.from_preset("matminer"), cf.IonProperty(fast=True)
	])
	compFeatureDic      = lambda redformula: dict(zip(
		featureCalcs.feature_labels(), featureCalcs.featurize(pm.Composition(redformula))
	))
	compSubFeatureLst = [
	'0-norm','10-norm','2-norm','3-norm','5-norm','7-norm','PymatgenData maximum atomic_mass',
	'PymatgenData maximum block','PymatgenData maximum group','PymatgenData maximum melting_point',
	'PymatgenData maximum mendeleev_no','PymatgenData maximum row','PymatgenData maximum thermal_conductivity',
	'PymatgenData mean atomic_mass','PymatgenData mean block','PymatgenData mean group',
	'PymatgenData mean melting_point','PymatgenData mean mendeleev_no','PymatgenData mean row',
	'PymatgenData mean thermal_conductivity','PymatgenData minimum atomic_mass','PymatgenData minimum block',
	'PymatgenData minimum group','PymatgenData minimum melting_point','PymatgenData minimum mendeleev_no',
	'PymatgenData minimum row','PymatgenData minimum thermal_conductivity','PymatgenData range atomic_mass',
	'PymatgenData range block','PymatgenData range group','PymatgenData range melting_point',
	'PymatgenData range mendeleev_no','PymatgenData range row','PymatgenData range thermal_conductivity',
	'PymatgenData std_dev X','PymatgenData std_dev atomic_mass','PymatgenData std_dev block',
	'PymatgenData std_dev group','PymatgenData std_dev melting_point','PymatgenData std_dev mendeleev_no',
	'PymatgenData std_dev row','PymatgenData std_dev thermal_conductivity','avg ionic char','max ionic char'
	]
	matFeatureDf = pd.DataFrame({'matSet':MaterialMappingDf.index})
	matFeatureDf = matFeatureDf.apply(lambda x: {
		'matSet':x['matSet'], **compFeatureDic(x['matSet'].split('_')[0])
	}, axis=1, result_type='expand').set_index('matSet')
	matFeatureDf = matFeatureDf[compSubFeatureLst]

	# compile PBE lattice features
	with MPRester(paramDic['mpAPIKey']) as m:
		qEntryLst = m.query({'material_id': {'$in': MaterialMappingDf['mp-id'].to_list()}},['cif','material_id']) 
	cifDic    = dict((elem['material_id'],elem['cif']) for elem in qEntryLst)
	latticeDf = pd.DataFrame()
	for matSet,dentry in MaterialMappingDf.iterrows():
		latticeDf = latticeDf.append({'matSet':matSet,**latticeDicOf(
			'PBE',CifParser.from_string(cifDic[dentry['mp-id']]).get_structures(primitive=True)[0]
		)},ignore_index=True)
	matFeatureDf = pd.concat([latticeDf.set_index('matSet'),matFeatureDf],axis=1)

	# compile PBEsol lattice features
	with bz2.open(paramDic['PBEsolDBPath']) as fh:
		PBEsolJson = json.loads(fh.read().decode('utf-8'))
	matFeatureDf = pd.concat([ pd.DataFrame(dict([(matset, latticeDicOf(
		'PBEsol', ComputedStructureEntry.from_dict(PBEsolJson["entries"][int(dbentry['PBEsol-index'])]).structure
	)) for matset, dbentry in MaterialMappingDf.dropna().iterrows()])).T.rename_axis('matSet'),matFeatureDf],axis=1)
	matFeatureDf[list(set(PBEsolParamsDf.columns) - set(matFeatureDf.columns))] = None
	matFeatureDf.update(PBEsolParamsDf)

	# add ICSD volumes/temperatures & export matFeatureDf
	matFeatureDf.update(MaterialMappingDf.apply(lambda x: ICSDParamsDf.loc[x['icsd-id']],axis=1))
	matFeatureDf = pd.concat([MaterialMappingDf.apply(lambda x: ICSDParamsDf.loc[x['icsd-id']],axis=1),matFeatureDf],axis=1)
	matFeatureDf.reset_index().to_parquet(paramDic['matFeaturePath'],compression='gzip') 


if __name__ == '__main__':
	if len(sys.argv)==2:
		main({
			'mpAPIKey':            sys.argv[1], #API Key for accessing data of the Materials Project via MPRester
			'MaterialMappingPath': 'features/materialsMapping.csv.gz',  
			'PBEsolParamsDfPath':  'features/partialPBEsolParams.csv.gz',
			'PBEsolDBPath':        'external/2021.04.06_ps.json.bz2',
			'ICSDParamsDfPath':    'external/ICSDParams.csv',
			'matFeaturePath':      'external/matFeature.parquet',
		})
	else:
		print('[error]: Provide the Materials Project API Key as an argument.')

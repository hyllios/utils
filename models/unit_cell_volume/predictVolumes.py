#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os,joblib
sys.path.append('external')
from MAPLE import MAPLE # import MAPLE implementation of Plumb et al. [https://github.com/GDPlumb/MAPLE, arXiv:1807.02910]

import numpy as np
import pandas as pd
import itertools
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

versionStr 	=  '06_04_2022_01'

# definitions
flatten = lambda lst: list(itertools.chain.from_iterable(lst))

def replace(inputDic,replaceDic):
	res = inputDic.copy()
	res.update(replaceDic)
	return res 

def write_pandas_csv(filepath,paramDic,asmDic,varLst,versionStr,fmtLst,dataDf):
	with open(filepath,'w') as dtaFile: 
		dtaFile.write(
'''################
# [default parameters]
#\t{0}
# [assignments]
#\t{1}
# [variables]
#\t{2}
# [model version]
#\t{3}
# [format]
#\t{4}
################\n'''.format(paramDic,asmDic,varLst,versionStr,fmtLst))
		dataDf.to_csv(dtaFile, index=False, header=False, sep='\t')

def getFeatureLabelsToDB(DB,matFeatureDf, ignoreLst=[]):
	DBStrLen = len(DB)
	res   = [lbl for lbl in matFeatureDf.columns if lbl.find(DB,-DBStrLen) > -1 and lbl not in ignoreLst]
	return res

extractTinK = lambda x: x['TInK_ICSD']

def extractFtImportances(est,ftLabelLst,outpath=None, modelStr='',versionStr=''):
	n_trees             = est.n_estimators
	n_features          = len(ftLabelLst)
	ftDic               = dict()
	estOf               = lambda k: est[k] if type(est)==RandomForestRegressor else est[k,0] # the latter assumes type(est)==GradientBoostingRegressor
	
	ftDic['featureLbl'] = ftLabelLst
	ftDic['featureImportance'] = est.feature_importances_
	
	scoreDic = dict()
	[ scoreDic.setdefault(estOf(k).tree_.feature[0],[]).append(estOf(k).tree_.impurity[0]) 
	for k in range(n_trees) if estOf(k).tree_.feature[0]>=0];  # negative feature[0] indicates leaf nodes
	ftDic['rootImportance'] = [sum(scoreDic.get(k,[0])) for k in range(n_features)]
	ftDic['rootImportance']/=sum(ftDic['rootImportance'])
	
	impDf  = pd.DataFrame(ftDic).sort_values('rootImportance', ascending=False)
	if outpath:
		write_pandas_csv(outpath, {'model':modelStr},{},[],versionStr,
						'[featureLbl,rootImportance,featureImportance]',impDf)
	return impDf.set_index('featureLbl')

class MAPLE_wrapper(MAPLE):
	def __init__(self, X_train, MR_train, X_val, MR_val, fe_paramDic, fe_type = "rf"):
		regularization = fe_paramDic['regularization']
		fe_paramDic.pop('regularization')
		n_estimators = fe_paramDic['n_estimators']
		max_features = fe_paramDic['max_features']
		min_samples_leaf = fe_paramDic['min_samples_leaf']
		super().__init__(X_train, MR_train, X_val, MR_val, fe_type, n_estimators, max_features, min_samples_leaf, regularization)

# define main routine
def main(paramDic):
	tinitRun = timer()
	inputSplittingA 	= paramDic['inputSplittingA']
	random_stateA   	= paramDic['random_stateA']
	TMaxInK 			= paramDic['TMaxInK']
	TAsInput 			= paramDic['TAsInput']
	DBCmp				= paramDic['DBCmp']
	setLabel 			= paramDic['setLabel']
	versionStr 			= paramDic['versionStr']
	matFeaturePath 		= paramDic['matFeaturePath']
	dataPath 			= paramDic['dataPath'] 
	istestrun 			= paramDic['istestrun']
	suggested_features 	= paramDic['suggested_features']
	fracTrain			= paramDic['fracTrain']
	fe_paramDic     	= replace(paramDic['fe_paramDic'],{'random_state':random_stateA[2]})
	feParamsWithout 	= lambda negLst: dict((k,v) for k,v in fe_paramDic.items() if k not in negLst)
	
	# load database & assemble features
	matFeatureDf  = pd.read_parquet(matFeaturePath)
	DataSetDic    = dict()
	DBLst         = ['PBEsol', 'PBE']
	DBLst.remove(DBCmp)
	DataExtDf     = matFeatureDf.set_index('matSet')
	if setLabel=='icsd_common':
		DataExtDf     = DataExtDf.loc[DataExtDf[['volume_PBE','volume_PBEsol']].dropna().index]
	DataExtDf['TInK_ICSD']      = DataExtDf.apply(lambda x: extractTinK(x),axis=1)
	DataExtDf     = DataExtDf[DataExtDf['TInK_ICSD']<TMaxInK]
	neglectLblLst = flatten([DataExtDf.filter(regex='_{0}$'.format(db)).columns.to_list() for db in DBLst])
	DataExtDf     = DataExtDf.loc[DataExtDf.filter(regex='_{0}$'.format(DBCmp), axis=1).dropna(axis=0).index]
	DataExtDf     = DataExtDf.drop(columns=['redformula', 'compound possible']+neglectLblLst, errors='ignore').dropna(axis=1)
	DataSetDic['icsd_all']      = DataExtDf.drop(columns=['refSet'], errors='ignore')
	DataSetDic['icsd_common']   = DataExtDf.drop(columns=['refSet'], errors='ignore')
	DataDf = DataSetDic[setLabel].copy()
	
	if istestrun:
		DataDf = DataDf.iloc[range(60)]

	[os.makedirs(pth, exist_ok=True) for pth in [dataPath]]
	dataExportPathRoot  = os.path.join(dataPath,'volume_db({0})_set({1})_ft({2})_frac({3})_v({4})_rnd({5})'.format(
		DBCmp,setLabel,'gen1'+('+T'if TAsInput else ''),fracTrain,versionStr,paramDic['setIdx']))
	dataExportPath      = dataExportPathRoot+'.dat'
	volumeDBCmpLbl            = 'volume_'+DBCmp
	DBCmpSimpleVRegLbl        = DBCmp+'_simpleVReg'
	volumeDBCmpSimpleVRegLbl  = 'volume_'+DBCmpSimpleVRegLbl

	# train models, part 1
	ICSDFeatures              = ['volume_ICSD'] + (['TInK_ICSD'] if TAsInput else [])
	yPredDic                  = dict()
	X_full                    = DataDf.copy().drop(getFeatureLabelsToDB('ICSD',DataDf,ICSDFeatures),axis=1)
	y_full                    = X_full.pop('volume_ICSD')
	if suggested_features and len(suggested_features)>1:
		X_full = X_full[suggested_features]
	X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=inputSplittingA[0], random_state=random_stateA[0])
	X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=inputSplittingA[1], random_state=random_stateA[1])
	# truncate training set
	if fracTrain!=1:
		X_train  = X_train.iloc[:round(fracTrain*len(X_train))]
		X_valid  = X_train.iloc[:round(fracTrain*len(X_valid))]
		y_valid  = y_full[X_valid.index.to_list()]
		y_train  = y_full[X_train.index.to_list()]
	# assemble combined training set train_total
	X_train_total 			  = pd.concat([X_train,X_valid])
	y_train_total             = pd.concat([y_train,y_valid])

	# define predictive models
	clfDic				= dict()
	clfDic['maple_rf']  = MAPLE_wrapper(X_train.values, y_train.values, 
									X_valid.values, y_valid.values, feParamsWithout([]))

	# train models, part 2
	modelLabelLst = ['maple_rf']
	for clfLbl in modelLabelLst:
		clf       = clfDic[clfLbl]
		if clfLbl in ['maple_rf','maple_gb']:
			yPredDic[clfLbl] = {   
				'testsplit_full':  clf.predict(X_full.values),
				'testsplit_test':  clf.predict(X_test.values),
				'testsplit_train': clf.predict(X_train.values) 
			}   
	X_full_VolumeFeature          = X_full.loc[:,[volumeDBCmpLbl]]
	X_train_total_VolumeFeature   = X_train_total.loc[:,[volumeDBCmpLbl]]	
	
	clfDic[DBCmpSimpleVRegLbl]    = linear_model.LinearRegression(fit_intercept=True)
	clfDic[DBCmpSimpleVRegLbl].fit(X_train_total_VolumeFeature, y_train_total)
	yPredDic[DBCmpSimpleVRegLbl]  = clfDic[DBCmpSimpleVRegLbl].predict(X_full_VolumeFeature)   
	
	alphaLst = [fe_paramDic['regularization']]
	RidgeAlphaLblOf = lambda alpha: '{0}_alpha{1}'.format(DBCmpSimpleVRegLbl,alpha)
	for alpha in alphaLst:
		DBCmpSimpleVRidgeLbl = RidgeAlphaLblOf(alpha)
		clfDic[DBCmpSimpleVRidgeLbl]    = linear_model.Ridge(fit_intercept=True)
		clfDic[DBCmpSimpleVRidgeLbl].fit(X_train_total_VolumeFeature, y_train_total)
		yPredDic[DBCmpSimpleVRidgeLbl]  = clfDic[DBCmpSimpleVRidgeLbl].predict(X_full_VolumeFeature) 

	# export data
	predictionDf = pd.DataFrame({ 
		'isTrain':      pd.Series({idxFull: int(idxFull in X_train.index.values) for idxFull in X_full.index}),
		'isTest':       pd.Series({idxFull: int(idxFull in X_test.index.values) for idxFull in X_full.index}),
		'isValid':      pd.Series({idxFull: int(idxFull in X_valid.index.values) for idxFull in X_full.index}),
		'volume_ICSD':  y_full,
		volumeDBCmpLbl: X_full[volumeDBCmpLbl],
		volumeDBCmpSimpleVRegLbl: yPredDic[DBCmpSimpleVRegLbl],
	})
	predictionDf.index.name = y_full.index.name
	for alpha in alphaLst:
		predictionDf['{0}_alpha{1}'.format(volumeDBCmpSimpleVRegLbl,alpha)] =  yPredDic[RidgeAlphaLblOf(alpha)]
	for clfLbl in modelLabelLst:
		predictionDf[volumeDBCmpLbl+'+'+clfLbl] =  yPredDic[clfLbl]['testsplit_full']

	write_pandas_csv(dataExportPath,
		{
			volumeDBCmpSimpleVRegLbl+'_m': clfDic[DBCmpSimpleVRegLbl].coef_[0],
			'used_features': X_full.columns.to_list(),
			'paramDic': paramDic
		},{},['volume_ICSD'],
		versionStr,'[{0}]'.format(', '.join(predictionDf.reset_index().columns.values)), 
		predictionDf.reset_index(),
	)   
		
	# export model & importances
	for clfLbl in set(['maple','maple_rf','maple_gb']) and set(modelLabelLst):
		try:
			clf 		= clfDic[clfLbl]
			est         = clf.estimator if clfLbl=='maple' else clf.fe
			outpathRoot = '{0}.model({1})'.format(dataExportPathRoot,clfLbl)
			joblib.dump(clf, outpathRoot+'.md')
			extractFtImportances(est, X_full.columns.values, outpath=outpathRoot+'.ft', modelStr=outpathRoot,versionStr=versionStr)
		except:
			print('[error in exporting models/importances]\n::',outpathRoot)
	rndIdx = paramDic['setIdx']
	print('[elapsed time/s of run {1} for {2}]\n:: {0:1.1f}'.format(timer() - tinitRun,rndIdx,DBCmp))


if __name__ == '__main__':
	paramDicLst	= []
	for TAsInput in [True,False]:
		for DBCmp in ['PBEsol','PBE']:
			for fracTrain in np.linspace(0,1,9)[1:]:
				for rndIdx in range(0,15):
					paramDicLst.append({
						'TMaxInK':         400,
						'TAsInput':        TAsInput,
						'DBCmp':           DBCmp,
						'setLabel':        'icsd_common',
						'fe_paramDic': {
							'bootstrap':         True,
							'max_depth':         None,  
							'max_features':      'sqrt', 
							'min_samples_leaf':  1,
							'min_samples_split': 2,
							'n_estimators':      200,
							'regularization':    0.001,
						},
						'suggested_features':   None,
						'versionStr':       versionStr,
						'matFeaturePath':   'external/matFeature.parquet',
						'dataPath':         'data',
						'random_stateA':    [rndIdx+1,rndIdx+2,rndIdx],
						'inputSplittingA':  [0.2,0.5],
						'setIdx':           rndIdx,
						'fracTrain':        fracTrain,
						'istestrun':        False
					})
				
	for paramDic in paramDicLst:
		main(paramDic)

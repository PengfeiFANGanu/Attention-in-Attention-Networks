from __future__ import absolute_import



#Bilinear Part
from .bilinear_baseline import *            #BASELINE: Global Net (Inception)
from .bilinear_baseline_GL import *         #BASELINE:Global + local
from .bilinear_baseline_AIA import *        #BASELINE + aia
from .bilinear_aia_noHard_net import *      #no hard att

#Linear Part
from .linear_baseline import *              #BASELINE: Global Net (Inception)
from .linear_baseline_GL import *           #BASELINE:Global + local
from .linear_baseline_AIA import *          #BASELINE + aia
from .linear_aia_GL import *                #BASELINE + GL + aia

#RFF Part
from .rff_baseline import *                 #BASELINE: Global Net (Inception)
from .rff_baseline_GL import *              #BASELINE:Global + local
from .rff_baseline_AIA import *             #BASELINE + aia
from .rff_aia_GL import *                   #BASELINE + GL + aia

#####INCEPTION#####
from .incep_m_lin import*
from .incep_m_bi import*
from .incep_m_rff import*
from .incep_g import*
from .incep_g_mlin import*
from .incep_g_mbi import*
from .incep_g_mrff import*


__model_factory = {
    #Bilinear Part
    'bilinear_baseline': BilinearBaseline,  
    'bilinear_baseline_GL': BilinearBaselineGL, 
    'bilinear_baseline_AIA': BilinearBaselineAIA,
    'bilinear_aia_GL':BilinearAIAnetCnoHard,

    #Linear Part
    'linear_baseline': LinearBaseline,
    'linear_baseline_GL': LinearBaselineGL, 
    'linear_baseline_AIA': LinearBaselineAIA,
    'linear_aia_GL': LinearAIAGL,

    #Rff Part
    'rff_baseline': RffBaseline,
    'rff_baseline_GL': RffBaselineGL,
    'rff_baseline_AIA': RffBaselineAIA,
    'rff_aia_GL': RffAIAGL,

    #multiplear attention
    'incepmlin': IncepMLIN,
    'incepmbi': IncepMBI,
    'incepmrff': IncepMRFF,
    'incepg': IncepG,
    'incepgmlin': IncepGMLIN,
    'incepgmbi': IncepGMBI,
    'incepgmrff': IncepGMRFF,




}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)
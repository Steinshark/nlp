import supy,steps,calculables,samples
import os,math,copy,itertools,ROOT as r, numpy as np

class topAsymm3(supy.analysis) :
    ''' Analysis for measurement of production asymmetry in top/anti-top pairs
    
    #  This analysis contains several secondary calculables, which need
    #  to be primed in the following order:
    #  
    #  Pass#
    #  0: ^selection : prime sim.*   : pileup
    #  1: ^finegrain : prime top.top : jet.scalingBQN
    #  2: ^finegrain : prime top.top : RawMassWTopCorrectPQBTwoDChiSquared,jetCSVProbabilityGivenBQN,LTopUnfitSq...
    #  3: ^finegrain : prime top.top : HTopSigmasPQBCombinationsLR
    #  4: ^finegrain : prime needed  : TridiscriminantWTopQCD
    #  5.            : *
    '''

    def parameters(self) :

        reweights = {
            'abbr'  :'pu',
            'func'  :'pileup',
            'var'   :'pileupTrueNumInteractionsBX0Target'
            }

        leptons = {
            'name'     : [                    'mu',                      'el'],
            'ptMin'    : [                   26.0 ,                     30.0 ],
            'etaMax'   : [                    2.1 ,                      2.5 ],
            'isoNormal': [             {"max":0.12},             {'max':0.10}],
            'isoInvert': [ {"min":0.13, "max":0.20},  {'min':0.11, 'max':0.15}],
            'isoExtreme': [ {"min":0.16, "max":0.20}, {'min':0.13, 'max':0.15}],
            }

        #btag working points: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagPerformanceOP
        csvWP = {"CSVL" : 0.244, "CSVM" : 0.679, "CSVT" : 0.898 }
        bCut = {"normal"   : {"index":0, "min":csvWP['CSVM']},
                "inverted" : {"index":0, "min":csvWP['CSVL'], "max":csvWP['CSVM']}}

        return { "vary" : ['selection','lepton','toptype','smear','jec','ptscale'],
                 "nullvary": list(itertools.combinations(['ju','jd','su','sd','mn','30'],2)),
                 "discriminant2DPlots": True,
                 "bVar" : "CSV", # "Combined Secondary Vertex"
                 "objects" : dict([(item,(item,'')) for item in ['jet','mu','el','met']]),
                 "lepton" : self.vary([ ( leptons['name'][index], dict((key,val[index]) for key,val in leptons.iteritems()))
                                        for index in range(2) if leptons['name'][index] in ['mu','el'][:2]]),
                 "reweights" : reweights,
                 "selection" : self.vary({"top" : {"bCut":bCut["normal"],  "iso":"isoNormal"},
                                          "QCD" : {"bCut":bCut["normal"],  "iso":"isoInvert"},
                                          "QCDx": {"bCut":bCut["normal"],  "iso":"isoExtreme"}
                                          }),
                 "toptype" : self.vary({"ph":"ph"}),#,'mn':'mn'}),
                 "ptscale" : self.vary({"20":20,"30":30}),
                 "smear" : self.vary({'sn':"Smear",'su':'SmearUp','sd':'SmearDown'}),
                 "jec" : self.vary({'jn':0,'ju':1,'jd':-1}),
                 "topSamples": ("ttj_%s",['ttj_%s.wGG.%s','ttj_%s.wQG.%s','ttj_%s.wQQ.%s','ttj_%s.wAG.%s']),
                 }

    @staticmethod
    def doSystematics(pars) : return 'ph_sn_jn_20' in pars['tag'] 

    @staticmethod
    def scaleFactor() : return 1.0

    ########################################################################################

    def listOfSampleDictionaries(self) : return [getattr(samples,item) for item in ['lepton119_fnal', 'top119_fnal', 'ewk119_fnal']]

    @staticmethod
    def muons(suffix="") :     return [f+suffix for f in ['Mu.A.1','Mu.A.2','Mu.B.1','Mu.C.1','Mu.C.2','Mu.C.3','Mu.D.1']]
    @staticmethod
    def electrons(suffix="") : return [f+suffix for f in ['El.A.1','El.A.2','El.B.1','El.C.1','El.C.2','El.C.3','El.D.1']]
    @staticmethod
    def jsonFiles() :
        return ['lumi/json/'+f for f in ['Cert_190456-196531_8TeV_13Jul2012ReReco_Collisions12_JSON.txt',
                                         'Cert_190782-190949_8TeV_06Aug2012ReReco_Collisions12_JSON.txt',
                                         'Cert_190456-196531_8TeV_13Jul2012ReReco_Collisions12_JSON.txt',
                                         'Cert_198022-198523_8TeV_24Aug2012ReReco_Collisions12_JSON.txt',
                                         'Cert_190456-208686_8TeV_PromptReco-NoReprocessing_Collisions12CD_JSON.txt',
                                         'Cert_201191-201191_8TeV_11Dec2012ReReco-recover_Collisions12_JSON.txt',
                                         'Cert_190456-208686_8TeV_PromptReco-NoReprocessing_Collisions12CD_JSON.txt']]
    @staticmethod
    def single_top() : return ['top_s_ph','top_t_ph','top_tW_ph','tbar_s_ph','tbar_t_ph','tbar_tW_ph']

    def listOfSamples(self,pars) :
        rw = pars['reweights']['abbr']
        lname = pars['lepton']["name"]

        def ewk(eL = None) :
            dy = ['dy%dj_mg'%n for n in range(1,5)] if "QCD" not in pars['tag'] else []
            w =  ['wbb_mg'] + ['w%dj_mg'%n for n in range(1,5)]
            return supy.samples.specify( names = ( w + dy ),
                                         effectiveLumi = eL, color = 28, weights = [rw,'sf'] )

        def single_top(eL = None) :
            return supy.samples.specify( names = self.single_top(),
                                         effectiveLumi = eL, color = r.kGray, weights = [rw,'sf'])
        
        def ttbar(eL = None) :
            tt = pars['toptype']
            color = [r.kBlue,r.kCyan,r.kCyan+1,r.kOrange]
            wSub =  [  "wGG",  "wQG",    "wAG",    "wQQ"]
            return sum([supy.samples.specify(names = 'ttj_%s'%tt, effectiveLumi = eL,
                                             color = c, weights = [w,rw,'sf']) for w,c in zip(wSub,color)],[])

        def ttextra(eL = None) :
            tt = pars['toptype']
            color = [r.kBlue,r.kCyan,r.kCyan+1,r.kOrange][1:]
            wSub =  [  "wGG",  "wQG",    "wAG",    "wQQ"][1:]
            return sum([supy.samples.specify(names = 'extra_%s'%tt, effectiveLumi = eL,
                                             color = c, weights = [w,rw,'sf']) for w,c in zip(wSub,color)],[])

        def calib(eL = None):
            colors = [r.kBlack, r.kBlack, r.kOrange, r.kBlue, r.kCyan, r.kGreen, r.kRed, r.kGray]
            names=['calib_mg','calib_mn','calib_ZP','calib_A2K',
                   'calib_A200','calib_L200','calib_R2K','calib_R200']
            return sum([supy.samples.specify(names=n, color=c, weights = [rw,'sf']) for c,n in zip(colors,names)], [])
        
        def data() :
            return sum( [supy.samples.specify( names = ds, weights = calculables.other.jw(jfn))
                         for ds,jfn in zip({"mu":self.muons(),
                                            "el":self.electrons()}[pars['lepton']['name']],
                                           self.jsonFiles())],[])

        return  ( data() + ewk() + ttbar() + single_top() + ttextra() + calib())

    ########################################################################################
    def listOfCalculables(self, pars) :
        obj = pars["objects"]
        jet = obj['jet']
        met = obj['met']
        mu = obj['mu']
        el = obj['el']
        lepton = obj[pars["lepton"]["name"]]
        calcs = sum( [supy.calculables.zeroArgs(module) for module in [calculables, supy.calculables]]+
                     [supy.calculables.fromCollections(getattr(calculables,item), [obj[it]])
                      for item,it in  [('jet','jet'),('electron','el'),('muon','mu')]], [])
        calcs += supy.calculables.fromCollections(calculables.top,[('genTop',""),('fitTop',"")])
        calcs += [
            calculables.met.AdjustedP4(met, jet, pars['smear'], djec=pars['jec']),
            calculables.jet.AdjustedP4(jet, pars['smear'], pars['jec']),
            calculables.jet.Indices(jet,ptMin = 20 ),
            calculables.jet.IndicesBtagged(jet,pars["bVar"]),
            supy.calculables.other.abbreviation('combinedSecondaryVertex','CSV',jet),
            calculables.muon.Indices(mu),
            calculables.electron.Indices(el),

            calculables.gen.genIndicesHardPartons({'ph':'POWHEG','mn':'MC@NLO'}[pars['toptype'][:2]]),
            calculables.gen.genIndexTtbarExtraJet({'ph':'POWHEG','mn':'MC@NLO'}[pars['toptype'][:2]]),
            calculables.gen.qPtMin(pars['ptscale']),
            calculables.top.TopJets( jet ),
            calculables.top.TopLeptons( lepton ),

            calculables.comb.TopComboQQBBLikelihood( pars['bVar'] ),
            calculables.comb.OtherJetsLikelihood( pars['bVar'] ),
            calculables.comb.TopRatherThanWProbability( priorTop = 0.05 ),

            calculables.met.MetMt( lepton, "AdjustedP4".join(met)),
            calculables.met.Covariance( met ),

            calculables.jet.pt( jet, index = 0, Btagged = True ),
            calculables.jet.pt( jet, index = 3, Btagged = False ),
            calculables.jet.absEta( jet, index = 3, Btagged = False),

            supy.calculables.other.pt( "AdjustedP4".join(met) ),
            supy.calculables.other.size( "Indices".join(jet) ),
            supy.calculables.other.abbreviation( pars['reweights']['var'], pars['reweights']['abbr'] ),
            supy.calculables.other.abbreviation( 'SF'.join(lepton), 'sf'),

            supy.calculables.other.QueuedBin( 5, ("fitTopTanhDeltaAbsY", "fitTopDPtDPhi"), (1,1), 'fitTop'),
            supy.calculables.other.QueuedBin( 5, ("genTopTanhDeltaAbsY", "genTopDPtDPhi"), (1,1), 'genTop'),
            ]
        if self.doSystematics(pars) :
            calcs.append(calculables.other.pileUpRatios( 'pileupTrueNumInteractionsBX0',
                                                         ['lumi/dsets_%s%s_pileup.root'%(pars['lepton']['name'],s)
                                                          for s in ['','_down','_up']]))
            if any(s in pars['sample'] for s in ['ttj','extra']):
                calcs.append(calculables.gen.genPdfWeights('CT10.LHgrid','CT10as.LHgrid'))
                calcs.append(calculables.gen.genPtWeights(8,'semi'))
        return calcs
    ########################################################################################

    def listOfSteps(self, pars) :
        obj = pars["objects"]
        jet = obj['jet']
        met = obj['met']
        lname = pars["lepton"]["name"]
        lepton = obj[lname]
        otherLepton = obj[{'el':'mu','mu':'el'}[lname]]
        lIsoMinMax = pars["lepton"][pars['selection']['iso']]
        bVar = pars["bVar"].join(jet)
        rw = pars['reweights']['abbr']
        tt = pars['toptype']
        topSamples = (pars['topSamples'][0]%tt,[s%(tt,'.'.join([rw,'sf'])) for s in pars['topSamples'][1]])
        topTag = pars['tag'].replace("QCD","top")
        ttSmpTag = {'samples':topSamples[1], 'tag':topTag}

        ssteps = supy.steps

        saDisable = not any(s in pars['sample'] for s in ['ttj','extra']) or not any(w in pars['sample'].split('.') for w in ['wQQ','wQG','wAG','wGG'])
        saWeights = []
        return (
            [ssteps.printer.progressPrinter()
             , ssteps.histos.value("genQ",200,0,1000,xtitle="#hat{Q} (GeV)").onlySim()
             , steps.gen.qRecoilKinematics().disable(saDisable)
             , getattr(self,pars['reweights']['func'])(pars)
             , ssteps.other.reweights( ssteps.histos.value('pileupTrueNumInteractionsBX0',100,0,60),
                                       'pileUpRatios', 2, self.doSystematics(pars)).onlySim()
             , ssteps.other.reweights( ssteps.histos.value( ('genTopTanhDeltaAbsY','genTopDPtDPhi'), (2,2), (-1,-1), (1,1) ),
                                       'genPdfWeights', 57, self.doSystematics(pars) ).disable(saDisable)
             , ssteps.other.reweights( ssteps.histos.value( ('genTopTanhDeltaAbsY','genTopDPtDPhi'), (2,2), (-1,-1), (1,1) ),
                                       'genPtWeights', 3, self.doSystematics(pars) ).disable(saDisable)
             , ssteps.histos.value( ('genTopTanhDeltaAbsY','genTopDPtDPhi'), (2,2),(-1,-1),(1,1)).disable(saDisable and 'calib' not in pars['sample'])
             , steps.gen.pdfWeightsPlotter(['genTopTanhRapiditySum','genTopPtOverSumPt',
                                            'genTopTanhDeltaAbsY','genTopDPtDPhi'],
                                           [0,0,-1,-1],
                                           [1,1,1,1]).disable(saDisable or not self.doSystematics(pars))
             ####################################
             , ssteps.filters.label('selection'),
             ssteps.filters.value("mvaTrigV0Exists",min=True), # Event Cleaning
             ssteps.filters.value("muHandleValid",min=True),
             ssteps.filters.multiplicity( max=1, min=1, var = "Charge".join(lepton)),
             ssteps.filters.multiplicity( max=0, var = "Charge".join(otherLepton)),
             ssteps.filters.multiplicity( min=1, var = 'Indices'.join(lepton)),
             [ssteps.filters.value('PassConversionVeto'.join(lepton), min=True, indices='Indices'.join(lepton), index=0),
              supy.steps.filters.label('empty')][lname!='el'],
             ssteps.filters.value("Pt".join(jet), min = 45, indices = "Indices".join(jet), index=0),
             ssteps.filters.value("Pt".join(jet), min = 35, indices = "Indices".join(jet), index=1),
             ssteps.filters.value("Pt".join(jet), min = 20, indices = "Indices".join(jet), index=2),
             ssteps.filters.value("Pt".join(jet), max = 20, indices = "Indices".join(jet), index=3, allowNone=True),
             ssteps.filters.value(bVar, indices = "IndicesBtagged".join(jet), **pars["selection"]["bCut"]),
             ssteps.filters.value('RelIso'.join(lepton), indices='Indices'.join(lepton), index=0, **lIsoMinMax),
             steps.trigger.singleLepton(lname=='mu')
             ####################################
             , ssteps.filters.label("selection complete")
             , steps.top.channelClassification().onlySim()
             , ssteps.histos.multiplicity('Indices'.join(jet))
             , ssteps.histos.value( 'genTopTanhDeltaAbsY', 2, -1, 1 ).disable(saDisable)
             , ssteps.histos.value( 'genTopTanhDeltaAbsY', 25, -1, 1 ).disable(saDisable)
             , ssteps.histos.value( 'genTopDPtDPhi', 2, -1, 1 ).disable(saDisable)
             , ssteps.histos.value( 'genTopDPtDPhi', 25, -1, 1 ).disable(saDisable)
             , ssteps.histos.value( 'fitTopTanhDeltaAbsY', 2, -1, 1 )
             , ssteps.histos.value( 'fitTopTanhDeltaAbsY', 25, -1, 1 )
             , ssteps.histos.value( 'fitTopDPtDPhi', 2, -1, 1 )
             , ssteps.histos.value( 'fitTopDPtDPhi', 25, -1, 1 )
             , ssteps.filters.label("secondaries")
             , calculables.jet.ProbabilityGivenBQN(jet, pars['bVar'], binning=(51,-0.02,1), **ttSmpTag)
             , calculables.jet.ScalingBQN(jet, samples = topSamples[1], tag = topTag)
             , supy.calculables.other.CombinationsLR( var='ProxyMassPQB', varMax=500, trueKey='IndicesGenTopPQH', label="m_{proxy}", **ttSmpTag)
             , supy.calculables.other.CombinationsLR( var='LTopUnfitSqrtChi2', varMax=10, trueKey='IndexGenTopL', label="#chi_{a}", **ttSmpTag)
             , supy.calculables.other.CombinationsLR( var='TopComboQQBBLikelihoodRatio', varMax=1, trueKey='IndicesGenTopQQBB', label='L_i^{CSV}/max(L^{CSV})', **ttSmpTag)
             , self.tridiscriminant(pars)

             , ssteps.filters.label('finegrain')
             , ssteps.histos.value('MetMt'.join(lepton), 120, 0, 120)
             , ssteps.histos.value('ProbabilityHTopMasses', 100,0,1)
             , ssteps.histos.value("TopRatherThanWProbability", 100,0,1)
             #, steps.top.chosenCombo()

             , ssteps.filters.label('object pt')
             , ssteps.histos.pt('AdjustedP4'.join(met), 125, 0, 250 )
             , ssteps.histos.pt('P4'.join(lepton), 125, 0, 250, indices = 'Indices'.join(lepton), index=0)
             , ssteps.histos.value("Pt".join(jet), 125, 0, 250, indices = 'Indices'.join(jet), index=0)
             , ssteps.histos.value("Pt".join(jet), 125, 0, 250, indices = 'Indices'.join(jet), index=1)
             , ssteps.histos.value("Pt".join(jet), 125, 0, 250, indices = 'Indices'.join(jet), index=2)
             , ssteps.histos.value("Pt".join(jet), 125, 0, 250, indices = 'Indices'.join(jet), index=3)

             , ssteps.filters.label('object eta')
             , ssteps.histos.absEta("P4".join(lepton), 100,0,4, indices = "Indices".join(lepton), index = 0)
             , ssteps.histos.absEta("AdjustedP4".join(jet), 100,0,4, indices = "Indices".join(jet), index = 0)
             , ssteps.histos.absEta("AdjustedP4".join(jet), 100,0,4, indices = "Indices".join(jet), index = 1)
             , ssteps.histos.absEta("AdjustedP4".join(jet), 100,0,4, indices = "Indices".join(jet), index = 2)
             , ssteps.histos.absEta("AdjustedP4".join(jet), 100,0,4, indices = "Indices".join(jet), index = 3)

             , ssteps.histos.value( 'RelIso'.join(lepton), 100,0,0.2, indices='Indices'.join(lepton), index=0 )
             , ssteps.histos.value( bVar, 100,0,1, indices='IndicesBtagged'.join(jet), index=0 )
             , ssteps.histos.value( bVar, 100,0,1, indices='IndicesBtagged'.join(jet), index=1 )

             ####################################
             #, steps.displayer.ttbar(jets=jet, met=obj['met'], muons = obj['mu'], electrons = obj['el'])
             , ssteps.filters.label('top reco')
             #, steps.top.combinatorialFrequency().onlySim()
             #, ssteps.histos.value('genTopRecoIndex', 10,-1.5,8.5)
             #, ssteps.histos.value('TopGenLikelihoodIndex', 10,-1.5,8.5)
             #, ssteps.histos.value('TopFitLikelihoodCorrectIndex',10,-1.5,8.5)

             , ssteps.filters.label('genTop')
             #, steps.top.kinFitLook('genTopRecoIndex')
             #, steps.top.resolutions('genTopRecoIndex')

             , ssteps.filters.label('recoTop')
             #, steps.top.kinFitLook('fitTopRecoIndex')
             #, steps.top.resolutions('fitTopRecoIndex')
             ####################################
             , ssteps.filters.label('kinematics')
             #, steps.top.kinematics('fitTop')
             #, steps.top.kinematic_resolution({'TanhRapiditySum':(100,0,1,0.5),
             #                                  'MassSum':(30,300,1200,450),
             #                                  'PtSum':(100,0,300,50),
             #                                  'TanhDeltaAbsY':(100,-1,1,0),
             #                                  'DPtDPhi':(100,-1,1,0)}, topSamples[1], topTag).disable(saDisable)
             #, ssteps.histos.mass('fitTopSumP4', 30, 300, 1200)
             #, ssteps.histos.pt(  'fitTopSumP4', 100, 0, 300)
             #, ssteps.histos.value('fitTopRapiditySum', 50, 0, 3, xtitle = '|t#bar{t}.y|')
             #, ssteps.histos.value('fitTopTanhRapiditySum', 100, 0, 1)
             #, ssteps.histos.value('fitTopPtOverSumPt', 100, 0, 1)
             #
             #, ssteps.filters.label('asymmetry')
             #, ssteps.histos.value('fitTopTanhDeltaAbsY',100,-1,1)
             #, ssteps.histos.value('fitTopDPtDPhi',100,-1,1)
             #, ssteps.histos.value('fitTopQueuedBin5',25,-1,1)
             #, ssteps.histos.value('genTopQueuedBin5',25,-1,1).disable(saDisable)
             ] +
            ###################################
            #self.signalSequence(pars,saDisable) +
            #self.signalSequence(pars,saDisable, ssteps.filters.mass('fitTopSumP4',min=450)) +
            #self.signalSequence(pars,saDisable, ssteps.filters.value('fitTopTanhRapiditySum',min=0.5)) +
            ###################################
             [ ssteps.filters.label('values')
             #, ssteps.filters.value('fitTopTanhRapiditySum',min=0.5)
             #, ssteps.histos.value('fitTopTanhRapiditySum', 100,0,1)
             #, ssteps.histos.value('fitTopPtOverSumPt', 100,0,1)
             #, ssteps.histos.value('TridiscriminantWTopQCD', 100,-1,1)
             #, ssteps.histos.value('fitTopTanhDeltaAbsY',100,-1,1)
             #, ssteps.histos.value('fitTopDPtDPhi',100,-1,1)
             #, ssteps.histos.value('fitTopQueuedBin5',25,-1,1)
             #, ssteps.histos.value('genTopQueuedBin5',25,-1,1).disable(saDisable)
             ])
    ########################################################################################

    def signalSequence(self, pars, saDisable, predicate=None) :
        ssteps = supy.steps
        triD = ('TridiscriminantWTopQCD',5,-1,1)
        isData = pars['sample'].split('.')[0] in ['El','Mu']

        asymm = "steps.top.signalhists(doGen=not saDisable)"
        kinem = "steps.top.kinematics3D('fitTop')"
        doSys = self.doSystematics(pars)

        pdf = 'ssteps.other.reweights( eval("%s"), "genPdfWeights", 57, doSys and not saDisable, predicate)'
        pu = 'ssteps.other.reweights( eval("%s"),  "pileUpRatios",  2, doSys and not isData,    predicate)'
        effEl = 'ssteps.other.reweights( eval("%s"), "elReweights", 4, doSys and not isData and "_el_" in pars["tag"], predicate)'
        effMu = 'ssteps.other.reweights( eval("%s"), "muReweights", 4, doSys and not isData and "_mu_" in pars["tag"], predicate)'
        pt = 'ssteps.other.reweights( eval("%s"), "genPtWeights", 3, doSys and not saDisable, predicate)'

        return [eval(pdf % asymm), eval(pu % asymm), eval(effEl % asymm), eval(effMu % asymm), eval(pt % asymm),
                eval(pdf % kinem), eval(pu % kinem), eval(effEl % kinem), eval(effMu % kinem), eval(pt % kinem)
                ]

    @classmethod
    def pileup(cls,pars) :
        rw = pars['reweights']['abbr']
        tt = pars['toptype']
        targetFile = 'lumi/dsets_%s_pileup.root'%pars['lepton']['name']
        return supy.calculables.other.Target("pileupTrueNumInteractionsBX0", thisSample = pars['baseSample'],
                                             target = (targetFile,"pileup"),
                                             groups = [('dy1j_mg',[]),('dy2j_mg',[]),('dy3j_mg',[]),('dy4j_mg',[]),
                                                       ("w1j_mg",[]), ("w2j_mg",[]), ("w3j_mg",[]), ("w4j_mg",[]),
                                                       ('single_top', ['.'.join([s,rw,'sf']) for s in cls.single_top()]),
                                                       ('ttj_%s'%tt,['ttj_'+'.'.join([tt,s,rw,'sf'])
                                                                     for s in ['','wGG','wQG','wAG','wQQ']]),
                                                       ('extra_%s'%tt,['extra_'+'.'.join([tt,s,rw,'sf'])
                                                                       for s in ['','wQG','wAG','wQQ']]),
                                                       ('calib_mg',[]),
                                                       ('calib_mn',[]),
                                                       ('calib_R200',[]),
                                                       ('calib_L200',[]),
                                                       ('calib_A200',[]),
                                                       ('calib_R2K',[]),
                                                       ('calib_A2K',[]),
                                                       ('calib_ZP',[]),
                                                       ('wbb_mg',[])
                                                       ]).onlySim()

    def tridiscriminant(self,pars) :
        rw = pars['reweights']['abbr']
        lname = pars['lepton']['name']
        tt = pars['toptype']
        tops = ['ttj_'+'.'.join([tt,s,rw,'sf']) for s in ['wGG','wQG','wAG','wQQ']]
        others = ['.'.join([o,rw,'sf']) for o in self.single_top() + ['w%dj_mg.%s.sf'%(d,rw) for d in [1,2,3,4]]]
        datas = {"mu" : self.muons('.jw'),
                 "el": self.electrons('.jw')}[lname]
        sf = self.scaleFactor()
        topTag = pars['tag'].replace("QCD","top")
        qcdTag = pars['tag'].replace("top","QCD")

        pi43 = {"pre":"wj",        "tag":topTag, "samples":['w%dj_mg.%s.sf'%(n,rw) for n in [1,2,3,4]]}
        zero = {"pre":"ttj_%s"%tt, "tag":topTag, "samples": tops}
        pi23 = {"pre":"Multijet",  "tag":qcdTag, "samples":['data']+tops+others, 'sf':[1] + [-sf]*len(tops+others)}

        return supy.calculables.other.Tridiscriminant( fixes = ("","WTopQCD"),
                                                       pi43 = pi43, zero = zero, pi23 = pi23,
                                                       correlations = pars['discriminant2DPlots'],
                                                       otherSamplesToKeep = datas,
                                                       dists = {"TopRatherThanWProbability" : (20,0,1),
                                                                'ProbabilityHTopMasses' : (20,0,1),
                                                                "MetMt".join(pars['objects'][lname]) : (20,0,100),
                                                                })
    ########################################################################################
    def tridiscriminant2(self,pars) :
        rw = pars['reweights']['abbr']
        lname = pars['lepton']['name']
        tt = pars['toptype']
        tops = ['ttj_'+'.'.join([tt,s,rw,'sf']) for s in ['wGG','wQG','wAG','wQQ']]
        topTag = pars['tag'].replace("QCD","top")

        return supy.calculables.other.Tridiscriminant( fixes = ("","QQggQg"),
                                                       zero = {"pre":"gg", "tag":topTag, "samples": tops[:1]},
                                                       pi23 = {"pre":"qq", "tag":topTag, "samples": tops[-1:]},
                                                       pi43 = {"pre":"qg", "tag":topTag, "samples": tops[1:-1]},
                                                       correlations = pars['discriminant2DPlots'],
                                                       dists = {"fitTopTanhRapiditySum" : (50,0,1),
                                                                'extraJetMoments2Sum' : (50,0,0.11),
                                                                })
    ########################################################################################
    def concludeAll(self) :
        self.orgMelded = {}
        self.sensitivityPoints = {}
        self.rowcolors = 2*[13] + 2*[45]
        super(topAsymm3,self).concludeAll()

        for tagSuffix in set( '_'.join(pars['tag'].split('_')[1:]) for pars in self.readyConfs) :
            self.meldScale(tagSuffix)
            self.plotMeldScale(tagSuffix)

    def conclude(self,pars) :
        rw = pars['reweights']['abbr']

        org = self.organizer(pars, verbose = True )

        org.mergeSamples(targetSpec = {"name":"Data 2012", "color":r.kBlack, "markerStyle":20},
                         sources=getattr(self,'muons' if pars['lepton']['name']=='mu' else 'electrons')('.jw') )
            
        org.mergeSamples(targetSpec={"name":"t#bar{t}", "color":r.kViolet}, allWithPrefix='ttj', keepSources=True)
        org.mergeSamples(targetSpec={"name":"Wbb", "color":r.kRed}, allWithPrefix='wbb')
        org.mergeSamples(targetSpec={"name":"W", "color":28}, allWithPrefix='w')
        org.mergeSamples(targetSpec={"name":"DY", "color":r.kYellow}, allWithPrefix="dy")
        org.mergeSamples(targetSpec={"name":"Single", "color":r.kGray}, sources=['.'.join([s,rw,'sf']) for s in self.single_top()])
        org.mergeSamples(targetSpec={"name":"St.Model", "color":r.kGreen+2}, sources=["t#bar{t}","W","DY","Single"],
                         keepSources=True)
        try:
            self.skimStats(org)
            self.printTable(org)
            self.skimControl(org)
        except: pass
        org.scale( lumiToUseInAbsenceOfData = 19590, toPdf=False )

        names = [ss["name"] for ss in org.samples]
        kwargs = {"detailedCalculables": False,
                  "blackList":["lumiHisto","xsHisto","nJobsHisto"],
                  "samplesForRatios" : next(iter(filter(lambda x: x[0] in names and x[1] in names,
                                                        [("Data 2012","St.Model")])), ("","")),
                  "sampleLabelsForRatios" : ("data","s.m."),
                  "detailedCalculables" : True,
                  "rowColors" : self.rowcolors,
                  "rowCycle" : 100,
                  "omit2D" : True,
                  }
        
        #supy.plotter(org, pdfFileName = self.pdfFileName(org.tag+"_log"),  doLog=True, pegMinimum=0.01, **kwargs ).plotAll()
        supy.plotter(org, pdfFileName = self.pdfFileName(org.tag+"_nolog"), doLog=False, **kwargs ).plotAll()

    def statsname(self,org):
        _,lepton,tt,smear,jec,pt = org.tag.split('_')
        print org.tag
        return {'DY':'dy',
                'W': 'wj',
                't#bar{t}':'tt',
                'ttj_%s.wGG.pu.sf'%tt:'ttgg',
                'ttj_%s.wQG.pu.sf'%tt:'ttqg',
                'ttj_%s.wAG.pu.sf'%tt:'ttag',
                'ttj_%s.wQQ.pu.sf'%tt:'ttqq',
                'extra_%s.wQQ.pu.sf'%tt:'ettqq',
                'extra_%s.wQG.pu.sf'%tt:'ettqg',
                'extra_%s.wAG.pu.sf'%tt:'ettag',
                'Single' : 'st',
                'Data 2012': 'data'
                }

    def skimStats(self,org) :
        statsname = self.statsname(org)
        fileName = '%s/stats_%s.root'%(self.globalStem,org.tag)
        tfile = r.TFile.Open(fileName,'RECREATE')

        for g in ['lumiHisto','xsHisto','meweighted','genTopTanhDeltaAbsY; genTopDPtDPhi'] :
            tfile.mkdir(g.replace(';','').replace(' ','_'),'_').cd()
            index = next(i for i in org.indicesOfStepsWithKey(g) if 'reweights' not in org.steps[i].name)
            for ss,hist in zip( org.samples,
                                org.steps[index][g] ) :
                if not hist or ss['name'] in ['St.Model','S.M.']: continue
                h = hist.Clone(statsname[ss['name']] if ss['name'] in statsname else
                               ss['name'])
                h.Write()
        for iRe,iStep in enumerate(org.indicesOfStep('reweights')) :
            step = org.steps[iStep]
            dirname = 'R%02d_'%iRe+''.join(step.title.split()).replace(';','_').replace('(','').replace(')','')
            dir = tfile.mkdir(dirname,'_')
            for g in sorted(step):
                dir.mkdir(g,'_').cd()
                for ss,hist in zip( org.samples,
                                    step[g] ) :
                    if not hist or ss['name'] in ['St.Model','S.M']: continue
                    h = hist.Clone(statsname[ss['name']] if ss['name'] in statsname else
                                   ss['name'])
                    h.Write()
        tfile.Close()
        print 'Wrote: ', fileName

    def skimControl(self,org):
        if 'ph_sn_jn_20' not in org.tag: return
        controlname = self.statsname(org)
        fileName = "%s/control_%s.root"%(self.globalStem,org.tag)
        tfile = r.TFile.Open(fileName,"RECREATE")

        # everything starting frome 'finegrain' through last absEta (except 'counts')
        # histos.mass('fitTopSumP4')
        # histos.value('fitTopRapiditySum')
        def save(step, Only=[]):
            for g in filter(lambda s:s!='counts',sorted(step)):
                if Only and not g in Only: continue
                tfile.mkdir(g.split(';')[0],'_').cd()
                for ss,hist in zip(org.samples, step[g]):
                    if not hist or ss['name'] in ['St.Model','S.M.'] : continue
                    h = hist.Clone(controlname[ss['name']] if ss['name'] in controlname else ss['name'])
                    h.Write()

        start,trip = False, False
        for step in org.steps:
            if step.name == 'TridiscriminantWTopQCD': save(step, ['TridiscriminantWTopQCD'])
            #if step.name == 'qRecoilKinematics': save(step, ['qRecoilPt'])
            start|= step.nameTitle == ('label','finegrain')
            if not start: continue
            trip|= 'absEta' == step.name
            other = [('kinematics','fitTop'),
                     ('kinFitLook','fitTopRecoIndex'),
                     ('mass','fitTopSumP4.mass'),
                     ('value','fitTopTanhRapiditySum'),
                     ]
            if trip and not (step.name == 'absEta' or
                             step.nameTitle in other):
                continue
            save(step)
            if step.nameTitle == other[-1] : break
        tfile.Close()
        print 'Wrote: ', fileName

    def printTable(self,org):
        space = '   &   '
        fileName = '%s/eff_%s.txt'%(self.globalStem,org.tag)
        samples = ['Data 2012','t#bar{t}','W','Single','DY']
        def eff(h):
            n = h.GetEffectiveEntries()
            F,P = [h.GetBinContent(i) for i in [1,2]]
            p = P/(F+P)
            return 100*p, 100*math.sqrt(p*(1-p)/n)
        
        with open(fileName,'w') as f:
            print>>f, space + space.join(samples)
            for step in org.steps:
                if not step.isSelector: continue
                if step.name=='label': continue
                sampleFP = dict([(ss['name'],FP) for ss,FP in zip(org.samples,step['counts'])])
                print>>f, step.nameTitle, space, "%0.3f"%(sampleFP['Data 2012'][2]/1000.), space, space.join( '%.4f(%.4f)'%eff(h) for h in [sampleFP[s] for s in samples[1:-1 if 'QCD' in org.tag else None]]), r'//'
        print 'Wrote: ', fileName

    def plotMeldScale(self,tagSuffix) :
        if tagSuffix not in self.orgMelded :
            print tagSuffix, "not in", self.orgMelded.keys()
            print "run meldScale() before plotMeldScale()"; return
        melded = copy.deepcopy(self.orgMelded[tagSuffix])
        lname,tt,sn,jn,ptMin = tagSuffix.split('_')
        for s in ['top.ttj_%s.%s.pu'%(tt,s) for s in ['wQQ','wQG','wAG','wGG']] :
            melded.drop(s)
        for log,label in [(False,""),(True,"_log")][:1] : 
            pl = supy.plotter(melded, pdfFileName = self.pdfFileName(melded.tag + label),
                              doLog = log,
                              blackList = ["lumiHisto","xsHisto","nJobsHisto"],
                              rowColors = self.rowcolors,
                              samplesForRatios = ("top.Data 2012","S.M."),
                              sampleLabelsForRatios = ('data','s.m.'),
                              rowCycle = 100,
                              omit2D = True,
                              pageNumbers = False,
                              ).plotAll()
        print
        print 'fitTopDPtDPhi indices:'
        print melded.indicesOfStepsWithKey('fitTopDPtDPhi')

    def meldScale(self,tagSuffix) :
        lname,tt,sn,jn,ptMin = tagSuffix.split('_')
        meldSamples = {"top_"+tagSuffix :( { 'mu': self.muons('.jw'),
                                             'el': self.electrons('.jw')}[lname]+
                                           ["ttj_%s"%tt]+self.single_top()+
                                           ["dy%dj_mg"%n for n in [1,2,3,4]]+
                                           ["w%dj_mg"%n for n in [1,2,3,4]]),
                       "QCD_"+tagSuffix : ( { 'mu':self.muons('.jw'),
                                              'el':self.electrons('.jw')}[lname] +
                                            ["ttj_%s"%tt]+self.single_top()+
                                            ["w%dj_mg"%n for n in [1,2,3,4]]),
                       }
        organizers = [supy.organizer( tag, keepTH2=False,
                                      sampleSpecs=[s for s in self.sampleSpecs(tag)
                                                   if any(item in s['name'] for item in meldSamples[tag])])
                      for tag in [p['tag'] for p in self.readyConfs if p["tag"] in meldSamples]]

        if len(organizers) < len(meldSamples) : return
        for org in organizers :
            org.mergeSamples( targetSpec={"name":"t#bar{t}", "color":r.kViolet}, allWithPrefix='ttj')
            org.mergeSamples( targetSpec={"name":"W", "color":r.kRed}, allWithPrefix='w')
            org.mergeSamples( targetSpec={"name":"DY", "color":28}, allWithPrefix="dy")
            org.mergeSamples( targetSpec={"name":"Single", "color":r.kGray}, sources=["%s.pu.sf"%s for s in self.single_top()])
            org.mergeSamples( targetSpec={"name":"Data 2012", "color":r.kBlack, "markerStyle":20},
                              sources={'mu':self.muons('.jw'),'el':self.electrons('.jw')}[lname])
            org.scale()
            if "QCD_" in org.tag :
                sm = ['t#bar{t}','W','Single']
                org.mergeSamples(targetSpec = {"name":"multijet","color":r.kBlue},
                                 sources=["Data 2012"]+sm,
                                 scaleFactors = [1]+len(sm)*[-self.scaleFactor()],
                                 force=True)

        self.orgMelded[tagSuffix] = supy.organizer.meld(organizers = organizers)
        org = self.orgMelded[tagSuffix]
        templateSamples = ['top.t#bar{t}','top.W','QCD.multijet']
        baseSamples = ['top.Single','top.DY']

        mfCanvas = r.TCanvas()
        mfFileName = "%s/measuredFractions_%s"%(self.globalStem, tagSuffix )
        supy.utils.tCanvasPrintPdf( mfCanvas, mfFileName, option = '[', verbose = False)
        with open(mfFileName+'.txt','w') as file : print >> file, ""
        
        def measureFractions(dist, rebin = 1) :
            before = next(org.indicesOfStep('label','selection complete'))
            distTup = org.steps[next(iter(filter(lambda i: before<i, org.indicesOfStepsWithKey(dist))))][dist]

            templates = [None] * len(templateSamples)
            bases = []
            for ss,hist in zip(org.samples,distTup) :
                contents = supy.utils.binValues(hist)
                if rebin!=1 :
                    contents = ( contents[:1] +
                                 [sum(bins) for bins in zip(*[contents[1:-1][i::rebin] for i in range(rebin)])] +
                                 contents[-1:] )
                if ss['name'] == "top.Data 2012" :
                    observed = contents
                elif ss['name'] in templateSamples :
                    templates[templateSamples.index(ss['name'])] = contents
                elif ss['name'] in baseSamples :
                    bases.append(contents)
                else : pass

            from supy.utils.fractions import componentSolver,drawComponentSolver
            cs = componentSolver(observed, templates, 1e4, base = np.sum(bases, axis=0) )

            def replaceAll(label, pairs) : return replaceAll(label.replace(*pairs[0]),pairs[1:]) if pairs else label
            replacements = [ ("top.ttj_%s.wQQ.pu.sf"%tt,"q#bar{q}#to^{}t#bar{t}"),
                             ("top.ttj_%s.wQG.pu.sf"%tt,"qg#to^{}t#bar{t}"),
                             ("top.ttj_%s.wAG.pu.sf"%tt,"#bar{q}g#to^{}t#bar{t}"),
                             ("top.ttj_%s.wGG.pu.sf"%tt,"gg#to^{}t#bar{t}"),
                             ("QCD.Data 2012","Multijet"),
                             ("top.W","W+jets"),
                             ('top.',"") ]
            stuff = drawComponentSolver( cs, mfCanvas, distName = dist,
                                         templateNames = [replaceAll(t,replacements) for t in templateSamples])
            supy.utils.tCanvasPrintPdf( mfCanvas, mfFileName, verbose = False)
            with open(mfFileName+'.txt','a') as file : print >> file, "\n",dist+"\n", cs
            return distTup,cs

        def mf2(dist) : return measureFractions(dist,1)

        distTup,cs = map(mf2,["KarlsruheDiscriminant","TridiscriminantWTopQCD"][1:])[-1]

        iTT = next(i for i,ss in enumerate(org.samples) if ss['name']=='top.t#bar{t}')
        nTT = distTup[iTT].Integral(0,distTup[iTT].GetNbinsX()+1)
        fractions = dict(zip(templateSamples,cs.fractions))
        for iSample,ss in enumerate(org.samples) :
            if ss['name'] in fractions :
                f = fractions[ss['name']]
                n = distTup[iSample].Integral(0,distTup[iSample].GetNbinsX()+1)
            elif ss['name'] in ['top.ttj_%s.%s.pu'%(tt,s) for s in ['wQQ','wQG','wAG','wGG']] :
                f = fractions['top.t#bar{t}']
                n = nTT
            else : continue
            org.scaleOneRaw(iSample, f * sum(cs.observed) / n )

        supy.utils.tCanvasPrintPdf( mfCanvas, mfFileName, option = ']')

        org.mergeSamples( targetSpec={"name":"S.M.", "color":r.kGreen+2}, sources=(templateSamples+baseSamples),
                          keepSources=True, force=True)


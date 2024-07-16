from supy import wrappedChain,utils
from calculables.other import ScaleFactors
import math, ROOT as r, numpy as np

class Indices(wrappedChain.calculable) :
    def __init__(self, collection = None, ptMin = 26, absEtaMax = 2.1, ) :
        self.fixes = collection
        for item in ['ptMin','absEtaMax'] : setattr(self,item,eval(item))
        self.stashed = ['P4','SignalID']
        self.stash(self.stashed)
        self.moreName = 'pt>%.1f; |eta|<%.1f; SignalID'%(ptMin, absEtaMax)

    def accept(self, p4, id) : return all([id,
                                           p4.pt() > self.ptMin,
                                           abs(p4.eta()) < self.absEtaMax])

    def update(self,_) :
        self.value = [i for i,a in enumerate(utils.hackMap(self.accept,
                                                           *[self.source[getattr(self,item)] for item in self.stashed]))
                      if a]

class SignalID(wrappedChain.calculable) :
    def __init__(self, collection = None ) :
        self.fixes = collection
        self.stashed = ['IsPFMuon',       'IsGlobalMuon',  'NormChi2',
                        'TrackingLayers', 'ValidMuonHits', 'Db',#Dxy
                        'Dz',             'ValidPixelHits','MatchedStations']
        self.stash(self.stashed)

    @staticmethod
    def passID(isPF, isGlobal, chi2n, ntrack, nmu, db, dz, npx, nstat ):
        return all([isPF, isGlobal, chi2n < 10, ntrack > 5, nmu > 0,
                    abs(db) < 0.2, dz < 0.5, npx > 0, nstat > 1])

    def update(self,_) :
        self.value = utils.hackMap(self.passID,
                                   *[self.source[getattr(self,item)] for item in self.stashed])


class TriggerScaleFactors(ScaleFactors):
    '''Scale factors (eff_data / eff_mc) for HLT_IsoMu24_eta2p1.'''
    def __init__(self, collection=None):
        self.fixes = collection
        self.stash(['P4','Indices'])
        self.setUp()

    def setUp(self):
        pattern ='DATA_over_MC_IsoMu24_eta2p1_TightIso_pt_abseta'
        absEtaBins = [('<0.9',(0,0.9)),
                      ('0.9-1.2',(0.9,1.2)),
                      ('1.2-2.1',(1.2,2.1))]
        columns = set()
        lumiSum = 0.
        central = np.array(3*[8*[0]])
        deltaUp2 = np.array(3*[8*[0]])
        deltaDn2 = np.array(3*[8*[0]])

        epochs = [('A',  891, 'data/MuonEfficiencies_Run_2012A_2012B_53X.root', True),
                  ('B', 4400, 'data/MuonEfficiencies_Run_2012A_2012B_53X.root', True),
                  ('C', 7020, 'data/MuonEfficiencies_Run_2012C_53X.root', False),
                  ('D', 7273, 'data/TriggerMuonEfficiencies_Run_2012D_53X.root',True)]
        for run,lumi,fileName,appendRun in epochs:
            tfile = r.TFile.Open(fileName)
            graphs = [tfile.Get(pattern + ebin[0] + ('_2012%s'%run if appendRun else '')) for ebin in absEtaBins]
            for g in graphs:
                columns.add(tuple([(g.GetX()[i] - g.GetErrorXlow(i), g.GetX()[i] + g.GetErrorXhigh(i)) for i in range(g.GetN())]))
            lumiSum += lumi
            central += lumi * np.array([[g.GetY()[i] for i in range(g.GetN())] for g in graphs])
            deltaUp2 += lumi * np.array([[g.GetErrorYhigh(i)**2 for i in range(g.GetN())] for g in graphs])
            deltaDn2 += lumi * np.array([[g.GetErrorYlow(i)**2 for i in range(g.GetN())] for g in graphs])
            tfile.Close()

        sys = 0.002
        self.central = central / lumiSum
        self.deltaUp = np.sqrt(deltaUp2 / lumiSum + sys*sys)
        self.deltaDn = -np.sqrt(deltaDn2 / lumiSum + sys*sys)

        assert len(columns) == 1
        self.rows = [e[1] for e in absEtaBins] # |eta|
        self.columns = list(list(columns)[0]) # pt


class SelectionScaleFactors(ScaleFactors):
    '''Scale factors (eff_data / eff_mc) for single muon selection.'''
    def __init__(self, collection=None):
        self.fixes = collection
        self.stash(['P4','Indices'])

        fileName = 'data/Muon_ID_iso_Efficiencies_Run_2012ABCD_53X.root'
        pattern = "DATA_over_MC_combRelIsoPF04dBeta<012_Tight_pt_abseta%s_2012ABCD"
        tfile = r.TFile.Open(fileName)

        absEtaBins = [('<0.9',(0,0.9)),
                      ('0.9-1.2',(0.9,1.2)),
                      ('1.2-2.1',(1.2,2.1))]
        columns = set()
        graphs = [tfile.Get(pattern % ebin[0]) for ebin in absEtaBins]
        for g in graphs:
            columns.add(tuple([(g.GetX()[i] - g.GetErrorXlow(i), g.GetX()[i] + g.GetErrorXhigh(i)) for i in range(g.GetN())]))
        sys = math.sqrt(0.005**2+0.002**2)
        self.central = np.array([[g.GetY()[i] for i in range(g.GetN())] for g in graphs])
        self.deltaUp = np.array([[math.sqrt(g.GetErrorYhigh(i)**2+sys**2) for i in range(g.GetN())] for g in graphs])
        self.deltaDn = np.array([[-math.sqrt(g.GetErrorYlow(i)**2+sys**2) for i in range(g.GetN())] for g in graphs])
        tfile.Close()

        assert len(columns) == 1
        self.rows = [e[1] for e in absEtaBins] # |eta|
        self.columns = list(list(columns)[0]) # pt


class SF(wrappedChain.calculable):
    def __init__(self, collection=None):
        self.fixes = collection
        self.stash(['TriggerScaleFactors','SelectionScaleFactors'])

    def update(self,_):
        self.value = self.source[self.TriggerScaleFactors][0] * self.source[self.SelectionScaleFactors][0]


class Reweights(wrappedChain.calculable):
    def __init__(self, collection=None):
        self.fixes = collection
        self.stash(['TriggerScaleFactors','SelectionScaleFactors'])

    def update(self,_):
        tSF = self.source[self.TriggerScaleFactors]
        sSF = self.source[self.SelectionScaleFactors]
        self.value = [sf/tSF[0] for sf in tSF[1:]] + [sf/sSF[0] for sf in sSF[1:]]

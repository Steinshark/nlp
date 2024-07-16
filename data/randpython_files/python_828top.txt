import math,itertools,ROOT as r
from supy import analysisStep,steps,calculables,utils
try:
    import numpy as np
    import scipy.stats
except:  pass

#####################################

class channelClassification(analysisStep) :
    labels = ['','ee','mm','tt','em','et','mt','ej','mj','tj','jj']
    nbins = len(labels)

    def uponAcceptance(self,ev) :
        iBin = self.labels.index(ev['ttDecayMode'])
        if iBin :
            self.book.fill( iBin, 'ttDecayMode', self.nbins, 0, self.nbins, xAxisLabels = self.labels )

#####################################
class kinFitLook(analysisStep) :
    def __init__(self,indexName) : self.moreName = indexName
    def uponAcceptance(self,ev) :
        index = ev[self.moreName]
        if index<0 : return
        topReco = ev["TopReconstruction"][index]
        residuals = topReco["residuals"]
        lepTopM = topReco['lepTopP4'].M()
        hadTopM = topReco['hadTopP4'].M()

        for name,val in residuals.iteritems() :
            self.book.fill(val, "residual_%s"%name+self.moreName, 50,-7,7, title = ";residual %s;events / bin"%name)
            self.book.fill(scipy.stats.norm.cdf(val), "residualCDF_%s"%name+self.moreName, 100,0,1, title = ";cdf(residual) %s;events / bin"%name)

        self.book.fill( (lepTopM, hadTopM), "topMassesFit"+self.moreName, (100,100),(0,0),(300,300),
                        title = ";fit mass_{top} (leptonic); fit mass_{top} (hadronic);events / bin",)
        
        self.book.fill( topReco['chi2'], "topReco_Chi2"+self.moreName, 50, 0 , 100, title = ';ttbar kin. fit #chi^{2};events / bin')
#####################################

class combinatorialFrequency(analysisStep) :
    def uponAcceptance(self,ev) :
        if not ev['genTopTTbar'] : return
        recos = ev['TopReconstruction']
        iP,iQ,iH,iL = ev['IndicesGenTopPQHL']

        igen=ev['genTopRecoIndex']

        iGenHadB = next((i for i,R in enumerate(recos) if R['iPQHL'][2]==iH) , -1)
        iGenLepB = next((i for i,R in enumerate(recos) if R['iPQHL'][3]==iL) , -1)
        iGenHadPQB  = next((i for i,R in enumerate(recos) if R['iPQHL'][:3]==(iP,iQ,iH)), -1 )
        iGenJets = next((i for i,R in enumerate(recos) if R['iPQHL']==(iP,iQ,iH,iL)), -1)

        for i,item in enumerate(['iGenHadB','iGenLepB','iGenHadPQB','iGenJets','igen']) :
            self.book.fill(eval(item), str(i)+item, 10,-1.5,8.5, title=";%s;events / bin"%item )
#####################################
class kinematics(analysisStep) :
    def __init__(self,indexName) : self.moreName = indexName
    def uponAcceptance(self,ev) :
        index = ev["%sRecoIndex"%self.moreName]
        if index < 0 : return
        topReco = ev["TopReconstruction"]
        self.book.fill( topReco[index]['hadTraw'].mass(), "rawHadTopMass", 100, 100,300, title = ";%s raw hadronic top mass;events / bin"%self.moreName)
        self.book.fill( topReco[index]['hadWraw'].mass(), "rawHadWMass", 100, 0,200, title = ";%s raw hadronic W mass;events / bin"%self.moreName)

        i5 = ev['fitTopFifthJetIndex']
        triD = ev['TridiscriminantWTopQCD']
        if i5!=None:
            up = 0.11
            v = max(0, min(up-1e-8, ev['Moments2Sum'.join(ev['TopJets'])][i5]))
            mass = '_loM' if ev['fitTopMassSum']<450 else '_hiM'
            rap = '_loY' if ev['fitTopRapiditySum']<0.5 else '_hiY'
            for item in ['',mass,rap]:
                self.book.fill( (v,triD), 'Moments2Sum_triD'+item, (50,5), (0,-1), (up,1))
#####################################
class kinematics3D(analysisStep) :
    def __init__(self,indexName) : self.moreName = indexName
    def uponAcceptance(self,ev) :
        index = ev["%sRecoIndex"%self.moreName]
        if index < 0 : return
        topReco = ev["TopReconstruction"]
        for low,var in  zip([0,0],["fitTopPtOverSumPt", "fitTopTanhRapiditySum"]) :
            v = tuple( max(L,min(val,U-1e-6)) for val,L,U  in zip((ev[var],ev['TridiscriminantWTopQCD']), (low,-1), (1,1)) )
            self.book.fill( v, '%s_triD'%var, (50,5), (low,-1), (1,1))
#####################################
class resolutions(analysisStep) :
    def __init__(self,indexName) : self.moreName = indexName
    def uponAcceptance(self,ev) :
        genTTbar = ev["genTopTTbar"]
        if not genTTbar : return

        topReco = ev["TopReconstruction"]

        index = ev[self.moreName]
        self.book.fill(index, '0_'+self.moreName, 21, -1.5, 19.5, title = ';%s;events / bin'%self.moreName)
        if index<0 : return
        
        if ev['genTTbarIndices']['semi'] :
            for s in ['lep','nu','bLep','bHad','q'] :
                self.book.fill(ev['%sDeltaRTopRecoGen'%s][index], '1_'+s+'DeltaRTopRecoGen', 50,0,2, title = ';%s DeltaR reco gen;events / bin'%s)
        
        gsP4 = ev['genSumP4']
        self.book.fill( topReco[index]['ttx'].pt() - gsP4.pt(), "resolution_pt", 100, -100, 100, title = ";%s #Delta_{reco-gen} ttx.pt;events / bin"%self.moreName )
        self.book.fill( topReco[index]['ttx'].mass() - gsP4.mass(), "resolution_m", 100, -200, 200, title = ";%s #Delta_{reco-gen} ttx.m;events / bin"%self.moreName )

        iLep = min(0,topReco[index]["lepCharge"])
        gen =  (ev["genP4"][genTTbar[0]], ev["genP4"][genTTbar[1]])
        reco = (topReco[index]['top'],topReco[index]['tbar'])
        unfit = (topReco[index]['lepTraw'], topReco[index]['hadTraw'])[::topReco[index]["lepCharge"]]

        for func in ['Rapidity'] :
            genFunc = (getattr(gen[0],func)(), getattr(gen[1],func)())
            recoFunc = (getattr(reco[0],func)(),getattr(reco[1],func)())
            unfitFunc = (getattr(unfit[0],func)(), getattr(unfit[1],func)())
            for f,fit in [('fit',recoFunc),('unfit',unfitFunc)] :
                self.book.fill( fit[iLep]     - genFunc[iLep],       "d%sLepTop_%s"%(func,f), 100,-1,1, title=";lep top #Delta %s_{%s reco gen};events / bin"%(func,f))
                self.book.fill( fit[not iLep] - genFunc[not iLep],   "d%sHadTop_%s"%(func,f), 100,-1,1, title=";had top #Delta %s_{%s reco gen};events / bin"%(func,f))
                self.book.fill( fit[0]-fit[1] - (genFunc[0]-genFunc[1]), "dd%sTTbar_%s"%(func,f), 100,-1,1, title = ";#Delta %s_{t#bar{t} %s reco}-#Delta %s_{t#bar{t} gen};events / bin"%(func,f,func))

        def XL(t,tbar):
            return math.tanh(abs(t.Rapidity())-abs(tbar.Rapidity()))
        def XT(t,tbar):
            phiPM = math.acos(math.cos( (t+tbar).phi() -
                                        (t-tbar).phi()))
            return -1 + 2 * phiPM / math.pi
        for func in ['XL','XT']:
            genFunc = eval(func)(*gen)
            recoFunc = eval(func)(*reco)
            unfitFunc = eval(func)(*unfit)
            self.book.fill( recoFunc - genFunc, func+"_reco", 200, -2, 2, title = ";%s^{rec}-%s^{gen}"%(func,func))
            self.book.fill( unfitFunc- genFunc, func+"_unfit", 200, -2, 2, title = ";%s^{unfit}-%s^{gen}"%(func,func))

class kinematic_resolution(calculables.secondary):
    def __init__(self, parBins, samples, tag):
        self.parBins = parBins # dict of 'var:(N,lo,hi,pivot)'
        self.samples = samples
        self.tag = tag

    def organize(self,org):
        org.mergeSamples(targetSpec={'name':'merged'}, sources=self.samples)

    def baseSamples(self): return self.samples

    def uponAcceptance(self,ev):
        for var,(N,lo,hi,_) in self.parBins.items():
            gen = min(hi, max(lo, ev['genTop' + var]))
            fit = min(hi, max(lo, ev['fitTop' + var]))
            self.book.fill((gen,fit), var, (N,N), (lo,lo), (hi,hi),
                           title=';genTop%s;fitTop%s;events / bin'%(var,var))

    def reportCache(self):
        fileName = '/'.join(self.outputFileName.split('/')[:-1]+[self.name])
        c = r.TCanvas()
        utils.tCanvasPrintPdf(c, fileName, False, '[')
        cache = self.fromCache(['merged'],self.parBins.keys(), self.tag)['merged']
        for par,hist in cache.items():
            pivot = self.parBins[par][3]
            lo = hist.ProjectionY(hist.GetName()+'_lo', 0, hist.GetXaxis().FindFixBin(pivot)-1)
            hi = hist.ProjectionY(hist.GetName()+'_hi', hist.GetXaxis().FindFixBin(pivot), -1)
            tot = hist.ProjectionY()
            lo.Divide(tot)
            hi.Divide(tot)
            tot.Scale(1./tot.Integral())
            lo.SetLineColor(r.kBlue)
            hi.SetLineColor(r.kRed)
            lo.SetMinimum(0)
            lo.SetMaximum(1)
            tot.SetFillColor(17)
            tot.SetFillStyle(1001)
            l = r.TLine(pivot,0,pivot,1)
            l.SetLineColor(r.kGreen)
            lo.Draw('hist')
            hi.Draw('hist same')
            tot.Draw('hist same')
            l.Draw()
            utils.tCanvasPrintPdf(c, fileName, False)
        utils.tCanvasPrintPdf(c, fileName, True, ']')


class chosenCombo(analysisStep) :
    def uponAcceptance(self,ev):
        TCL = ev['TopCandidateLikelihood']
        iPQHL = max(TCL, key=TCL.__getitem__)
        iQQBB = iPQHL[:2] + tuple(sorted(iPQHL[2:]))

        self.book.fill(ev['HTopSigmasPQB'][iPQHL[:3]], 'MSD', 100, 0, 5, title=";MSD;events / bin")
        self.book.fill(ev['LTopUnfitSqrtChi2'][iPQHL[3]], 'chia', 100, 0, 10, title=";#chi_{a};events / bin")
        self.book.fill(ev['TopComboQQBBLikelihoodRatio'][iQQBB], 'LiCSV', 100, 0, 1, title=";L_i^{CSV}/max(L^{CSV});events / bin")


class signalhists(analysisStep):
    def __init__(self,doGen):
        self.doGen = doGen
        self.vars = ('genTopQueuedBin5' if doGen else 'fitTopQueuedBin5','fitTopQueuedBin5','TridiscriminantWTopQCD')
        self.nbins = (25,25,5)
        self.lo = (-1,-1,-1)
        self.hi = (1,1,1)
        
    def uponAcceptance(self,ev):
        vals = tuple([ev[item] for item in self.vars])
        zipped = zip(vals,self.vars,self.nbins,self.lo,self.hi)
        self.book.fill(*zip(*zipped[1:]))
        self.book.fill((ev['fitTopTanhDeltaAbsY'],ev['TridiscriminantWTopQCD']),'fitTopTanhDeltaAbsY_TridiscriminantWTopQCD', (25,5), (-1,-1), (1,1))
        self.book.fill((ev['fitTopDPtDPhi'],      ev['TridiscriminantWTopQCD']),'fitTopDPtDPhi_TridiscriminantWTopQCD',       (25,5), (-1,-1), (1,1))
        self.book.fill((ev['fitTopTanhDeltaAbsY'],ev['TridiscriminantWTopQCD']),'fitTopTanhDeltaAbsY_TridiscriminantWTopQCD13', (13,5), (-1,-1), (1,1))
        self.book.fill((ev['fitTopDPtDPhi'],      ev['TridiscriminantWTopQCD']),'fitTopDPtDPhi_TridiscriminantWTopQCD13',       (13,5), (-1,-1), (1,1))
        if ev['genTopTTbar']:
            self.book.fill((ev['genTopTanhDeltaAbsY'],ev['fitTopTanhDeltaAbsY']), 'genTopTanhDeltaAbsY_fitTopTanhDeltaAbsY', (2,2), (-1,-1), (1,1))
            self.book.fill((ev['genTopDPtDPhi'],ev['fitTopDPtDPhi']), 'genTopDPtDPhi_fitTopDPtDPhi', (2,2), (-1,-1), (1,1))
        if self.doGen: self.book.fill(*zip(*zipped))

        # higher resolution tridiscriminant
        zipped[2] = list(zipped[2])
        zipped[2][2] *= 5
        self.book.fill(*zipped[2])
        

# Author: YongJin H. (Twitter: @chromianbiomath)
# Date:   Oct. 12th. 2020
# Project: Nested Library Analysis
# ______________________________________________

from sys   import path
path.append('../CyNLA')
from CyNLA import *
from CyNLA import __product as product
from seaborn import heatmap, kdeplot, histplot, color_palette
import matplotlib.pyplot    as plt
import matplotlib as mpl; mpl.use('tkagg') ## for Mac
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
# R functions
from rpy2.robjects.packages import importr
detectBatch = importr('cpm').detectChangePointBatch
rdollar     = importr('base').__dict__['$']
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
import os, string
# ______________________________________________

def Run_and_Return( SEED, T_c, s_PE, s_ME, SKIPSTEP, JustSystem = False, normalize = True, method = "SMap"):
    np.random.seed(SEED)
    initial_a    = .375
    final_a      = .625
        
    # Stablization for Initial state
    Initial_X   = Simulator(lambda t, X: dx.AH3SP_Lake(X, a = initial_a),
                            initial = np.array([.6, .85, .15, 8.5]), t_eval = np.arange(0, 250, .25),
                            prob_PE = .01, sigm_PE = .1)[-1,:]
    # Start generating the non-stationary system
    AH3SP       = Simulator(lambda t, X: dx.AH3SP_Lake(X, a = initial_a + (final_a - initial_a)*(t/5000)),
                            initial = Initial_X, t_eval = np.arange(0, 5000, .25),
                            prob_PE = .01, sigm_PE = s_PE)[::7, :]
    # Find the change point of the external factor
    changepoint = np.argmax( np.gradient(AH3SP[:, 0])**2 )
    # Adjust the location of change point for purpose of demonstrations
    AH3SP = AH3SP[ (changepoint - T_c)::1, :][:1000, :]
    # Normalizing followed by adding measurement noises
    MEAN  = np.mean(AH3SP, axis = 0)
    STD   = np.std(AH3SP, axis = 0)
    AH3SP =  ( (AH3SP - MEAN) / STD )
    AH3SP[:, 1:]    = AH3SP[:, 1:] + np.random.normal(0, s_ME, AH3SP[:, 1:].shape)
    if (not normalize):
        AH3SP = AH3SP * STD + MEAN

    if JustSystem:
        return AH3SP
    else:
        # Set the NestLibraryAnalysis Object
        nlaObj = nla(AH3SP[:,3:])
        # Run NLA
        nlares = nlaObj.run(FlowRef = "751 1000", LargestLib = "1 750", SmallestLib = "501 750",
                            skipstep = SKIPSTEP, numThreads = 6, method = method)
        rawS, tilS, trdS = nla.SignalProcessing(nlares, SIGMA = 1)
        rawS        = rawS.unstack()
        rawS.name   = str(SEED)
        return rawS
    # End ===============

def CUR(# Construct, Update, and Return
        SEED,
        T_c            = 250,
        s_ProcessError = .05,
        s_MeasureError = .05,
        SKIPSTEP       =  10,
        method         = "SMap"):

    Location = ""
    Filename = "Tc{:d}sPE{:d}sME{:d}SKPSTP{:d}.txt".format( T_c, int(100*s_ProcessError), int(100*s_MeasureError), SKIPSTEP )

    if not (SEED is None):     
        try:
            ORI = pd.read_csv(Location + Filename, index_col = [0, 1])
        except FileNotFoundError:
            rawS    = Run_and_Return( SEED, T_c, s_ProcessError, s_MeasureError, SKIPSTEP, method = method)
            # Construct
            pd.DataFrame(rawS).to_csv(Location + Filename)
            ORI = pd.read_csv(Location + Filename, index_col = [0, 1])    
        if not (str(SEED) in ORI.columns):
            rawS    = Run_and_Return( SEED, T_c, s_ProcessError, s_MeasureError, SKIPSTEP , method = method)
            pd.concat([ORI, rawS], axis = 1).to_csv(Location + Filename)
        else:
            rawS    = ORI[str(SEED)]
        return rawS 

    else:
        return pd.read_csv(Location + Filename, index_col = [0, 1])
    # End ===============

def Visualize( SEED,
               T_c            = 250,
               s_ProcessError = .05,
               s_MeasureError = .05,
               SKIPSTEP       =  10,
               SIGMA          = 2.5,
               method         = "SMap"):
    rawS = CUR( SEED, T_c, s_ProcessError, s_MeasureError, SKIPSTEP ).unstack().T
    rawS, tilS, trdS = nla.Signal_from_rawS(rawS, SIGMA)
    AH3SP = Run_and_Return( SEED, T_c, s_ProcessError, s_MeasureError, SKIPSTEP, JustSystem = True, method = method)
    # 
    if 1:
        # plot
        fig, [ax0, ax] = plt.subplots(2, 1, figsize=(10, 6))
        ax0.plot(AH3SP[:,2], alpha = .9, color = 'k', linewidth = .8, label = 'observation')
        ax0.set_ylabel(f"observed variable $y$")
        ax0.axvspan(751, 1000, color = 'g', alpha = .2)
        tax0 = ax0.twinx()
        tax0.plot(AH3SP[:,0], alpha = .4, color = 'r', label = 'regime')
        tax0.set_ylabel(f"unobserved regime indicator $N$", color = 'red')
        tax0.tick_params(axis='y', colors='red')
        tax = ax.twinx()
        trdS.index   -= 1
        trdS.columns -= 1
        # heatmap
        _ = heatmap(trdS.apply(np.sign).T,
                    cmap = "bone",
                    linecolor = 'none',
                    cbar = False,
                    alpha = .4,
                    vmin = -1.5, vmax = 1.5,
                    ax = tax,
                    linewidths = -1)
        _ = tax.set_yticklabels([])
        _ = tax.set_yticks([])
        _ = tax.set_ylabel("")
        tax.xaxis.set_ticks_position("top")
        tax.xaxis.set_label_position("top")
        for T in rawS:
            _ = ax.plot(np.arange(len(rawS)) + .5, rawS[T].values, color = 'g',
                        marker = 'o', lw = 0, markersize = 3, mfc = 'none', alpha = .05, label = "")
            _ = ax.plot(np.arange(len(tilS)) + .5, tilS[T].values, color = 'g',
                        alpha = .1, label = "")
        #_ = ax.plot(np.arange(len(rawS)) + .5, np.diag(rawS), color = 'g', marker = 'o', lw = 0, markersize = 3, label = "")
        _ = tax.plot([], [], color = 'g', marker = 'o', lw = 0, markersize = 3, mfc = 'none', alpha = 1, label = r"$\mathtt{RMSE}$")
        _ = tax.plot([], [], color = 'g', alpha = 1, label = r"$\widetilde{\ \,\mathtt{RMSE}}$")
        _ = ax.set_xlabel("time")
        tax.legend(loc = "upper right")
        ax0.set_xlim(-20, 1020)
        tax.set_xlim(-20/SKIPSTEP, 1020/SKIPSTEP)
        _ = tax.set_xticklabels(tax.get_xticklabels()[::5])
        _ = tax.set_xticks(np.array(tax.get_xticks()[::5]))
        fig.set_tight_layout(True)
        plt.show()
        return fig
    # End ===============

def detectCP(series, _cpmType = 'Student', valley_check = True):
    series = series.dropna()
    base   = series.values.tolist()
    if len(base) < 10: # too short for R package 'cpm'
        return np.nan
    try:
        cpmres = detectBatch( base, cpmType = _cpmType, alpha = 0.001)
    except:
        return np.nan
    if rdollar(cpmres, 'changeDetected')[0]:
        changepoint = rdollar(cpmres, 'changePoint')[0]+1
        if valley_check:    # || for NLA we would like to pulse the alarm when there is a valley shape in RMSE curve
            if series[:changepoint].mean() >= series[changepoint:].mean():
                return np.nan
            else:
                return series.index[changepoint-1]
        else:               # || otherwise, for CPMs, we simply report the output
            return series.index[changepoint-1] # minus 1 because R counts from 1 rather than 0
    else:
        return np.nan

def summary( T_c, s_PE, s_ME, SKIPSTEP, SIGMA ):
    Location = ""
    Filename = "Tc{:d}sPE{:d}sME{:d}SKPSTP{:d}.txt".format( T_c, int(100*s_PE), int(100*s_ME), SKIPSTEP )
    try:
        DF = pd.read_csv(Location + Filename, index_col = [0, 1])
    except:
        return None
    resTable = {}
    for SEED in DF.columns:
        signS          = nla.Signal_from_rawS( DF[SEED].unstack().T, SIGMA )[-1].apply(np.sign)
        resTable[SEED] = signS.apply(detectCP).median() # mean or median?
    return resTable

if __name__ == "__main__":
    trueTc = 300
    # -- Figure 0
    for SEED in np.arange(4000, 4200):
        s_PE = 0.05 # strength of process error
        for s_ME in np.arange(0.05, 0.05 + np.finfo(float).eps, 0.05).round(2):
            _ = CUR( SEED, T_c = trueTc, s_ProcessError = s_PE, s_MeasureError = s_ME )


if __name__ == "__main__":
    trueTc = 300
    s_ME   = 0.05
    # -- Figure 0
    simu = pd.DataFrame(Run_and_Return( 4040, trueTc, 0.05, s_ME, 10, True, False), columns = ['N', 'x', 'y', 'z'])
    simu['a'] = np.linspace(.375, .625, 2858)[1562-trueTc:1562+(1000-trueTc)]
    fig, axes = plt.subplots(2, 1, figsize = (10, 2.5), tight_layout = True, sharex = True)
    taxes = [ax.twinx() for ax in axes]
    _ = taxes[0].plot(simu.N, 'k-', lw = 1); _ = axes[0].set_yticks([]); _ = axes[0].set_yticklabels([]);
    _ = axes[0].set_ylabel('N', rotation = 0); axes[0].yaxis.set_label_coords(-.025, .5)
    _ = taxes[1].plot(simu.a, 'k-', lw = 1); _ = axes[1].set_yticks([])
    _ = axes[1].set_ylabel('a', rotation = 0); axes[1].yaxis.set_label_coords(-.025, .5)
    _ = axes[1].set_xlabel('time')
    fig.tight_layout()
    plt.show()
    for SEED in np.arange(4000, 4200):
        s_PE = 0.05 # strength of process error
        for s_ME in np.arange(s_ME, s_ME + np.finfo(float).eps, 0.05).round(2):
            _ = CUR( SEED, T_c = trueTc, s_ProcessError = s_PE, s_MeasureError = s_ME )
    ###############################################################
    # for thesis # -- Figure 1
    simu = pd.DataFrame(Run_and_Return( 4040, trueTc, 0.05, s_ME, 10, True),
                        columns = ['N', 'x', 'y', 'z'])
    fig, axes = plt.subplots(4, 1, figsize = (10, 6), tight_layout = True, sharex = True)
    for i in range(4):
        _ = axes[i].plot(simu[simu.columns[-(i+1)%4]], 'k-', lw = 1)
        _ = axes[i].set_ylabel(simu.columns[-(i+1)%4], rotation=0)
    else:
        ax2 = axes[i].twinx()
        _ = ax2.plot( np.linspace(.375, .625, len(simu)), 'r--', lw = .8,alpha = .7)
        _ = ax2.set_yticklabels(ax2.get_yticklabels(), color = 'red')
        _ = ax2.set_ylabel('a', color = 'red', rotation=0)
        _ = axes[i].set_xlabel("time")
    fig.tight_layout()
    plt.show()
    # for thesis # -- Figure 2
    fig = Visualize(4040, trueTc, 0.05, s_ME, 10, 3)
    plt.show()
    # for thesis # -- Figure 3
    DFYs    = pd.DataFrame(np.array([ Run_and_Return(seed, trueTc, 0.05, s_ME, 10, True)[:500, 2] for seed in np.arange(4000, 4200) ]).T,
                           columns = np.arange(4000, 4200))
    NLAres  = pd.Series(summary( trueTc, .05, s_ME, 10, 3 )).values
    ComprDF = pd.DataFrame({"NLA": pd.Series(NLAres, index = np.arange(4000, 4200)),
                            "Student": DFYs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Student', valley_check = False)),
                            "Bartlett": DFYs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Bartlett', valley_check = False)),
                            "Mann-Whitney": DFYs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Mann-Whitney', valley_check = False)),
                            "Kolmogorov-Smirnov": DFYs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Kolmogorov-Smirnov', valley_check = False))})
    fig, axes = plt.subplots(1, 2, figsize = (12, 6), tight_layout = True, sharex = True)
    chromianpalette = [color_palette("Paired")[i] for i in [3, 11, 9, 7, 1]]
    for ax in axes:
        ax.set_xlabel(r"Change point $\widehat{\tau}$ detected from $y$")
        ax.axvline(300, color = 'red', linewidth = .75, linestyle = '--', alpha = .5)
    axes[1].grid(color = 'black', alpha = .25, linewidth = .3)
    PLT0 = histplot(data = ComprDF, element="step", binwidth = 10, alpha = .15, linewidth = .25,
                    kde = True, kde_kws = {"bw_adjust": .5}, line_kws = {"linewidth": 1.2},
                    fill = True, ax = axes[0], legend = True, palette = chromianpalette)
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, int(1.1 * axes[0].get_ylim()[1]))
    PLT1 = histplot(data = ComprDF, element="step", binwidth = 10, alpha = .8, linewidth = 1, kde = False,
                    cumulative = True, fill = False, ax = axes[1], legend = False, palette = chromianpalette)
    axes[1].set_ylabel("Cumulative count")
    axes[0].set_xlim(0, 500)
    fig.savefig("ComprCPM_onY.png", dpi = 600)
    plt.show()
    # for thesis # -- Figure 4
    data = pd.read_csv("data.csv", header = 1, index_col = [0,])
    series0  = data.Value
    series0.index = pd.to_datetime(series0.index, format='%Y%m')
    series   = series0.rolling(12).mean().shift(-6).dropna()
    series   = series['1875':'2005']
    # ####
    import nolds
    print("The Hurst exponent:\n\t{:f}".format(nolds.hurst_rs(series)) )
    print("The estimate alpha for the Hurst parameter:\n\t{:f}".format(nolds.dfa(series)))
    emb_dims = pd.Series(range(1, 16), name = 'dimension')
    lypn = emb_dims.apply(lambda E: nolds.lyap_r(series, emb_dim = E, min_tsep = 400))
    lypn.index = emb_dims
    lypn.name  = 'Largest Lyapunov exponent'
    print("The estimate largest Lyapunov exponent:")
    print(lypn)
    # ####
    nlaObj   = nla(series.values.reshape(-1, 1))
    ShrkFrom = series.index.get_loc('2000-12-01') + 1
    ShrkUntl = series.index.get_loc('1950-01-01') + 1
    ReferTo  = series.index.get_loc('1920-01-01')
    nlares   = nlaObj.run(FlowRef = "1 {:d}".format(ReferTo),
                          LargestLib = "{:d} {:d}".format(ReferTo, ShrkFrom),
                          SmallestLib = "{:d} {:d}".format(ReferTo, ShrkUntl),
                          skipstep = 12,
                          numThreads = 4)
    rawS, tilS, trdS = nla.SignalProcessing(nlares, SIGMA = 2, FromRight = True)
    Index = np.arange( -(ShrkUntl//12) +(ShrkFrom//12) )
    # ####
    fig, [ax0, ax1] = plt.subplots(2, 1, sharex = False, figsize = (10, 5), tight_layout = True)
    ax0.plot(series0[series.index[0]:'2000-12-01'], 'k-', lw = .2, alpha = .6)
    ax0.plot(series[:'2000-12-01'], 'k-', lw = 1, label = "1-year moving average")
    ax0.axvspan(series.index[0], series.index[ReferTo], color = 'green', alpha = .3)
    ax0.legend()
    ax0.set_title('Monthly PDO index')
    TtrdS = pd.DataFrame(trdS.values, index = Index, columns = Index)
    changepoint = np.sign(TtrdS).apply(detectCP).dropna().values
    tax1  = ax1.twinx()
    Triangle = pd.DataFrame(trdS.apply(np.sign))
    Triangle.index = Index
    _ = heatmap(Triangle.T,
                cmap = "bone",
                linecolor = 'none',
                cbar = False,
                alpha = .2,
                vmin = -1.5, vmax = 1.5,
                ax = tax1,
                linewidths = -1)
    for LB in list(rawS):
        _ = ax1.plot( Index + .5, rawS[LB], color = 'g', marker = 'o', lw = 0, markersize = 3, mfc = 'none', alpha = .05, label = "" )
        _ = ax1.plot( Index + .5, tilS[LB], color = 'g', lw = 0.6, alpha = .05, label = "" )
    ax1.set_xlim(0, ShrkFrom // 12)
    _ = ax1.set_xticklabels(series.index.year[rawS.index[::10]])
    _ = ax1.set_xticks(np.array(ax1.get_xticks()[::10]))
    _ = tax1.set_yticks([])
    ax0.set_xlim(series.index[0] - np.timedelta64(6, 'Y'), series.index[ShrkFrom] + np.timedelta64(6, 'Y'))
    ax1.set_xlim(-487/6 +8/12, 57 +8/12)
    ax1.set_xlabel('time')
    tax1.set_ylabel(None)
    _ = tax1.plot([], [], color = 'g', marker = 'o', lw = 0, markersize = 3, mfc = 'none', label = "RMSE")
    _ = tax1.plot([], [], color = 'g', label = r"$\widetilde{\ \,\mathtt{RMSE}}$")
    tax1.legend(loc = 'upper left')
    plt.show()
    # for thesis # -- Figure 3-2
    # s_ME    = 0.05
    ComprDF = {}
    for s_ME in np.arange(0,  0.3+np.finfo(float).eps, 0.05).round(2):
        ComprDF[s_ME] = pd.Series(summary( trueTc, .05, s_ME, 10, 3 )).values
    ComprDF   = pd.DataFrame(ComprDF, columns = pd.Series(np.arange(0,  0.3+np.finfo(float).eps, 0.05).round(2), name = r"$\sigma^2$"))
    from seaborn import violinplot
    fig, axes = plt.subplots(1, 2, figsize = (8.5, 6.5), tight_layout = True, sharex = True)
    chromianpalette = color_palette("rainbow", len(ComprDF.T))
    for ax in axes:
        ax.grid(color = 'black', alpha = .25, linewidth = .3)
        ax.axvline(300, color = 'red', linewidth = .75, linestyle = '--', alpha = .5)
        ax.set_xlabel(r"Detected change point $\widehat{\tau}$")
    PLT0 = violinplot(data = ComprDF[ComprDF.columns[::-1]], ax = axes[0], legend = True, palette = chromianpalette, orient = 'h',
                      linewidth = 1.5, alpha = .75, inner = 'box')
    PLT1 = histplot(data = ComprDF[ComprDF.columns[::-1]], element="step", binwidth = 10, alpha = .8, linewidth = 1.2,
                    kde = False, kde_kws = {"bw_adjust": .5}, line_kws = {"linewidth": .75},
                    cumulative = True, fill = False, ax = axes[1], legend = True, palette = chromianpalette[::-1])
    axes[0].set_ylabel(r"Strength of measurement error ($\sigma^2$)")
    fig.savefig("ErrorRobustness.png", dpi = 600)
    plt.show()
    # for Appendix B
    os.chdir("./onX/")
    DFXs    = pd.DataFrame(np.array([ Run_and_Return(seed, trueTc, 0.05, s_ME, 10, True)[:500, 1] for seed in np.arange(4000, 4200) ]).T,
                           columns = np.arange(4000, 4200))
    NLAres  = pd.Series(summary( trueTc, .05, s_ME, 10, 3 )).values
    CompDFX = pd.DataFrame({"NLA": pd.Series(NLAres, index = np.arange(4000, 4200)),
                            "Student": DFXs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Student', valley_check = False)),
                            "Bartlett": DFXs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Bartlett', valley_check = False)),
                            "Mann-Whitney": DFXs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Mann-Whitney', valley_check = False)),
                            "Kolmogorov-Smirnov": DFXs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Kolmogorov-Smirnov', valley_check = False))})
    os.chdir("../")
    os.chdir("./onZ/")
    DFZs    = pd.DataFrame(np.array([ Run_and_Return(seed, trueTc, 0.05, s_ME, 10, True)[:500, 3] for seed in np.arange(4000, 4200) ]).T,
                           columns = np.arange(4000, 4200))
    NLAres  = pd.Series(summary( trueTc, .05, s_ME, 10, 3 )).values
    CompDFZ = pd.DataFrame({"NLA": pd.Series(NLAres, index = np.arange(4000, 4200)),
                            "Student": DFZs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Student', valley_check = False)),
                            "Bartlett": DFZs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Bartlett', valley_check = False)),
                            "Mann-Whitney": DFZs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Mann-Whitney', valley_check = False)),
                            "Kolmogorov-Smirnov": DFZs.apply(lambda res: detectCP(pd.Series(res), _cpmType = 'Kolmogorov-Smirnov', valley_check = False))})
    os.chdir("../")
    fig, axes = plt.subplots(2, 2, figsize = (12, 12), tight_layout = True)
    chromianpalette = [color_palette("Paired")[i] for i in [3, 11, 9, 7, 1]]
    axes[0, 0].set_xlabel(r"Change point $\widehat{\tau}$ detected from $x$")
    axes[0, 1].set_xlabel(r"Change point $\widehat{\tau}$ detected from $x$")
    axes[1, 0].set_xlabel(r"Change point $\widehat{\tau}$ detected from $z$")
    axes[1, 1].set_xlabel(r"Change point $\widehat{\tau}$ detected from $z$")
    PLT00   = histplot(data = CompDFX, element="step", binwidth = 10, alpha = .15, linewidth = .25,
                       kde = True, kde_kws = {"bw_adjust": .4}, line_kws = {"linewidth": 1.2},
                       fill = True, ax = axes[0, 0], legend = True, palette = chromianpalette)
    PLT10   = histplot(data = CompDFZ, element="step", binwidth = 10, alpha = .15, linewidth = .25,
                       kde = True, kde_kws = {"bw_adjust": .6}, line_kws = {"linewidth": 1.2},
                       fill = True, ax = axes[1, 0], legend = False, palette = chromianpalette)
    PLT01 = histplot(data = CompDFX, element="step", binwidth = 10, alpha = .8, linewidth = 1, kde = False,
                    cumulative = True, fill = False, ax = axes[0, 1], legend = False, palette = chromianpalette)
    PLT11 = histplot(data = CompDFZ, element="step", binwidth = 10, alpha = .8, linewidth = 1, kde = False,
                    cumulative = True, fill = False, ax = axes[1, 1], legend = False, palette = chromianpalette)
    for ax in axes.flatten():
        ax.axvline(300, color = 'red', linewidth = .75, linestyle = '--', alpha = .5)
        ax.set_xlim(0, 500)
    axes[0, 1].grid(color = 'black', alpha = .25, linewidth = .3)
    axes[0, 1].set_ylabel("Cumulative count")
    axes[1, 1].grid(color = 'black', alpha = .25, linewidth = .3)
    axes[1, 1].set_ylabel("Cumulative count")
    fig.savefig("ComprCPM_onXZ.png", pdi = 600)
    plt.show()
    # Summary
    print( pd.concat([CPDF.agg(['count', 'median', 'mean', 'std']).T for CPDF in [CompDFX, ComprDF, CompDFZ]], axis = 1) )

# === END OF THE SCRIPT === #

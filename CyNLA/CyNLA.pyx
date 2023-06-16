# cython: language_level=3, boundscheck=False
# distutils: language = c++

# Version: 0.20.9.11

cimport numpy       as cnp
import  numpy       as np
import  pyBindEDM
from itertools import product as __product
from multiprocessing import cpu_count, Pool


def __SMap(kwargs):                                 ##
    chrmEDM = chromianEDM(kwargs.pop("Array"))       #
    return chrmEDM.SMap(**kwargs)                    # SMap and Simplex projection set in the top level
                                                     #    so that we can apply multiprocessing on them.
def __Simplex(kwargs):                               # To users: try not to use them directly.
    chrmEDM = chromianEDM(kwargs.pop("Array"))       #
    return chrmEDM.Simplex(**kwargs)                ##

cdef class chromianEDM(object):
    """ An interface to pyBindEDM constructed for the purpose of minimizing calling pandas. """
    
    judgement = ['RMSE', 'MAE', 'rho']

    cdef object         bindDF
    cdef cnp.ndarray    Array

    def __cinit__( self, cnp.ndarray Array ):
        """ Initial setting of chromianEDM in Cython """
        DF          = pyBindEDM.DF()
        DF.dataList = [ ("Obs{:d}".format(i), Array[:,i].tolist()) for i in range(Array.shape[1]) ]
        DF.timeName = 'time'
        DF.time     = [ str(t) for t in range(Array.shape[0]) ]
        self.Array  = Array
        self.bindDF = DF

    @property           # callable
    def bindDF(self):
        return self.bindDF

    @property           # callable
    def Array(self):
        return self.Array

    def SMap( self,
              lib               = "",
              pred              = "",
              int  E            = 0,
              theta             = .0,
              int Tp            = 1,
              int tau           = -1,
              int knn           = 0,
              exclusionRadius   = 0,
              smapFile          = "",
              jacobians         = "",
              embedded          = False,
              verbose           = False,
              const_pred        = False,
              judgement         = False):
        ''' Sequential Locally Weighted Global Linear Map (Sugihara 1994) '''

        solver      = None
        pathIn      = pathOut       = "./"
        dataFile    = predictFile   = ""
        columns     = target        = self.bindDF.dataList[0][0]
        D = pyBindEDM.SMap( pathIn,             dataFile,           self.bindDF,    # self.bindDF to be the dataFrame (pyInput)
                            pathOut,            predictFile,        lib,
                            pred,               E,                  Tp,
                            knn,                tau,                theta,
                            exclusionRadius,    columns,            target,
                            smapFile,           jacobians,          solver,
                            embedded,           const_pred,         verbose )['predictions']

        if judgement in chromianEDM.judgement:
            LeftBoundary, RightBoundary = [ int(k) for k in lib.split() ]
            D = ( (LeftBoundary, RightBoundary, E, theta ),
                   pyBindEDM.ComputeError(D['Observations'], D['Predictions'])[judgement] )
        return D

    def Simplex(self,
                lib               = "",
                pred              = "",
                int E             = 0,
                theta             = .0,
                int Tp            = 1,
                int tau           = -1,
                int knn           = 0,
                exclusionRadius   = 0,
                smapFile          = "",
                jacobians         = "",
                embedded          = False,
                verbose           = False,
                const_pred        = False,
                judgement         = False):
        ''' Simplex Projection, with some redundant arguments set to be consistant with SMap '''

        pathIn      = pathOut       = "./"
        dataFile    = predictFile   = ""
        columns     = target        = self.bindDF.dataList[0][0]
        D = pyBindEDM.Simplex(  pathIn,         dataFile,       self.bindDF,    pathOut,        predictFile,
                                lib,            pred,           E,              Tp,             knn,
                                tau,            exclusionRadius,                columns,        target,
                                embedded,       const_pred,     verbose )

        if judgement in chromianEDM.judgement:
            LeftBoundary, RightBoundary = [ int(k) for k in lib.split() ]
            D = ( (LeftBoundary, RightBoundary, E, theta ),
                   pyBindEDM.ComputeError(D['Observations'], D['Predictions'])[judgement] )
        return D

    def Judge(self,
              method      = 'SMap',                                                     # or 'Simplex'
              judgement   = 'RMSE',                                                     # or 'MAE' or 'rho'
              E_range     = range(2, 11),
              theta_range = [.01, .10, .30, .50, .75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],  # as default in pyEDM
              lib         = "",
              pred        = "",
              int Tp      = 1,
              int tau     = -1,
              embedded    = False,
              verbose     = False,
              const_pred  = False,
              numThreads  = cpu_count()-1):     # // Somehow it seem to use a single cpu but a little bit faster.
        '''Determine the predicting performance by means of RMSE, rho, or MAE.'''

        KWARGS      = []

        if method == "Simplex":
            theta_range = [.0]
            PredictFunc = __Simplex
        elif method == "SMap":
            PredictFunc = __SMap

        for E, theta in list(__product( E_range, theta_range )):
            KWARGS.append({ "Array": self.Array,
                            "judgement": judgement,
                            "E": E,
                            "theta": theta,
                            "lib": lib,
                            "pred": pred,
                            "Tp": Tp,
                            "tau": tau,
                            "embedded": embedded,
                            "verbose": verbose,
                            "const_pred": const_pred } )
        with Pool(numThreads) as PoolWorkers:
            Performance = PoolWorkers.map( PredictFunc, tuple(KWARGS) )
        return dict(Performance)


#####################################################################
#   Nested Library Analysis based on Empirical Dynamical Modeling   #

from    scipy.ndimage   import  gaussian_filter1d   as __gaussian_filter1d
import  pandas          as      pd

cdef GaussianFilter( pdseries, sigma ):
    returnedvalue   = pdseries.copy()   # Initialize
    transformed     = __gaussian_filter1d( pdseries.dropna(), sigma, mode = 'nearest' )
    returnedvalue[~np.isnan(pdseries)]  = transformed
    return returnedvalue

cdef SignOfGradient( pdseries ):
    returnedvalue   = pdseries.copy()
    try:
        returnedvalue[~np.isnan(pdseries)]  = np.gradient( pdseries.dropna() )
    except ValueError:
        returnedvalue[:]    = np.nan
    return returnedvalue

cdef class nla(object):
    """ Nested Library Analysis """

    cdef cnp.ndarray    block

    def __cinit__(self, cnp.ndarray block):
        self.block  = block

    @property
    def block(self):
        return self.block

    def run(self,
            FlowRef,
            LargestLib,
            SmallestLib,
            int skipstep,
            method              = 'SMap',
            judgement           = 'RMSE',
            E_range             = range(2, 11),
            theta_range         = [.01, .10, .30, .50, .75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],  # as default in pyEDM
            int Tp              = 1,
            int tau             = -1,
            int numThreads      = cpu_count() -1,
            ):

        # Construct a nested set of libraries
        LargestLib  = [ int(idx) for idx in LargestLib.split(" ")  ]
        SmallestLib = [ int(idx) for idx in SmallestLib.split(" ") ]
        NestedLibs  = list( __product(np.arange(LargestLib[0], SmallestLib[0]+.5, skipstep).astype(int),
                                      np.arange(SmallestLib[1], LargestLib[1]+.5, skipstep).astype(int)) )
        TaskTotal   = len(NestedLibs)
        TaskNumber  = 0
        NLARES      = {}
        chrmDF      = chromianEDM(self.block)
        for L in range( TaskTotal ):
            res = chrmDF.Judge( method          = method,
                                judgement       = judgement,
                                E_range         = E_range,
                                theta_range     = theta_range,
                                lib             = "{:d} {:d}".format( *NestedLibs[L] ),
                                pred            = FlowRef,
                                Tp              = Tp,
                                tau             = tau,
                                numThreads      = numThreads)
            TaskNumber += 1
            print(" Progressing... \t{:3.2f} %".format(100 * TaskNumber/TaskTotal), end = "\r")
            NLARES  = {**NLARES, **res}
        return NLARES

    @staticmethod
    def SignalProcessing( nlares, FromRight = False, SIGMA = 4 ):
        nlares          = pd.Series( nlares )
        nlares.index    = nlares.index.droplevel( 1 - int(FromRight) )
        index           = list( set(nlares.reset_index()['level_0']) )
        index           = np.sort( index )

        rawSignal       = pd.DataFrame( index = index, columns = index )
        rawSignal.columns.name  = 'Trial'       # Masters: boundaries of largest library in each trial.
        rawSignal.index.name    = 'Boundary'    # Workers: boundaries of libraries in each trial.
        nlares              = nlares.unstack(level = 0)
        nlares.columns.name = 'Trial'

        for trial in rawSignal.columns:
            parameter           = nlares[ trial ].idxmin()  # Choose the best (E, theta) according to the master
            rawSignal[ trial ]  = nlares.loc[ parameter ]   # And then all the workers obey the master's choice
        TRAINGLE    = np.triu(np.ones(rawSignal.shape).astype(bool))
        if not FromRight:
            TRAINGLE = TRAINGLE.T
        rawSignal   = rawSignal.where( TRAINGLE )

        return nla.Signal_from_rawS( rawSignal, SIGMA )

    @staticmethod
    def Signal_from_rawS( rawSignal, SIGMA ):
        tilSignal   = rawSignal.apply( lambda x: GaussianFilter( x, sigma = SIGMA ) )
        trdSignal   = tilSignal.apply( lambda x: SignOfGradient( x ) )
        return rawSignal, tilSignal, trdSignal

#####################################################
#   Functions for generating pedagogical examples   #

from    scipy.integrate         import  solve_ivp

def Simulator( function,    initial,    t_eval,
               float prob_PE = .1,                  # Probability of occurrence of process error within a time unit
               float sigm_PE = .15,                 # The strength of process error
               ):

    T_min, T_max    = np.min(t_eval), np.max(t_eval)

    number_PE = np.random.poisson( lam = prob_PE*(T_max-T_min) )
    moment_PE = np.random.uniform( T_min, T_max, size = (number_PE+2, ))
    moment_PE[ 0 ]      = T_min
    moment_PE[-1:]      = T_max + 1
    moment_PE           = np.sort( moment_PE, axis=0 )

    ProcessError        = np.random.normal( 1, sigm_PE, size = (len(moment_PE),len(initial)) )
    ProcessError[0,:]   = ProcessError[-1,:]    = 1                                 # No process error at the begining and last moment

    DataSheet   = initial.reshape(1, -1).copy()
    LatestData  = DataSheet.copy()

    for i in range( 1, len(moment_PE) ):
        Sampling    = t_eval[ np.logical_and( moment_PE[i-1] <= t_eval, t_eval < moment_PE[i] ) ]
        Sampling    = np.concatenate(( Sampling, moment_PE[i:][0:1] ))
        LatestData  = solve_ivp( function,
                                 t_span = ( moment_PE[i-1], moment_PE[i] ),
                                 y0     = LatestData[-1, :] * ProcessError[i, :],   # Perturb before a run, after a sampling moment.
                                 t_eval = Sampling,
                                 ).y.T
        DataSheet   = np.concatenate(( DataSheet, LatestData[:-1, :]), axis = 0)    # Update the data sheet

    return DataSheet

cdef class dx:
    """ Derivatives of functions for modeling, which assumed to be used with Simulator(). """

    @staticmethod
    cdef cnp.float_t TypeII( cnp.float_t R,
                             cnp.float_t h,
                             cnp.float_t a = 1,
                             ) nogil:
        # Type II functional response
        return ( a*R / ( 1 + a*h*R ) )
    @staticmethod
    cdef cnp.float_t LakeCycle( cnp.float_t N,
                                cnp.float_t a = .2,
                                cnp.float_t b = .8,
                                cnp.float_t c = 1 ,
                                int         p = 8 ,
                                cnp.float_t H = 1 ,
                                ) nogil:
        # A model for nutrient cycles in lakes
        return ( a - b*N + c*( N**p / (N**p + H**p) ) )

    @staticmethod
    def AH3SP_Lake( cnp.float_t[:]  STATE,
                    cnp.float_t     a   = .2,
                    cnp.float_t     b   = .8,
                    cnp.float_t     c   =  1,
                    int             p   =  8,
                    cnp.float_t     H   =  1,
                    object          A1  = lambda N: (.75*N + 3.7),
                    cnp.float_t     A2  = .1,
                    cnp.float_t     B1  = 3.25,
                    cnp.float_t     B2  = 2.,
                    object          D1  = lambda N: (.05*N + .4),
                    cnp.float_t     D2  = .01):

        # STATE[0] = N stands for nutrients in a lake.
        cdef cnp.float_t[4] dSTATE
        dSTATE[0]   = dx.LakeCycle( N = STATE[0], a = a, b = b, c = c, p = p, H = H )

        # (STATE[1] -> STATE[2] -> STATE[3]) is a food chain model given by Alan Hastings,
        # whereas the mortality D1 and the type II functional response F1
        # are influenced by the nutrient STATE[0].
        cdef cnp.float_t    F1  = A1(STATE[0]) * dx.TypeII( R = STATE[1], h = B1 )
        cdef cnp.float_t    F2  =      A2      * dx.TypeII( R = STATE[2], h = B2 )

        dSTATE[1]   =   STATE[1]*(1-STATE[1])   - F1*STATE[2]
        dSTATE[2]   = - STATE[2]*D1(STATE[0])   + F1*STATE[2]   - F2*STATE[3]
        dSTATE[3]   = - STATE[3]*D2                             + F2*STATE[3]

        return dSTATE

# == END OF THE SCRIPT == #

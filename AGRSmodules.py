import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import gsw

# ----------------------------------------------------------------------- #
# round down or up to nearest /a decimal (default a = 10)

def varmin(var, a = 10):
    varmin = np.floor(var.min() * a) / a
    return varmin

def varmax(var, a = 10):
    varmax = np.ceil(var.max() * a) / a
    return varmax

def setup_text_plots(fontsize=8, usetex=True):
    """
    This function adjusts matplotlib settings so that all figures in the
    textbook have a uniform format and look.
    """
    mpl.rc('legend', fontsize=fontsize, handlelength=3)
    mpl.rc('axes', titlesize=fontsize)
    mpl.rc('axes', labelsize=fontsize)
    mpl.rc('xtick', labelsize=fontsize)
    mpl.rc('ytick', labelsize=fontsize)
    mpl.rc('text', usetex=usetex)
    mpl.rc('font', size=fontsize, family='serif',
                  style='normal', variant='normal',
                  stretch='normal', weight='normal')


# ----------------------------------------------------------------------- #
# Eftirhugt
#   16 nov 2016, 16 jun 2016  
def GetHydr(HydrPath, StodPath):
    """ 
    GetHydr
    aquire useful information about the trip and its stations

    GetHydr enters the trips hydr and stodfil and returns
    station name, station number, day, time, lat, lon
    
    Every trip has the files hydr and stodfil. In *hydr* every line contains information about one of
    the trips stations. The informaton we need are the station number, time and date, and location. 
    In stodfil we get information that links every station number to a station name. 
    
    Input:  HydrPath, StodPath
    Output: Hydr
    """
    
    # read hydr file
    dfH = pd.read_csv(HydrPath, header = None, 
                        usecols = (1,10,11,12,13), 
                        names = ('StNum', 'Date', 'Time','Lon','Lat'),
                        dtype = {'StNum': np.int64, 'Lon': np.float64, 'Lat': np.float64},
                        index_col = ['StNum'])
    # read stod file
    dfS = pd.read_csv(StodPath, header = None, 
                        usecols = (0, 3), skiprows = 15, 
                        names = ['StNum', 'StName'],
                        index_col = ['StNum'])
    # make Hydr variable
    Hydr = pd.concat([dfS.StName,dfH], axis = 1)
    #Hydr.sort(columns='StName')
    
    return Hydr


# ----------------------------------------------------------------------- #
# Eftirhugt
#   16 nov 2016, 16 jun 2016  
def GetData(PathHav, station):
    """ 
    GetData 
    aquire ocean data for each station

    GetData enters each station file, gets relevant data, 
    calculates density from the GSW package and returns 
    depth, pressure, temperature, salinity, fluorescence, oxygen and density

    The data is aguired from two files, since the oxygen data is in a separate file. 
    The relevant data is picked (num and qual are ignored). 
    
    Input:  PathHav, station
    Output: Data
    """
    
    # prepare variables and create paths
    StNum, lon, lat = station.name, station.Lon, station.Lat
    StPath1 = PathHav + '{}.raw'.format(StNum)
    StPath2 = PathHav + 'Ox{}.raw'.format(StNum)

    
   #GET DATA
    # get complicated data from txt with genfromtxt and make DataFrame
    dfDat = pd.DataFrame(np.genfromtxt(StPath1, 
                        skip_header = 9,
                        delimiter = (5,8,8,8,8,6,5), 
                        missing_values = ' ',
                        names = ('depth','pres','temp','sal','flu','num','qual')))
    # get ox data    
    dfOx = pd.DataFrame(np.genfromtxt(StPath2, 
                        skip_header = 9,
                        delimiter = (5,8,8,6,4), 
                        missing_values = ' ',
                        names = ('depth','pres','ox','num','qual')))  
    # pick out relevant data
    D = pd.concat([dfDat.loc[:,'depth':'flu'], dfOx['ox']], axis=1);
    D = D[pd.notnull(D.depth)] #dropping faulty rows where depth = NaN
     
       
   #CALCULATE DENSITY from gibbs sea water 
    SA = gsw.SA_from_SP(D.sal, D.pres, lon, lat)    # gsw.SA_from_SP(SP,p,long,lat)
    CT = gsw.CT_from_t(SA, D.temp, D.pres)          # gsw.CT_from_t(SA,t,p)
    p_ref = 0
    dens = gsw.rho_CT_exact(SA, CT, p_ref) - 1000   # sigma0 = rho_CT_exact(SA, CT, 0) - 1000
                                                
        # SP : salinity (PSS-78) [unitless], 
        # p : pressure [dbar]
        # lon : decimal degrees east [0..+360] or [-180..+180] 
        # lat : decimal degrees (+ve N, -ve S) [-90..+90]
        # SA : Absolute salinity [g/kg ]
        # t : in situ temperature degC 
    
    GSW = pd.DataFrame({'CT': CT, 'SA': SA, 'dens':dens}) 
       
   # Create Data frame
    Data = pd.concat([D, GSW], axis=1)
    Data = Data.set_index('depth', drop = False)
    
    return Data


# ----------------------------------------------------------------------- #
# Eftirhugt
#   16 nov 2016, 16 jun 2016  
def TurData(TripNo):

    """ 
    TurData
    acuires information and data for A WHOLE TRIP

    TurData calls GetHydr and GetData to get all trip information and data. 
    
    Input:  TripNo
    Output: Hydr, Data
    """

   # CREATE PATHS NEEDED
    # cruise folder
    Path = '../DATA/CTD/cru{}/'.format(TripNo)
    # specific files
    #SecPath = Path + 'sect{}.csv'.format(TripNo)
    HydrPath = Path + 'hydr{}.dat'.format(TripNo)
    StodPath = Path + 'stodfil{}.csv'.format(TripNo)
    PathHav = Path + 'hav{}/'.format(TripNo) # data folder

   # GETTING HYDR AND DATA
    # get trip info
    Hydr = GetHydr(HydrPath, StodPath);
    
    # iterate through Hydr lines (stations) to aquire data
    for a, StNum in enumerate(Hydr.index):
        HydrLine = Hydr.loc[StNum]
        D = GetData(PathHav, HydrLine)
                
        # create great Data Frame
        if a == 0:  Data = pd.concat([D], keys=[StNum])
        else:       Data = pd.concat([Data,pd.concat([D], keys=[StNum])])
            
    Data.index.names = ['StNum', 'depth']
    
    return Hydr, Data


# ----------------------------------------------------------------------- #
# Eftirhugt
#   17 nov 2016
def StodData(TimeDef):
    
    """ 
    StodData
    acuires information and data for PREDEFINED STATIONS

    TurData calls GetHydr and GetData to get all trip information and data. 
    
    Input:  TripNo
    Output: Hydr, Data
    """

    # MAKING HYDR
    # collecting whole Hydr info for all trips involved
    for a, TripNo in enumerate(TimeDef.TripNo.unique()):
        Path = '../DATA/CTD/cru{}/'.format(TripNo)
        HydrPath = Path + 'hydr{}.dat'.format(TripNo)
        StodPath = Path + 'stodfil{}.csv'.format(TripNo)

        Hydr0 = GetHydr(HydrPath, StodPath)

        if a == 0: Hydr = Hydr0
        else: Hydr = pd.concat([Hydr, Hydr0])

    # picking Hydr for the relevand station numbers        
    StNum = TimeDef.StNo.astype(np.int64).tolist()
    Hydr = Hydr.loc[StNum]

    # MAKING DATA
    for a, TripNo in enumerate(TimeDef.TripNo):
        #preparing for GetData
        Path = '../DATA/CTD/cru{}/'.format(TripNo)
        PathHav = Path + 'hav{}/'.format(TripNo)
        station = Hydr.iloc[a]

        # getting data
        D = GetData(PathHav, station)
        # setting Data in a DataFrame
        if a == 0: Data = pd.concat([D], keys = [station.name])
        else:      Data = pd.concat([Data,pd.concat([D], keys = [station.name])])
    
    # reindex so that Data index is in same order as defined
    Data = Data.reindex(StNum, level = 0) 
    Data.index.names = ['StNum', 'depth']
    
    return Hydr, Data

# ----------------------------------------------------------------------- #
# Eftirhugt
#   17. nov 2016 - plot FO og zoom á relevant øki    
def PlotMap(Hydr, MapLabels = True, Savefig = False, Figpath = False, Form = 'jpg'):
    """ 
    PlotMap 
    PlotMap plots station locations on a map. 
    
    Input:  TripNo, Hydr, MapLabels (station labels, default = True)
    
    Output: None
    """

   # INITIATING FIGURE
    fig, ax = plt.subplots(figsize = (10,12))
    
   # BASEMAP PROJECTION
     # find data lon, lat bounds and round to pretty numbers
    lonmin,latmin = Hydr[['Lon','Lat']].min()
    lonmax,latmax = Hydr[['Lon','Lat']].max()
    lonmin,latmin = varmin(lonmin-0.005, a = 100), varmin(latmin-0.005, a = 100)
    lonmax,latmax = varmax(lonmax+0.005, a = 100), varmax(latmax+0.005, a = 100)

    m = Basemap(projection = 'merc', resolution = None,
                llcrnrlat = latmin, urcrnrlat = latmax,
                llcrnrlon = lonmin, urcrnrlon = lonmax)

    # Draw islands from txt file and fill.
    for island in os.listdir('Coasts'):
        lon, aa, lat  = np.genfromtxt('Coasts/'+island, delimiter = ' ').T
        xpt, ypt = m(lon, lat)
        m.plot(xpt,ypt,'black', linewidth = 1)
        plt.fill(xpt,ypt,'tan')
    
   # PLOTTING STATIONS INTO MAP   
    for lon, lat, name in zip(Hydr.Lon, Hydr.Lat, Hydr.StName):
        xpt, ypt = m(lon, lat)
        m.plot(xpt, ypt, 'bo', markersize = 9)
        # setting labels if requested
        if MapLabels == True:
            xp, yp = m(Hydr.Lon.mean(), Hydr.Lat.mean())
            xoffset, yoffset = 0.05*xp, 0.05*xp
            plt.text(xpt - xoffset, ypt + yoffset, name, fontsize = 9)
    
    #plt.title('{}'.format(MapName), fontsize=14)        
    plt.title('CTD stations - {}'.format(Hydr.Date.iloc[0]), fontsize = 20)   
    
    # Draw meridional and zonal lines
     # define gap between thick and thin lines depending on map wiev   
    if (latmax - latmin) > 0.05: dlat, dlon, ddlat, ddlon = 0.1, 0.05, False, False
    elif (latmax - latmin) <= 0.05: dlat, dlon, ddlat, ddlon = 0.05, 0.02, 0.01, 0.01 
     # draw thick lines  
    parallels = np.arange(varmin(latmin, a = 20), varmax(latmax, a = 20), dlon)    
    meridians = np.arange(varmin(lonmin, a = 20), varmax(lonmax, a = 20), dlat)
    m.drawparallels(parallels, labels = [1,0,0,0], linewidth = 0.5, color = 'k')
    m.drawmeridians(meridians, labels = [0,0,0,1], linewidth = 0.5, color = 'k')
     # draw thin lines
    if ddlat:
        parallels = np.arange(latmin, latmax, ddlon)
        meridians = np.arange(lonmin, lonmax, ddlat)
        m.drawparallels(parallels, labels = [1,0,0,0], linewidth = 0.2, color = 'k')
        m.drawmeridians(meridians, labels = [0,0,0,1], linewidth = 0.2, color = 'k')
    
    if Savefig:
        path = Figpath + '/map.{}'.format(Form)
        plt.savefig(path, format=Form, dpi=400, bbox_inches='tight')
# ----------------------------------------------------------------------- #    
# Eftirhugt
#   17. nov 2016 
def PlotProfiles(TripNo, Hydr, Data, Profiles = False, Savefig = False, Figpath = False, Form = 'jpg', Scale= False):

    """    
    PlotProfiles
    creates a figure for each station, whith one plot for each parameter
    
    Input:  Hydr, Data, Profiles
    Output: None    
    """

    
    # possibility to choose spesific stations or all
    if Profiles:    Hydro = Hydr.iloc[Profiles,:]
    else:           Hydro = Hydr 
    
    # Setting Scale for Vars
    Vars = ['temp','sal','flu','ox','dens']
     # default
    datamin = Data[Vars].min() 
    datamax = Data[Vars].max() 
     # chosen global scale
    if Scale: datamin[Vars], datamax[Vars] = Scale
    
    # Units to display on plot
    SI = [r'$^\circ C$',
          r'$g/kg$',
          r'${\rho}_{\theta} - 1000 \/ kg/m^3$',
          r'$mg/m^3$',r'$mg/L$']
    
    # iterating through stations and plotting profiles  
    for StNum in Hydro.index:
        
        # preparing data
        Name = Hydro.loc[StNum].StName
        Dat = Data.loc[StNum] [Vars]
        Dep = Data.loc[StNum] ['depth']
        
        # initiate figure
        f, axarr = plt.subplots(1, len(Vars), sharey=True, figsize = (8,8))
        plt.suptitle('Station name: {}. Station number: {}'.format(Name, StNum), fontsize=18)
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=None, wspace=None, hspace=None)
        
        # iterate through variables and plot subplots
        for V, Var in enumerate(['temp', 'sal', 'dens', 'flu', 'ox']):
            # plot
            axarr[V].plot(Dat[Var],-Dep,'b')

            #set axes, titles etc.
            axarr[V].set_title(Var, fontsize = 16)      # Title - parameter name
            axarr[V].set_xlabel(SI[V], fontsize = 16)   # Xlabel - SI
            if V == 0: axarr[V].set_ylabel(r'Depth $m$', fontsize = 16) # Ylabel on first subplot
            axarr[V].locator_params(axis='x',nbins=4)   # Xticks
            axarr[V].set_xlim(xmin = datamin[Var].min(), xmax = datamax[Var].max()) #Xlim
            if Dep.max() < 60: plt.ylim(ymin = -60)     # Ylim
            
        #left  = 0.125  # the left side of the subplots of the figure
        #right = 0.9    # the right side of the subplots of the figure
        #bottom = 0.1   # the bottom of the subplots of the figure
        #top = 0.9      # the top of the subplots of the figure
        #wspace = 0.2   # the amount of width reserved for blank space between subplots
        #hspace = 0.5   # the amount of height reserved for white space between subplots
        
        if Savefig:
            path = Figpath + '/{0}.{1}'.format(Name,Form)
            plt.savefig(path, format=Form, dpi=400, bbox_inches='tight')

# ----------------------------------------------------------------------- #
def MakeSection(TripNo, Hydr, Data, Columns = False, Vars = False, Savefig = False, Figpath = False, Form = 'jpg'):
    # Set path where to find section definition    
    SecPath = '../DATA/CTD/cru{0}/sect{0}.csv'.format(TripNo)
        
   # GET SECTIONS INFORMATION
    # read section definition, all or predefined cols
    if Columns: SecDef = pd.read_csv(SecPath, header=None, dtype = 'str', usecols = Columns)
    else:       SecDef = pd.read_csv(SecPath, header=None, dtype = 'str')
    
   # SETTING PLOTTING THINGS
    # which variables to plot
    if Vars: Vars = Vars
    else: Vars = ['temp','sal','flu','ox','dens']
    # names to plot
    VarName = ['Temperature', 'Salinity', 'Fluorescence', 'Oxygen']
    # limits and colors   
    VarMin = Data.min()[Vars]
    VarMax = Data.max()[Vars]
    ColorStep = pd.Series([0.05, 0.02, 0.2, 0.1, 0.1], Vars)
    ColorMin  = np.floor(VarMin * 10.0) / 10.0 - ColorStep
    if ColorMin['flu'] < 0: ColorMin.flu = 0.0
    
    ColorMax  = np.ceil(VarMax * 10.0) / 10.0 + ColorStep
        

    
    #ColorStep = [0.01, 0.01,  0.1, 0.05, 0.05 ]
    #ColorMin  = [ 9.8, 34.8,  0.0, 6.7, 26.7]
    #ColorMax  = [10.6, 35.2, 10.0, 9.6, 27.1]
    #ColorMin  = [3.5, 28.5, 0.0, 5.0, 22.0]
    #ColorMax  = [11.3, 35.3, 20.0, 13.2, 28.0]
    
   # ITERATE THROUGH SECTIONS (COLUMNS IN DEF FILE)  
    for colindex in SecDef:
        column = SecDef[colindex]
       # BREAKING DOWN COLUMN 
        # picking name, direction and data out of column
        SecName = column[0] 
        SecDir = str.lower(column[1])
        SecSt = column[2:].dropna(axis=0) # actual station numbers in section
        SecSt = pd.to_numeric(SecSt).tolist()  # list of staton numbers as integers
        StNames = Hydr.StName[Hydr.index.isin(SecSt)].reindex(SecSt).tolist()
        
       # PREPARING DATA FOR PLOTTING 
        # picking the relevant section stations from Data   
        SecDat = Data.loc[SecSt] 
        # Making one df per variable. Rotate so that stations = columns
        Temp = SecDat.temp.unstack(0)
        Sal = SecDat.sal.unstack(0)
        Flu = SecDat.flu.unstack(0)
        Ox = SecDat.ox.unstack(0)
        Rho = SecDat.dens.unstack(0)
        # The order of these stations is not necessarily the order we need. 
        # We reorder the columns by SecSt
        Temp, Sal, Flu, Ox, Rho = Temp[SecSt], Sal[SecSt], Flu[SecSt], Ox[SecSt], Rho[SecSt]
        
       # CREATING DEPTH VS LONLAT MESH (x = lonlat, y = depth)
        # picking lonlat in section, in right order, to list.
        if SecDir == 'lon':  x = Hydr.Lon[Hydr.index.isin(SecSt)].reindex(SecSt).tolist()
        else:                x = Hydr.Lat[Hydr.index.isin(SecSt)].reindex(SecSt).tolist()
        y = range(len(Temp)) 
        X,Y = np.meshgrid(x, y)   
        
        # lengths and name
        if max(y) < 60: ylen = 60
        else: ylen = max(y)
                
        if SecDir == 'lon': 
            xlen = (max(x)-min(x))*800
            xlab = 'Tvørskurður'
        else: 
            #xlen = (max(x)-min(x))*600
            xlen = (max(x)-min(x))*150
            xlab = 'Longdarskurður'

       # ITERATE THROUGH VARIABLES AND PLOT 
        for a, var in enumerate([Temp, Sal, Flu, Ox]):
            fig = plt.figure(figsize = (xlen,10))

            V1 = np.arange(ColorMin[a],ColorMax[a],ColorStep[a])
            V2 = np.arange(ColorMin[4],ColorMax[4],ColorStep[4])

            C1 = plt.contourf(X, -Y, var, V1, cmap=plt.cm.rainbow)
            C2 = plt.contour(X,-Y, Rho, V2, colors='k')
            plt.clabel(C2, colors='k', fontsize=12, fmt='%2.1f')

            plt.vlines(x,-ylen,0)
            cbar = plt.colorbar(C1)
            cbar.ax.set_ylabel(VarName[a],fontsize = 16)
            cbar.ax.tick_params(labelsize=12)

            plt.title(SecName, fontsize = 20)
            plt.xlabel(xlab, fontsize = 16)
            plt.ylabel('Dýpi', fontsize = 16)
            
            #plt.xticks(x,[Deg for Deg in x], rotation = 'vertical')
            plt.xticks(x,[Name for Name in StNames], rotation = 'vertical')
            
            plt.margins(0.02)
                
            if Savefig:
                path = Figpath + '/Sect{}_{}.{}'.format(SecName, Vars[a], Form)
                plt.savefig(path, dpi=400, format = Form, bbox_inches='tight')


# ----------------------------------------------------------------------- #
def PlotTimeSeries(Hydr, Data, StName, Savefig = False, Figpath = False, Form = 'jpg', 
                   DensLine = True, DensDelta = False, FigDim = (16,6)):
    
   # PREPARING PLOT 
    # creating X,Y data mesh for plotting 
    y = range(Data.depth.max().astype(int)) # range from 0 to max depth
    x = pd.to_datetime(Hydr.Date, format = '%d-%m-%Y').astype('int') # datetime integers for stations
    X,Y = np.meshgrid(x,y)

    # names, limits and colors
    VarName = ['Temperature', 'Salinity', 'Fluorescence', 'Oxygen']   
    Vars = ['temp','sal','flu','ox','dens']
    VarMin = Data.min()[Vars]
    VarMax = Data.max()[Vars]
    ColorStep = pd.Series([0.05, 0.05, 0.2, 0.1, 0.2],[Vars])
    ColorMin = np.floor(VarMin * 10.0) / 10.0 - 2 * ColorStep
    ColorMax = np.ceil(VarMax * 10.0) / 10.0 + 2 * ColorStep
    
    # not iterating over Rho or Rho colorstep   
    Rho = Data.dens.unstack(0)
    if DensDelta: 
        V2 = np.arange(ColorMin[4],ColorMax[4],DensDelta)
    else:
        V2 = np.arange(ColorMin[4],ColorMax[4],ColorStep[4])
        
   # ITERATE THROUGH VARIABLES AND PLOT 
    for a, var in enumerate(Vars[:4]):
    
        # creating df's with each variable. 
        # 3d data: value at depth and station
        var = var = Data[var].unstack(0)     
        var.index = -var.index

        fig = plt.figure(figsize = FigDim)

        V1 = np.arange(ColorMin[a],ColorMax[a],ColorStep[a])
        C1 = plt.contourf(X, -Y, var, V1, cmap=plt.cm.rainbow)
        
        if DensLine:
            C2 = plt.contour(X,-Y, Rho, V2, colors='k')
            plt.clabel(C2, colors='k', fontsize=10, inline=1, fmt='%2.1f')

        plt.vlines(X[0],-Y.max(),0)
        cbar = plt.colorbar(C1)

        plt.title('{0}, {1}'.format(StName, VarName[a]), fontsize = 16)
        plt.ylabel('Dýpi', fontsize = 14)
        plt.xticks(X[0],np.array(Hydr.Date), rotation = 'vertical')
        plt.margins(0.02)    
        
        if Savefig:
            if Figpath: Figpath = Figpath
            else: Figpath = '../DATA/CTD/Timeseries/'
            path = Figpath + '{}_{}.{}'.format(StName, Vars[a], Form)
            plt.savefig(path, format = Form, dpi=400, bbox_inches='tight')


def TSdiagram(TripNo, Savefig, Figpath, Form):
    # Get trip data 
    Hydr, Data = TurData(TripNo, Save = False)
    
    # Create variables with user-friendly names
    temp  = Data.temp
    salt  = Data.sal
    ox    = Data.ox
    
  #PREPARE DATA MESH FOR DENSITY LINES
    # Figure out boudaries (mins and maxs) 
    # round down or up to nearest 0.1 decimal
    smin, smax = varmin(salt), varmax(salt)
    tmin, tmax = varmin(temp), varmax(temp)
    
    # Calculate how many gridcells we need in the x and y dimensions
    sdim = round( (smax-smin)*10 )
    tdim = round( (tmax-tmin)*100 )
    
    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(tmin, tmax, num = tdim)
    si = np.linspace(smin, smax, num = sdim)
    
    # Create empty grid of zeros
    dens = np.zeros((tdim,sdim))
    # Loop to fill in grid with densities - 1000
    for j in range(0,int(tdim)):
        for i in range(0, int(sdim)):
            dens[j,i] = gsw.rho(si[i], ti[j], 0) - 1000
    
  #MAKING FIGURE
    fig1 = plt.figure()
    plt.title('Trip number: {}'.format(TripNo), fontsize = 14)
    plt.xlabel('Salinity', fontsize = 14)
    plt.ylabel('Temperature (C)', fontsize = 14)
    
  #PLOTTING DENSITY LINES 
    C1 = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(C1, fontsize=12, inline=1, fmt='%2.2f')

  # PLOTTING DATA
    for StNum in Hydr.index:
        # Create variables with user-friendly names
        Temp  = temp.loc[StNum]
        Salt  = salt.loc[StNum]
        Ox    = ox.loc[StNum]
        
        C2 = plt.scatter(Salt, Temp, c = Ox, s = 50, marker='o', cmap = 'rainbow')

    cbar = plt.colorbar(C2)
    cbar.ax.set_ylabel('oxygen')
    
    if Savefig:
        path = Figpath + 'TSdiagram.{}'.format(Form)
        plt.savefig(path, format = Form, dpi=400, bbox_inches='tight')
    
    
def TSdiagrams(TripNo):
    # Get trip data 
    Hydr, Data = TurData(TripNo, Save = False)
    
    # Create variables with user-friendly names
    temp  = Data.temp
    salt  = Data.sal
    ox    = Data.ox
    
    #PREPARE DATA MESH FOR DENSITY LINES
    # Figure out boudaries (mins and maxs) 
    # round down or up to nearest 0.1 decimal
    smin, smax = varmin(salt), varmax(salt)
    tmin, tmax = varmin(temp), varmax(temp)
    
    # Calculate how many gridcells we need in the x and y dimensions
    sdim = round( (smax-smin)*10 )
    tdim = round( (tmax-tmin)*100 )
    
    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(tmin, tmax, num = tdim)
    si = np.linspace(smin, smax, num = sdim)
    
    # Create empty grid of zeros
    dens = np.zeros((tdim,sdim))
    # Loop to fill in grid with densities - 1000
    for j in range(0,int(tdim)):
        for i in range(0, int(sdim)):
            dens[j,i] = gsw.rho(si[i], ti[j], 0) - 1000
             
    for StNum in Hydr.index: 
        # Create variables with user-friendly names
        Temp  = temp.loc[StNum]
        Salt  = salt.loc[StNum]
        Ox    = ox.loc[StNum]
        
        fig = plt.figure()
        fig1 = plt.figure()
        plt.title('Trip: {0}. Station name: {1}'.format(TripNo, Hydr.StName[StNum]), fontsize = 16)
        plt.xlabel('Salinity', fontsize = 14)
        plt.ylabel('Temperature (C)', fontsize = 14)
        
        
        C1 = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
        plt.clabel(C1, fontsize=12, inline=1, fmt='%2.2f')
        
        C2 = plt.scatter(Salt, Temp, c = Ox, s = 50, marker='o', cmap = 'rainbow')
        cbar = plt.colorbar(C2)
        cbar.ax.set_ylabel('oxygen')
        
        if Savefig:
            path = Figpath + 'TSdiagram{}.{}'.format(Hydr.StName[StNum], Form)
            plt.savefig(path, format = Form, dpi=400, bbox_inches='tight')   

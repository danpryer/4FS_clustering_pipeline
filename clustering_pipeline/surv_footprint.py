# some functions relating specifically to the 4FS analysis

# from imports import *
import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib.path import Path
import pymangle

# functions below used to look up the 4FS survey footprint and check if a
# point (in RA, DEC) is in our out of the survey mask.



# 4most s8 footprint - input ra, dec in degrees units and it outputs True or false if
# its in the field (first return is for s8 surveys minus ELG, second return is ELG)
def fourmost_get_s8foot(ra_deg, dec_deg, mask_path):
    ra = ra_deg.value
    dec = dec_deg.value
    svbdict = fourmost_get_survbitlist()
    survbit = np.zeros(len(ra),dtype=int)
    for bname in ['des','desi','kidss','kidsn','atlassgcnotdes','atlasngc']:
        isb           = fourmost_get_survbit_indiv(ra_deg,dec_deg,bname, mask_path)
        survbit[isb] += 2**svbdict[bname]
    # bg/lrgi/qso/lya
    iss8  = ((((survbit & 2**svbdict['des'])>0) | ((survbit & 2**svbdict['atlassgcnotdes'])>0)) & ((survbit & 2**svbdict['desi'])==0)) | ((((survbit & 2**svbdict['kidsn'])>0) & (ra>154)) | ((survbit & 2**svbdict['atlasngc'])>0))
    # elg
    iss8elg = (((ra>330) | (ra<90)) & (dec>-35.5) & (dec<-26)) | ((ra>50) & (ra<90) & (dec>-40) & (dec<-26))
    return iss8,iss8elg




# get survey footprint bit
def fourmost_get_survbit_indiv(ra_deg,dec_deg,bname, mask_path):
    #
    # Galactic l,b
    c       = SkyCoord(ra=ra_deg, dec=dec_deg, frame='fk5')
    l,b     = c.galactic.l.value,c.galactic.b.value
    lon,lat = c.barycentrictrueecliptic.lon.degree,c.barycentrictrueecliptic.lat.degree
    ra = ra_deg.value
    dec = dec_deg.value
    # 4most/s8 , s8elg
    if (bname in ['s8elg','s8']):
        iss8,iss8elg = fourmost_get_s8foot(ra_deg, dec_deg)
        if (bname=='s8elg'):
            keep = iss8elg
        else: # s8
            keep = iss8
    # desi: no ply file for the moment...
    elif (bname=='desi'):
        # desi sgc
        polyra = np.array([0 ,-25,-35,-50,-54,-45,10,10,  60, 70, 70,53,42,42,38,0])
        polydec= np.array([33,33, 25,  8,  -8, -15,-15,-20,-20,-15,0, 0, 10,20,33,33])
        sgcpoly = Path(np.concatenate(
                            (polyra. reshape((len(polyra),1)),
                                polydec.reshape((len(polyra),1))),
                            axis=1))
        # desi ngc
        polyra = np.array([275,107,115,130,230,230,230,255,265,275])
        polydec= np.array([33, 33, 12, -10,-10, -2, -2, -2,  13, 33])
        ngcpoly = Path(np.concatenate(
                            (polyra. reshape((len(polyra),1)),
                             polydec.reshape((len(polyra),1))),
                            axis=1))
        #
        tmpradec         = np.transpose(np.array([ra,dec]))
        tmp              = (ra>300)
        tmpradec[tmp,0] -= 360.
        keep = np.zeros(len(ra),dtype=bool)
        for poly in [sgcpoly,ngcpoly]:
                keep[poly.contains_points(tmpradec)] = True
    elif (bname=='erosita'):
        # johan email 30/07/2018 14:12
        keep = (abs(b)>15) & (l>180)
    elif (bname=='waveswide'):
        # https://wavesurvey.org/project/survey-design/
        keep = (((ra>155) & (ra<240) & (dec>-5) & (dec<5))
                |
                (((ra>330) | (ra<50)) & (dec>-36) & (dec<-26)))
    elif (bname=='euclid'):
        keep = (np.abs(b)>=30) & (np.abs(lat)>5.)
    else:
        if (bname[:3]=='vhs'):
            ## -70<dec<0
            mng    = pymangle.Mangle(mask_path+'vhsdec.ply')
            polyid = mng.polyid(ra,dec)
            keepdec= (polyid!=-1)
            # |b|>bmin
            mng    = pymangle.Mangle(mask_path+bname[3:6]+'.ply')
            polyid = mng.polyid(l,b)
            keepb  = (polyid!=-1)
            ##
            keep   = (keepdec) & (keepb)
        else:
            mng    = pymangle.Mangle(mask_path+bname+'.ply')
            polyid = mng.polyid(ra,dec)
            keep   = (polyid!=-1)
        if (bname=='vhsb20clean'):
            # Jext<0.1 and low nstar selection [both VHS and DES]
            ra60    = (ra>55)  & (ra<65)  & (dec>-5)  & (dec<0)
            ra70    = (ra>67)  & (ra<72)  & (dec>-16) & (dec<-13)
            ra100   = (ra>50)  & (ra<180) & (dec>-20) & (dec<0)   & (b>-23) & (b<0)
            ra120   = (ra>100) & (ra<180)                         & (b>-15) & (b<0)
            ra230   = (ra>228) & (ra<270) & (dec>-40) & (dec<-20) & (b>0)
            ra250   = (ra>235) & (ra<270) & (dec>-20) & (dec<0)   & (b>0)
            ra300   = (ra>180) & (ra<360) & (dec>-70) & (dec<0)   & (b>-25) & (b<0)
            LMC     = (ra>70)  & (ra<90)  & (dec>-70) & (dec<-65)
            keep    = ((keep) &
                        (~ra60)  & (~ra70)  &
                        (~ra100) & (~ra120) &
                        (~ra230) & (~ra250) & (~ra300) &
                        (~LMC))
    #print( bname, len(ra[keep]))
    return keep





# get survey footprint bitlist
def fourmost_get_survbitlist():
    mydict                = {}
    mydict['des']         = 0
    mydict['kidss']       = 1
    mydict['kidsn']       = 2
    mydict['atlassgcnotdes'] = 3
    mydict['atlasngc']    = 4
    mydict['kabs']        = 5
    mydict['vhsb10']      = 6
    mydict['vhsb15']      = 7
    mydict['vhsb20']      = 8
    mydict['vhsb20clean'] = 9
    mydict['desi']        = 10
    mydict['erosita']     = 11
    mydict['waveswide']   = 12
    mydict['euclid']      = 13
    mydict['s8elg']       = 14
    mydict['s8']          = 15
    return mydict

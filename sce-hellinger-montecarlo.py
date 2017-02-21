import csv
import sqlite3
import numpy
import numpy.random as random

def hellinger(p, q):
    if numpy.sum(numpy.sqrt(p * q)) > 1:
        return 1.0
    else:
        return numpy.sqrt(2 * (1 - numpy.sum(numpy.sqrt(p * q))))

## to query sqlite dataset
##
##conn = sqlite3.connect('pilotdataset.sqlite')
##c = conn.cursor()
##conn.text_factory = str
##
##kats1sql = c.execute("SELECT kat1.kat1 FROM kat1 WHERE kat1.kat1 IS NOT NULL GROUP BY kat1.kat1").fetchall()
##kats2sql = c.execute("SELECT kat2.kat2 FROM kat2 WHERE kat2.kat2 IS NOT NULL GROUP BY kat2.kat2").fetchall()
##
##nkat1 = len(kats1sql)
##nkat2 = len(kats2sql)
##
##kats1 = [i[0] for i in kats1sql]
##kats2 = [i[0] for i in kats2sql]
##
##data1 = c.execute("SELECT quant.key, bibl.project, quant.nfr, kat1.kat1, kat2.kat2, date.date1, date.date2  FROM quant LEFT JOIN bibl ON quant.key = bibl.key LEFT JOIN kat1 ON quant.key = kat1.key LEFT JOIN kat2 ON quant.key = kat2.key LEFT JOIN date ON quant.key = date.key WHERE (date.date2 >= -200 AND date.date1 <= 20  AND kat1.kat1 IS NOT NULL)")
##filename = 'data1.csv'
##with open(filename, 'wt') as output:
##    writer = csv.writer(output)
##    writer.writerows(data1)
d1 = numpy.genfromtxt('data1.csv', delimiter = ',')
##data2 = c.execute("SELECT quant.key, bibl.project, quant.nfr, kat1.kat1, kat2.kat2, date.date1, date.date2  FROM quant LEFT JOIN bibl ON quant.key = bibl.key LEFT JOIN kat1 ON quant.key = kat1.key LEFT JOIN kat2 ON quant.key = kat2.key LEFT JOIN date ON quant.key = date.key WHERE (date.date2 >= -200 AND date.date1 <= 20  AND kat2.kat2 IS NOT NULL)")
##filename = 'data2.csv'
##with open(filename, 'wt') as output:
##    writer = csv.writer(output)
##    writer.writerows(data2)
d2 = numpy.genfromtxt('data2.csv', delimiter = ',')
##conn.close()
##
#################################################
##
## raw data written as data1.csv and data2.csv for parameterizations kat1 and kat2



nan = numpy.isnan(d1[0,4:5])
d1[nan] = 9999
nan = numpy.isnan(d2[0,4:5])
d2[nan] = 9999
nan = numpy.isnan(d1)
d1[nan] = 0
nan = numpy.isnan(d2)
d2[nan] = 0

v = 0.1

startdate = int(-200 *v)
enddate = int(20 * v)

sitecol = 1
quantcol = 2
kat1col = 3
kat2col = 4
date_a = 5
date_b = 6

t = range(startdate, enddate + 1)
lent = len(t)

projectlist = [101,201,210,220,221,301,309,315,350,401,505,610,612,680]
regionlist = [200,300,600,999]
regionlist_r = [200,300,600]
lenregionlist_r = len(regionlist_r)

pchoice = 999

bcount = int(input("Number of Monte Carlo permutations: "))

#mc approximation of phi
for b_i in range(0,bcount):
    for wchoice in ['WR','UR']:
        if wchoice == 'WR':
            H_kat1 = numpy.empty([lent,lenregionlist_r,0])
            H_kat2 = numpy.empty([lent,lenregionlist_r,0])
            for nkat in [nkat1,nkat2]:
                if nkat == nkat1:
                    kats = kats1
                    katcol = kat1col
                    katlabel = 'kat1'
                    d = random.permutation(d1)
                elif nkat == nkat2:
                    katcol = kat2col
                    katlabel = 'kat2'
                    d = random.permutation(d2)
                    kats = kats2
                records = d.shape[0]
                for project in projectlist:
                    alphaprior = 1/float(nkat) 
                    priorA = 1
                    alphatable = numpy.empty([lent,nkat])
                    Atable = numpy.empty([lent,nkat])
                    alphatable.fill(alphaprior)
                    Atable.fill(priorA)
                    numpy.save('post-{0}-{1}-alpha.npy'.format(project,katlabel),alphatable)
                    numpy.save('post-{0}-{1}-A.npy'.format(project,katlabel),Atable)
                for row in range(0,records): 
                    fromproject = int(d[row,sitecol])
                    a = round(int(d[row,date_a])*v)
                    b = round(int(d[row,date_b])*v)
                    alphatable = numpy.load('post-{0}-{1}-alpha.npy'.format(fromproject,katlabel))
                    Atable = numpy.load('post-{0}-{1}-A.npy'.format(fromproject,katlabel))
                    for t_j in t:
                        jrow = abs(startdate-t_j)
                        if a <= t_j <= b:
                            quant = d[row,quantcol] * (1 / float(b-a+1))
                            alphatable[jrow,kats.index(d[row,katcol])] = alphatable[jrow,kats.index(d[row,katcol])] + quant
                            Atable[jrow,:] = Atable[jrow,:] + quant
                    numpy.save('post-{0}-{1}-alpha.npy'.format(fromproject,katlabel),alphatable)
                    numpy.save('post-{0}-{1}-A.npy'.format(fromproject,katlabel),Atable)
                    for region in regionlist:
                        if region == 200:
                            regionset = [201,210,220,221]
                        if region == 300:
                            regionset = [301,309,315,350]
                        if region == 600:
                            regionset = [505,610,612,680]
                        if region == 999:
                            regionset = projectlist    
                        alphaavg = numpy.zeros([lent,nkat])
                        Aavg = numpy.zeros([lent,nkat])
                        Aavg.fill(len(projectlist))
                        for project in regionset:
                            filename_alpha = 'post-{0}-{1}-alpha.npy'.format(project,katlabel)
                            filename_A = 'post-{0}-{1}-A.npy'.format(project,katlabel)
                            posterioralpha = numpy.load(filename_alpha)
                            posteriorA = numpy.load(filename_A)
                            avg = posterioralpha/posteriorA
                            alphaavg = alphaavg + avg
                        alphaavg = 1/float(len(regionset)) * alphaavg
                        filename_avg = 'post-WR-{0}-{1}.npy'.format(region,katlabel)
                        numpy.save(filename_avg, alphaavg)
                    Hresult = numpy.empty([lent,lenregionlist_r])
                    pdist = numpy.load('post-{0}-{1}-{2}.npy'.format(wchoice,pchoice,katlabel))
                    for region in regionlist_r:
                        qdist = numpy.load('post-{0}-{1}-{2}.npy'.format(wchoice,region,katlabel))
                        for t_j in t:
                            jrow = abs(startdate-t_j)
                            p = pdist[jrow,:]
                            q = qdist[jrow,:]
                            Hresult[jrow,regionlist_r.index(region)] = hellinger(p, q)
                    if nkat == nkat1:
                        H_kat1 = numpy.dstack((H_kat1,Hresult))
                    if nkat == nkat2:
                        H_kat2 = numpy.dstack((H_kat2,Hresult))
                    print('Phi calculated for row {0} of {1}. {2}'.format(row,d.shape[0],katlabel))
            numpy.save('H-kat1-WR-{0}.npy'.format(b_i),H_kat1)
            numpy.save('H-kat2-WR-{0}.npy'.format(b_i),H_kat2)
        elif wchoice == 'UR':
            H_kat1 = numpy.empty([lent,lenregionlist_r,0])
            H_kat2 = numpy.empty([lent,lenregionlist_r,0])
            for nkat in [nkat1,nkat2]:
                if nkat == nkat1:
                    kats = kats1
                    katcol = kat1col
                    katlabel = 'kat1'
                    d = random.permutation(d1)
                elif nkat == nkat2:
                    katcol = kat2col
                    katlabel = 'kat2'
                    d = random.permutation(d2)
                    kats = kats2
                records = d.shape[0]
                for region in regionlist:
                    if region == 200:
                        regionset = [201,210,220,221]
                    if region == 300:
                        regionset = [301,309,315,350]
                    if region == 600:
                        regionset = [505,610,612,680]
                    elif region == 999:
                        regionset = projectlist
                    alphaprior = 1/float(nkat)
                    priorA = 1
                    alphatable = numpy.empty([lent,nkat])
                    Atable = numpy.empty([lent,nkat])
                    alphatable.fill(alphaprior)
                    Atable.fill(priorA)
                    numpy.save('post-UR-{0}-{1}-alpha.npy'.format(region,katlabel),alphatable)
                    numpy.save('post-UR-{0}-{1}-A.npy'.format(region,katlabel),Atable)
                for row in range(0,records):
                    for region in regionlist:
                        if region == 200:
                            regionset = [201,210,220,221]
                        if region == 300:
                            regionset = [301,309,315,350]
                        if region == 600:
                            regionset = [505,610,612,680]
                        elif region == 999:
                            regionset = projectlist
                        if d[row,sitecol] in regionset:
                            a = round(int(d[row,date_a])*v)
                            b = round(int(d[row,date_b])*v)
                            alphatable = numpy.load('post-UR-{0}-{1}-alpha.npy'.format(region,katlabel))#load that project's
                            Atable = numpy.load('post-UR-{0}-{1}-A.npy'.format(region,katlabel))#load that project's
                            for t_j in t:
                                jrow = abs(startdate-t_j)
                                if a <= t_j <= b:
                                    quant = d[row,quantcol] * (1 / float(b-a+1))
                                    alphatable[jrow,kats.index(d[row,katcol])] = alphatable[jrow,kats.index(d[row,katcol])] + quant
                                    Atable[jrow,:] = Atable[jrow,:] + quant
                            numpy.save('post-UR-{0}-{1}-alpha.npy'.format(region,katlabel),alphatable)
                            numpy.save('post-UR-{0}-{1}-A.npy'.format(region,katlabel),Atable)   
                    Hresult = numpy.empty([lent,lenregionlist_r])
                    filenamep_alpha = 'post-{0}-{1}-{2}-alpha.npy'.format(wchoice,pchoice,katlabel)
                    filenamep_A = 'post-{0}-{1}-{2}-A.npy'.format(wchoice,pchoice,katlabel)
                    avgalpha = numpy.load(filenamep_alpha)
                    avgA = numpy.load(filenamep_A)
                    pdist = avgalpha/avgA
                    for region in regionlist_r:
                        posterioralpha = numpy.load('post-{0}-{1}-{2}-alpha.npy'.format(wchoice,region,katlabel) )
                        posteriorA = numpy.load('post-{0}-{1}-{2}-A.npy'.format(wchoice,region,katlabel))
                        qdist = posterioralpha/posteriorA
                        for t_j in t:
                            jrow = abs(startdate-t_j)
                            p = pdist[jrow,:]
                            q = qdist[jrow,:]
                            Hresult[jrow,regionlist_r.index(region)] = hellinger(p, q)
                    if nkat == nkat1:
                        H_kat1 = numpy.dstack((H_kat1,Hresult))
                    if nkat == nkat2:
                        H_kat2 = numpy.dstack((H_kat2,Hresult))
                    print('Phi calculated for row {0} of {1}. {2}'.format(row,d.shape[0],katlabel))
            numpy.save('H-kat1-UR-{0}.npy'.format(b_i),H_kat1)
            numpy.save('H-kat2-UR-{0}.npy'.format(b_i),H_kat2)
    print('MC set n {0} complete'.format(b_i+ 1))


#mode and hpd
from scipy.stats import gaussian_kde
from rpy2.robjects.packages import importr
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects import globalenv
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
r = ro.r

importr("coda", "C:/Users/USER/Documents/R/win-library/3.1")

v = 0.1

uHkat1 = numpy.empty([lent,lenregionlist_r,0])
uHkat2 = numpy.empty([lent,lenregionlist_r,0])
wHkat1 = numpy.empty([lent,lenregionlist_r,0])
wHkat2 = numpy.empty([lent,lenregionlist_r,0]) 

for b_i in range(0,40):
    uHkat1load = numpy.load('phi/H-kat1-UR-{0}.npy'.format(b_i))
    uHkat2load = numpy.load('phi/H-kat2-UR-{0}.npy'.format(b_i))
    wHkat1load = numpy.load('phi/H-kat1-WR-{0}.npy'.format(b_i))
    wHkat2load = numpy.load('phi/H-kat2-WR-{0}.npy'.format(b_i))
    uHkat1 = numpy.dstack((uHkat1,uHkat1load))
    uHkat2 = numpy.dstack((uHkat2,uHkat2load))
    wHkat1 = numpy.dstack((wHkat1,wHkat1load))
    wHkat2 = numpy.dstack((wHkat2,wHkat2load))
    print('{0} complete'.format(b_i))

matrixlist = [uHkat1,uHkat2,wHkat1,wHkat2]
matrixlab = ['uHkat1', 'uHkat2', 'wHkat1', 'wHkat2']

i = 0
for matrix in matrixlist:
    modematrix = numpy.empty((matrix.shape[0],matrix.shape[1]))
    upper = numpy.empty((matrix.shape[0],matrix.shape[1]))
    lower = numpy.empty((matrix.shape[0],matrix.shape[1]))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            y = matrix[row,col,:]
            yr = ro.FloatVector(y)
            globalenv['yr'] = yr
            r("ymc <- yr")
            hpd = r("HPDinterval(as.mcmc(ymc), prob=0.9)")
            print(hpd[0],hpd[1])
            upper[row,col] = hpd[1]
            lower[row,col] = hpd[0]
    numpy.savetxt('{0}-upper.csv'.format(matrixlab[i]),upper, fmt="%10.50e", delimiter=",")
    numpy.savetxt('{0}-lower.csv'.format(matrixlab[i]),lower, fmt="%10.50e", delimiter=",")
    i = i + 1

i = 0
for matrix in matrixlist:
    modematrix = numpy.empty((matrix.shape[0],matrix.shape[1]))
    upper = numpy.empty((matrix.shape[0],matrix.shape[1]))
    lower = numpy.empty((matrix.shape[0],matrix.shape[1]))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            grid = numpy.linspace(0, 1.4, 100)
            density = gaussian_kde(matrix[row,col,:], bw_method = 'silverman')
            y = density.evaluate(grid)
            hp = numpy.where(y == max(y))
            mode = grid[hp]
            modematrix[row,col] = mode
            print('mode calculated')
    numpy.savetxt('{0}-mode.csv'.format(matrixlab[i]),modematrix, fmt="%10.50e", delimiter=",")


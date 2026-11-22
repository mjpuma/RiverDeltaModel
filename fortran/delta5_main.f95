Program delta5_main
implicit none
!***********************************************************************
!     Routine:  delta_main.f95                                         *
!     Function:  Model for river delta stratigraphy that simulates     *   
!                sediment dynamics, river avulsion, and sea level      *
!                change                                                *
!     Written:  Aug-Sept 2006                                          *
!     By:       M.J. Puma                                              *
!     Revised:  2026, corrections 1-9 marked 'Correction N (2026)'.    *
!               See README.md for the list.                            *
!***********************************************************************
    
integer*4, parameter :: prec10 = selected_real_kind(10)
integer*4, parameter :: numsed =  1
integer*4, parameter :: xlim = 71 ! change output format
integer*4, parameter :: ylim = 51 ! if this is changed
integer*4, parameter :: zlim =  20000
integer*4, parameter :: numcell = 10000
integer*4,parameter :: ntdim = 610000
integer*4,parameter :: ntavul= 30000
integer*4,parameter :: nvis = 100   ! visit durations stored per cell

integer*4, parameter :: neta = 11
integer*4, parameter :: nlev = 12
integer*4, parameter :: nQs = 14
integer*4, parameter :: nB = 15
integer*4, parameter :: nH = 16
integer*4, parameter :: nbond = 17
integer*4, parameter :: etain = 18
integer*4, parameter :: nHavg = 22
integer*4, parameter :: navul = 23
integer*4, parameter :: ndepthch = 24
integer*4, parameter :: ndepthov = 25
integer*4, parameter :: nchlay = 27
integer*4, parameter :: nvisit = 28
integer*4, parameter :: nvisitn = 29




integer*4, parameter :: nstrat2 = 30
integer*4, parameter :: nsedtyp2 = 31
integer*4, parameter :: nstrat5 = 32
integer*4, parameter :: nsedtyp5 = 33
integer*4, parameter :: nstrat10 = 34
integer*4, parameter :: nsedtyp10 = 35
integer*4, parameter :: nstrat20 = 36
integer*4, parameter :: nsedtyp20 = 37
integer*4, parameter :: nstrat25 = 38
integer*4, parameter :: nsedtyp25 = 39
integer*4, parameter :: nstrat30 = 40
integer*4, parameter :: nsedtyp30 = 41
integer*4, parameter :: nstrat40 = 42
integer*4, parameter :: nsedtyp40 = 43
integer*4, parameter :: nstrat50 = 44
integer*4, parameter :: nsedtyp50 = 45

integer*4, parameter :: nqsout2 = 46
integer*4, parameter :: nqsout5 = 47
integer*4, parameter :: nqsout10 = 48
integer*4, parameter :: nqsout20 = 49
integer*4, parameter :: nqsout25 = 50
integer*4, parameter :: nqsout30 = 51
integer*4, parameter :: nqsout40 = 52
integer*4, parameter :: nqsout50 = 53

integer*4, parameter :: nsedtim2 = 54
integer*4, parameter :: nsedtim5 = 55
integer*4, parameter :: nsedtim10 = 56
integer*4, parameter :: nsedtim20 = 57
integer*4, parameter :: nsedtim25 = 58
integer*4, parameter :: nsedtim30 = 59
integer*4, parameter :: nsedtim40 = 60
integer*4, parameter :: nsedtim50 = 61

integer*4, parameter :: nTseries = 62
integer*4, parameter :: nTseries2 = 63
integer*4, parameter :: nvisit2 = 64
integer*4, parameter :: nvisitn2 = 65

! STATIC MEMORY
integer*4 numstorm,itemp0,i_on,i_Qwflag,i_etaflag,&
          i,j,k,jj,kk,zz,zzz,avul_flag,flag_etacheck,&
          imidx,xmax,ymax,i_act,j_act,num_avul,istorm,&
          flag_scour,i_old,j_old,itemp,jtemp,num_act,ind_avul,&
          i_now,j_now,layer_now,i_out,flag_layer,&
          i_next,j_next,nlay_max
logical avul_test
integer*4,dimension(xlim,ylim) :: latbond,top_lay,num_ch_lay,&
                    flag_visit,flag_visit2,visit_num,visit_num2

real(kind = prec10) Lb,a,g,beta,sec_yr,&
          Qs_bed,Qw_vol,I_tavg,pi,H_down,&
          temp,S_init,S_incise,S_temp,B_width,dL,&
          qtback,qtfrnt,qt_dif,qt_now,a_u,S_equil,&
          D,R,lambda,alpha_r,taustar_c,p,&
          n,ks,tau_star,alpha_so,alpha_sa,tau_norm,nk,&
          H_0,qs_zero,qw_zero,Cf,&
          qstar,dtredu,delta_agg_max,dtmin,dtmax,&
          eustacy,t,tprint,tprint2,tprint3,dt,tmax,dtprint,dtprint2,&
          dtprint3,new_eta,sea_rise,eta_before,&
          base_init,elev_temp,depth_temp,teq,L_act,&
          subside,avul_cond,num_cells, fine_dep, basevar,delta_incise
real(kind = prec10),dimension(xlim,ylim) :: Hij,Qsij,eta, &
                    old_eta,delta_eta,delta_agg,delta_strat,&
                    depth_chan,depth_over,Havg,Navg,levee,old_levee,&
                    visit,last_t,last_t2,visit2
real(kind = prec10),dimension(xlim,ylim,nvis) :: Tv_series,Tv_series2
                    
real(kind = prec10),dimension(ylim) :: levee_maxj,H_maxj

real(kind = prec10),dimension(5):: S,L
real(kind = prec10),dimension(5,2):: coord


! DYNAMIC MEMORY
! Stratigraphy matrices
integer*4,dimension(xlim,ylim,zlim) :: sed_typ
real(kind = prec10),dimension(xlim,ylim,zlim) :: strat,sed_tim


! Avulsion matrix
real(kind = prec10),dimension(ntavul,3) :: avul_out                
                                      
real(kind = prec10),dimension(ntdim) :: qw_vec,next_tim_qw,qs_vec,B_vec
real(kind = prec10),dimension(ntdim) :: qout2,qout5,qout10,qout20,qout25,qout30,qout40,qout50

integer*4,dimension(xlim*ylim) :: S_index,i_vec,j_vec
real(kind = prec10),dimension(xlim*ylim) :: S_act,deltaL_act,chan,&
                     H,qt,eta_act,old_eta_act,levee_act, old_levee_act

!   Open read file and files to save specific pieces of output
open(neta,file='etaout.txt',RECL=9216)
open(nlev,file='levout.txt',RECL=9216)
open(nQs,file='Qsout.txt',RECL=9216)
open(nB,file='Bout.txt',RECL=9216)
open(nH,file='Hout.txt',RECL=9216)
open(nbond,file='latout.txt',RECL=9216)
open(nHavg,file='Havg.txt',RECL=9216)
open(navul,file='avul_out.txt',RECL=9216)
open(ndepthch,file='channel.txt',RECL=9216)
open(ndepthov,file='over.txt',RECL=9216)
open(nchlay,file='numchlay.txt',RECL=9216)
open(nstrat2,file='strat2.txt',RECL=9216)
open(nsedtyp2,file='sedtyp2.txt',RECL=9216)
open(nstrat5,file='strat5.txt',RECL=9216)
open(nsedtyp5,file='sedtyp5.txt',RECL=9216)
open(nstrat10,file='strat10.txt',RECL=9216)
open(nsedtyp10,file='sedtyp10.txt',RECL=9216)
open(nstrat20,file='strat20.txt',RECL=9216)
open(nsedtyp20,file='sedtyp20.txt',RECL=9216)
open(nstrat25,file='strat25.txt',RECL=9216)
open(nsedtyp25,file='sedtyp25.txt',RECL=9216)
open(nstrat30,file='strat30.txt',RECL=9216)
open(nsedtyp30,file='sedtyp30.txt',RECL=9216)
open(nstrat40,file='strat40.txt',RECL=9216)
open(nsedtyp40,file='sedtyp40.txt',RECL=9216)
open(nstrat50,file='strat50.txt',RECL=9216)
open(nsedtyp50,file='sedtyp50.txt',RECL=9216)
open(nvisit,file='visit.txt',RECL=9216)
open(nvisitn,file='visitnum.txt',RECL=9216)
open(nvisit2,file='visit2.txt',RECL=9216)
open(nvisitn2,file='visitnum2.txt',RECL=9216)

open(nsedtim2,file='sedtim2.txt',RECL=9216)
open(nsedtim5,file='sedtim5.txt',RECL=9216)
open(nsedtim10,file='sedtim10.txt',RECL=9216)
open(nsedtim20,file='sedtim20.txt',RECL=9216)
open(nsedtim25,file='sedtim25.txt',RECL=9216)
open(nsedtim30,file='sedtim30.txt',RECL=9216)
open(nsedtim40,file='sedtim40.txt',RECL=9216)
open(nsedtim50,file='sedtim50.txt',RECL=9216)

open(nqsout2,file='qst2.txt',RECL=9216)
open(nqsout5,file='qst5.txt',RECL=9216)
open(nqsout10,file='qst10.txt',RECL=9216)
open(nqsout20,file='qst20.txt',RECL=9216)
open(nqsout25,file='qst25.txt',RECL=9216)
open(nqsout30,file='qst30.txt',RECL=9216)
open(nqsout40,file='qst40.txt',RECL=9216)
open(nqsout50,file='qst50.txt',RECL=9216)

open(nTseries,file='Tv_series.txt',RECL=9216)
open(nTseries2,file='Tv_series2.txt',RECL=9216)

call random_seed()   ! standard Fortran (the 2007 code used the FTN95 routine date_time_seed@)

!**********************************************************************
! Assign parameters
xmax = 71        ! change output format
ymax = 50       ! if dimensions change
a = 100           ! cell length [m]
Lb = a*ymax        ! length of domain [m]

num_cells = xmax*ymax
imidx = 36 

Qw_vol = 20.d0     ! Volumetric water feed rate[m**3/s]
Qs_bed = 0.0025d0  ! Volumetric sediment feed rate [m**3/s] 
i_Qwflag = 0   ! 0 if constant Qw/Qs, 1 if variable Qw/Qs  
i_etaflag = 0  ! 0 random initial eta, 1 if read from file  

tmax = 710.d0 ! maximum simulation time [eff yr]
teq = 110.d0 ! time to wait before averaging [eff yr]
dtmax = 0.0001d0   ! max time increment [eff yr]
tprint = dtmax		! keeps track of time to print output [eff yr]
dtprint = 1.d0   ! time interval to print output [eff yr]  
tprint2  = 0.001d0   ! keeps track of time to avg output [eff yr] 
dtprint2 = 0.001d0   ! time interval to print output [eff yr] 
tprint3  = 0.1d0   ! keeps track of time to print qs [eff yr] 
dtprint3 = 0.1d0   ! time interval to print qs output [eff yr] 
I_tavg = 0.05d0  ! parameter for time averaging
sec_yr = (365.d0*24.d0* 60.d0*60.d0) ! time converted to seconds

beta = 1.d0      ! parameter in avulsion eqn [-]
g = 9.81d0 ! gravitational acceleration [m/s**2]

D=0.0003d0    ! grain diameters [m] 
R=1.65d0      ! submerged specific grav [-]
lambda=0.4d0  ! porosity of deposited sediment [-]

taustar_c = 0.d0  ! critical Shields stress below for sediment transport
n = 2.5d0          ! exponent in the sediment-transport relation
alpha_r = 15.d0  ! dimensionless resistance coef. in Manning-Strickler 
nk = 2.d0    ! a dimensionless parameter used for roughness of a flat bed
             ! between 1.5 and 3 Kamphuis(1974) 
ks = nk*D  ! a roughness height characterizing the bumpiness of the bed[m]
alpha_so = 11.25d0 ! coefficients in the Meyer-Peter and Mueller 
alpha_sa = 1.5d0  !   sediment-transport eqn (Parker et al, 1998)
p = 1.d0/6.d0    ! dimensionless exponent in Manning-Strickler relation

! Calculate channel with based on Church, M and Rood, K. (1983) data
B_width = D * 0.274d0 * (Qw_vol/((R*g*D)**(0.5d0)*(D**2.d0) ))**0.565d0 ![m]
qw_zero = Qw_vol/B_width ! Volumetric inflow of water per unit width [m**2/s]
qs_zero = Qs_bed/B_width! Volumetric sediment feed rate per unit width [m**2/s]

! Slope assuming S as a function of Qw and Qs.  It gives the 
!  average slope expected for constant Qw and Qs.   
S_equil = ((R*((Qs_bed/(alpha_sa*alpha_so)/D/B_width/SQRT(R*g*D))**(1/n) &
           +taustar_c))**((3+2*p)/(2+2*p)))* &
           (alpha_r**(1/(1+p)))*((Qw_vol/D/B_width/SQRT(g*D))**(-1/(1+p)))
S_incise = 1.0d-6 ! Slope of incised channel

S_init = S_equil ! Initial slope


pi = 4.d0*atan(1.d0)
base_init = 50.d0 ! Initial base of delta [m]
H_down = 5.d0 ! initial channel flow depth @ downstream end [m]
eustacy = 2.d0*pi/(4800.d0) ! period of base level cycling
basevar = 0.d0! Magnitude of base level cycling [m]
sea_rise = 0.d0 ! rate of sea-level rise (i.e. depth of downstream flow) [m/yr]
subside = 0.001d0  ! subsidence rate [m/yr] Parker et al 1998
fine_dep = 0.5d0*subside/I_tavg ! floodplain deposition [m/yr]

a_u = 1.d0         ! upstream (upwind) weighting parameter
dtmin = 1.0d-8
dtredu = 0.5


!**********************************************************************
! Initialize variables
do zz = 1,xmax
   do zzz = 1,(ymax+1)
      latbond(zz,zzz) = 0
      num_ch_lay(zz,zzz) = 0 
      Hij(zz,zzz) = 0.d0
      Havg(zz,zzz) = 0.d0
      Navg(zz,zzz) = 0.d0
      qsij(zz,zzz) = 0.d0
      delta_agg(zz,zzz) = 0.d0
      delta_eta(zz,zzz) = 0.d0
      delta_strat(zz,zzz) = 0.d0  
      depth_chan(zz,zzz) = 0.d0
      depth_over(zz,zzz) = 0.d0

      last_t(zz,zzz) = 0.d0
      last_t2(zz,zzz) = 0.d0     
      visit(zz,zzz) = 0.d0
      visit2(zz,zzz) = 0.d0
      flag_visit(zz,zzz) = 0
      flag_visit2(zz,zzz) = 0      
      visit_num(zz,zzz) = 0
      visit_num2(zz,zzz) = 0      
      do kk = 1,nvis
         Tv_series(zz,zzz,kk) = 0.d0
         Tv_series2(zz,zzz,kk) = 0.d0
      enddo
   enddo
enddo  

do kk = 1,zlim
   do zz = 1,xmax
      do zzz = 1,(ymax+1)
         strat(zz,zzz,kk) = 0.d0
         sed_typ(zz,zzz,kk) = -1 
         sed_tim(zz,zzz,kk) = -1.d0
      enddo
   enddo
enddo 

istorm = 1
num_avul = 0
i_out = 1 ! counter for q out vector

t = 0.d0

do i = 1,num_cells
   S_act(i) = 0.d0
   deltaL_act(i) = 0.d0
   chan(i) = 0.d0
   H(i) = 0.d0
   qt(i) = 0.d0
   S_index(i) = 0
   eta_act(i) = 0.d0
   old_eta_act(i) = 0.d0
   levee_act(i) = 0.d0
   old_levee_act(i) = 0.d0   
   i_vec(i) = 0
   j_vec(i) = 0
enddo

do i = 1,5
   S(i) = 0.d0
enddo

L(1) = real(a,kind = prec10)
L(2) = sqrt(2.d0)*real(a,kind = prec10)
L(3) = real(a,kind = prec10)
L(4) = sqrt(2.d0)*real(a,kind = prec10)
L(5) = real(a,kind = prec10)

coord(1,1) = -1
coord(1,2) = 0
coord(2,1) = -1
coord(2,2) = +1
coord(3,1) = 0
coord(3,2) = +1
coord(4,1) = +1
coord(4,2) = +1
coord(5,1) = +1
coord(5,2) = 0


!**********************************************************************
! Produce allogenic forcing (Water influx into the delta) 

if (i_Qwflag == 0) then
   ! Constant Qw
   numstorm = 1
   next_tim_qw(1) = 0.d0
   next_tim_qw(2) = tmax
   qw_vec(1) =  qw_zero
   qs_vec(1) = qs_zero
   B_vec(1) = B_width
else
   ! Variable Qw
   numstorm = int(tmax/0.001d0) ! storm ever 3.65 days
   temp = tmax* 365.d0*24.d0

   ! Compute the water flux from uniform distribution         
   do i=1,numstorm+2 
     next_tim_qw(i) = 0.001d0 * real(i-1,kind = prec10)
     call random_number(temp)
    ! Variable Qs     
     qw_vec(i)=Qw_vol/B_width
     qs_vec(i) = (0.0001d0 + temp*(0.0049d0-0.0001d0)) / B_width

!$$$$$$     ! Variable Qw
!$$$$$$      qw_vec(i) = 5.0 + temp*(35.0-5.0)
!$$$$$$      B_width = D*0.274* (qw_vec(i) / ((R*g*D)**(0.5)*(D**2.d0) ))**0.585
!$$$$$$      qw_vec(i)=qw_vec(i)/B_width
!$$$$$$      qs_vec(i)=Qs_bed/B_width
     
     B_vec(i) = B_width
   enddo
endif


!**********************************************************************
! Compute initial bed elevation

if (i_etaflag == 1) then
   call bed_elev(xmax,ymax,eta)
endif

do i = 1,xmax
   do j = 1,(ymax+1)

      if (i_etaflag == 0) then 
         call random_number(temp)
         eta(i,j)= base_init + S_init*Lb - S_init*real(a,kind=prec10)* &
               (real(j,kind = prec10) - 1.d0) + &
               (S_init*real(a,kind = prec10)*(temp-0.5d0))
      endif

      levee(i,j) = eta(i,j) 
      old_levee(i,j) = levee(i,j)   
      old_eta(i,j) = eta(i,j)

      strat(i,j,1) = eta(i,j)
      sed_typ(i,j,1) = 0 ! initial sediment
      sed_tim(i,j,1) = 0.d0 ! 
      
      sed_typ(i,j,2) = 1 ! floodplain sediment
      sed_tim(i,j,2) = 0.d0 ! 
      top_lay(i,j) = 2
   enddo
enddo

! Identify path of steepest decent
latbond(imidx,1) = 1
sed_typ(imidx,1,2) = 2

eta_act(1) = eta(imidx,1)
levee_act(1) = levee(imidx,1)
i = imidx
j = 1
i_vec(1) = i
j_vec(1) = j
i_old = 0
j_old = 0
num_act = 1

do    
  ! Exit if the right boundary has been reached
   if (j == (ymax+1)) EXIT
   call slope5(i,j,latbond,xmax,eta,L,L_act,S)  
  ! Identify next active cell 
   itemp0 = maxloc(S,1) 
   if (S(itemp0) == -999.d0) then
      ! Correction 7 (2026): no free neighbour: continue straight downstream
      itemp0 = 3
   endif
   itemp = coord(itemp0,1)
   jtemp = coord(itemp0,2)
   i_old = i
   j_old = j
   i = i+itemp
   j = j+jtemp  
  ! Update matrices for active channels
   latbond(i,j)=1
  ! Update vectors for active channels
   S_index(num_act) = itemp0   
   if (S(itemp0)< 10d-8) then
      S(itemp0) = S_incise
      new_eta =  eta(i_old,j_old) - S_incise * L(itemp0)                 
      if (new_eta > 0.d0) then
         delta_incise = new_eta - eta(i,j) 
         eta(i,j) = new_eta            
      else
         delta_incise = -eta(i,j)
         eta(i,j) = 0.d0
      endif     
      levee(i,j) = levee(i,j) + delta_incise 
   endif
   S_act(num_act) = S(itemp0)
   deltaL_act(num_act) = L(itemp0) 
   eta_act(num_act+1) = eta(i,j)
   levee_act(num_act+1) = levee(i,j)
   i_vec(num_act+1) = i
   j_vec(num_act+1) = j  
  ! Update counter of number of active channels
   num_act = num_act + 1
enddo

! Compute backwater equation to obtain channel depth 
H_0 = H_down
call  incise_H(num_act,H_0,S_act,deltaL_act,&
               ks,qw_zero,alpha_r,g,chan)

! Incise initial channel to one channel depth               
do k = 1,num_act
   i = i_vec(k)
   j = j_vec(k)
   depth_temp = levee_act(k)-eta_act(k)
   if (depth_temp < chan(k)) then
     eta_act(k) = levee_act(k) - chan(k)
     depth_temp = chan(k)
     eta(i,j) = eta_act(k)
     ! Correction 5 (2026): reset the reference elevation, so the cut is not
     !  later recorded as negative deposition
     old_eta(i,j) = eta(i,j)
   endif
  ! Initialize stratigraphy with incised channel
   sed_typ(i,j,2) = 2      ! channel sediment
   strat(i,j,1) = eta(i,j)
enddo
old_eta_act = eta_act
old_levee_act = levee_act


!**********************************************************************
! Begin time loop
do 
   dt = dtmax
   t = t + dt
   if (t>tmax) EXIT

   if (t > next_tim_qw(istorm))then
      qw_zero = qw_vec(istorm)
      qs_zero = qs_vec(istorm)
      B_width = B_vec(istorm)
      istorm = istorm + 1
   endif

   qsij(imidx,1) = qs_zero   
           
! Compute time btw chnnel visits and number of visits
   if (t>teq) then
      call tim_visit(t,xmax,ymax,latbond,flag_visit,last_t,visit,visit_num,Tv_series,&
                        flag_visit2,last_t2,visit2,visit_num2,Tv_series2) 
   endif
  
! Calculate sea level
   if(t>teq) then
      H_0 = H_down + sea_rise*((t-teq)/I_tavg)
!$$$$$$    H_0 = H_down + basevar*sin(t*eustacy)
   endif

  !iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
  ! Iteration loop    
   flag_etacheck = 0
   do
      if (flag_etacheck==1) EXIT
      flag_scour = 0
      
     ! Calculate elevation of water surface using the backwater eqn
      call backwater(num_act,i_vec,j_vec,H_0,S_act,deltaL_act,&
                     ks,qw_zero,alpha_r,g,H,Hij)
     ! Compute Sediment Transport based on H and Qw
      do k = 1,num_act
         i = i_vec(k)
         j = j_vec(k)
          
         Cf = (1.d0/alpha_r**2.d0) * (H(k)/ks) ** (-1.d0/3.d0)
         tau_star = Cf * ((qw_zero  / H(k))**2.d0)/(R*g*D) 
         
         if (k < num_act) then  
            S_temp = S_act(k)   
            tau_norm = ((ks**(1.d0/3.d0) * qw_zero**2.d0 / &
                    alpha_r**2.d0 / g) ** (3.d0/10.d0)) * &
                    (S_temp ** (7.d0/10.d0)) / R / D

            if (tau_star > tau_norm) then
               tau_star = tau_norm
            endif
         endif   

         if (tau_star > taustar_c) then 
            qstar = alpha_so*alpha_sa*(tau_star-taustar_c)** n 
         else
            qstar = 0.d0
         endif

         qt(k) = sqrt(R*g*D)*D*qstar
         qsij(i,j)= qt(k)
      enddo 
      

    ! Solve the Exner Equation to compute the new bed elevation
    !   The elevation of the bed surface is for time t+delta t    
      delta_agg_max = 0.d0
      do k = 1,num_act
         i = i_vec(k)
         j = j_vec(k)        
         qt_now = qt(k)

         if (k == 1) then
            qtback = qs_zero
            dL = L(2)
         else
            qtback = qt(k - 1)
            dL = deltaL_act(k-1)
         endif
                        
         if (k == num_act) then 
            qt_dif = qtback - qt_now
         else
            qtfrnt = qt(k + 1)
            qt_dif = a_u*(qtback-qt_now) + (1.d0-a_u)*(qt_now-qtfrnt)
         endif

        ! Compute the channel deposition and elevation change    
         delta_agg(i,j) = (dt*sec_yr)/(1.d0-lambda)/dL*qt_dif
         delta_eta(i,j) = delta_agg(i,j) - subside*dt/I_tavg
         elev_temp = eta(i,j) + delta_eta(i,j)

        ! Check to see if eroded below mimimum elevation 
         if (elev_temp<0.d0) then
            delta_agg(i,j) = -eta(i,j) + subside*dt/I_tavg
            if (delta_agg(i,j)<0.d0) then 
               delta_agg(i,j) = 0.d0
            endif
            delta_eta(i,j) = -eta(i,j)
            qt_dif = delta_agg(i,j)*(1.d0-lambda)*dL/(dt*sec_yr)
            eta(i,j) = 0.d0
            
           ! Recompute q(t) based on erosion restriction
            if (k == num_act) then 
               qt(k)  = qtback - qt_dif
            else
               ! Correction 1 (2026): invert qt_dif = a_u*(qtback-qt) + (1-a_u)*(qt-qtfrnt)
               qt(k) = (qt_dif - a_u*qtback + (1.d0-a_u)*qtfrnt) / (1.d0 - 2.d0*a_u)
            endif
         else 
            eta(i,j) = elev_temp
         endif

         levee(i,j) = levee(i,j) + delta_eta(i,j)

        ! Find the largest increase in elevation during timestep
         if (delta_agg(i,j) > delta_agg_max) then
            delta_agg_max = delta_agg(i,j)  
         endif 

        ! Update elevation vector
         eta_act(k) = eta(i,j)
         levee_act(k) = levee(i,j)

        ! Check to see if scour occurs.  If it does, reduce timestep
         if (k>1) then
            if (eta_act(k)>eta_act(k-1)) flag_scour = 1
         endif     
      enddo


      if (flag_scour == 0) then
         flag_etacheck = 1
      else
        ! Reset variables
         t = t - dt
         dt = dt*dtredu
         
         if(dt < dtmin) then
            write(*,10) t,dt,dtmin
            10  format(// & 
               2X,'DELTA T BELOW MIN VALUE ',/ &
               't = ',e14.3,5x,'dt = ',e14.3,5x,'dtredu = ',e14.3/)
            STOP
         else
           t = t + dt
           ! Clear/reset variables  
            do zzz=1,(ymax+1)   
               do zz=1,xmax
                  Hij(zz,zzz) = 0.d0
                  qsij(zz,zzz) = 0.d0
                  delta_eta(zz,zzz) = 0.d0
                  delta_agg(zz,zzz) = 0.d0
               enddo
            enddo
            do zz = 1,num_act
               qt(zz) = 0.d0
               H(zz) = 0.d0
            enddo
            eta_act = old_eta_act
            eta = old_eta
            levee_act = old_levee_act
            levee = old_levee
            
         endif
      
      endif
   end do ! End Check on delta_eta
  !iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii

!$$$$$$    ! find maximum active levee height and corresponding flow depth for each column j 
!$$$$$$    do j = 1,(ymax+1)
!$$$$$$       levee_maxj(j) = 0.d0
!$$$$$$       H_maxj(j) = 0.d0
!$$$$$$       do i = 1,xmax
!$$$$$$          if ((levee(i,j) > levee_maxj(j)).and.(latbond(i,j)==1)) then
!$$$$$$             levee_maxj(j) = levee(i,j)
!$$$$$$             H_maxj(j) = Hij(i,j)
!$$$$$$          endif
!$$$$$$       enddo    
!$$$$$$    enddo
  
   do i = 1,xmax
      do j = 1,(ymax+1)
       ! Compute the floodplain sedimentation and update bed elevation
         call sedplain(i,j,latbond,levee_maxj,H_maxj,delta_agg,&
                    delta_eta,subside,fine_dep,dt,I_tavg,eta,levee)
                    
       ! Compute total (since last avulsion) aggradation for entire domain           
         delta_strat(i,j) = delta_strat(i,j)+delta_agg(i,j) 
      enddo
      
   enddo

!**********************************************************************
   ! Module for avulsion and creation of a new channel
   !  Update: the locations of the active channels     
   avul_flag = 0
   k=1
   do 
     ! Check if an avulsion has occurred or reached right boundary 
      if ((k==num_act).or.(avul_flag == 1)) EXIT
      i = i_vec(k)
      j = j_vec(k)

     ! Identify the indices of the active channel
      i_act = i_vec(k+1)
      j_act = j_vec(k+1)

     ! Index of non-active neighbor with steepest slope
      call slope5(i,j,latbond,xmax,eta,L,L_act,S)     
      itemp0 = maxloc(S,1) 
      itemp = coord(itemp0,1)
      jtemp = coord(itemp0,2)
      ! Correction 3 (2026): length from (i,j) to the next channel cell
      if ((i_act /= i) .and. (j_act /= j)) then
         L_act = L(2)
      else
         L_act = L(1)
      endif
      
      avul_cond = beta*Hij(i,j)

     ! Check if avulsion criterion is met
     ! Correction 7 (2026): a cell with no free neighbour cannot avulse
     avul_test = .false.
     if (S(itemp0) /= -999.d0) then
        avul_test = ((eta(i,j)-avul_cond-eta(i+itemp,j+jtemp)) / L(itemp0) ) > &
                    ((eta(i,j) - eta(i_act,j_act)) / L_act)
     endif
     if (avul_test) then
         avul_flag = 1
         ind_avul = k
        ! Record time and location of avulsion
         num_avul = num_avul + 1
         avul_out(num_avul,1) = t
         avul_out(num_avul,2) = real(a,kind = prec10)*real(i,kind = prec10)
         avul_out(num_avul,3) = real(a,kind = prec10)*real(j,kind = prec10)
               
        ! Update the stratigraphic record over entire domain
        !  clear matrix delta_strat 
         call strat3D(xmax,ymax,delta_strat,sed_typ,sed_tim,top_lay,strat)

      ! Make the channel downstream of the avulsion inactive
         do jj = ind_avul,(num_act-1)
            i_now = i_vec(jj+1)
            j_now = j_vec(jj+1)           
               
            deltaL_act(jj) =0.d0
            S_act(jj) = 0.d0
            S_index(jj) = 0
            latbond(i_now,j_now) = 0   

            eta_act(jj+1) = 0.d0
            levee_act(jj+1) = 0.d0 
            chan(jj+1) = H(jj+1) 
            H(jj+1) = 0.d0    
            i_vec(jj+1) = 0
            j_vec(jj+1) = 0
         enddo
                              
       ! Create new channel based on path of steepest decent
       !  for the next timestep               
         num_act = ind_avul          
         k = ind_avul
         do 
            i = i_vec(k)
            j = j_vec(k)
          ! Exit if the right boundary has been reached
            if (j == (ymax+1)) EXIT
            call slope5(i,j,latbond,xmax,eta,L,L_act,S)  

          ! Identify next active cell 
            itemp0 = maxloc(S,1) 
            if (S(itemp0) == -999.d0) then
               ! Correction 7 (2026): no free neighbour: continue straight downstream
               itemp0 = 3
            endif
            itemp = coord(itemp0,1)
            jtemp = coord(itemp0,2)
            i = i+itemp
            j = j+jtemp

          ! Update matrices for active channels
            latbond(i,j)=1

          ! Update vectors for active channels
            S_index(k) = itemp0
            deltaL_act(k) = L(itemp0) 
            S_act(k) = S(itemp0)
            
            eta_act(k+1) = eta(i,j)
            levee_act(k+1) = levee(i,j)                
            i_vec(k+1) = i
            j_vec(k+1) = j
   
          ! Update counter of number of active channels
            num_act = num_act + 1
            k = k+1
         enddo !end channel creation  
                     
        ! Incise initial channel to one channel depth               
         do k = (ind_avul+1),num_act
            i = i_vec(k)
            j = j_vec(k)
            if (chan(k) < 10d-8) then
               chan(k) = H_0
            endif
            depth_temp = levee_act(k)-eta_act(k)
            if (depth_temp < chan(k)) then
               eta_before = eta(i,j)
               eta_act(k) = levee_act(k) - chan(k)
               depth_temp = chan(k)
               eta(i,j) = eta_act(k)
               ! Correction 9 (2026): remove the thickness actually cut, not a full channel depth
               delta_incise = eta(i,j) - eta_before
               call incise(i,j,delta_incise,sed_typ,sed_tim,top_lay,strat)
            endif
         enddo

        ! Update slopes; lower downstream cell if negative        
         do  k = ind_avul,(num_act-1)   
            i = i_vec(k)
            j = j_vec(k)
            i_next = i_vec(k+1)
            j_next = j_vec(k+1)

            i_on = S_index(k)
            deltaL_act(k) = L(i_on)
            S_act(k) = (eta_act(k)-eta_act(k+1)) / L(i_on) 

            if (S_act(k)< 10d-8) then
               S_act(k) = S_incise
               ! Correction 2 (2026): length of this step (was L(itemp0), left over from another step)
               new_eta =  eta(i,j) - S_incise * L(i_on)   
               delta_incise = new_eta - eta(i_next,j_next)          

               if (new_eta > 0.d0) then
                  eta(i_next,j_next) = new_eta                
               else
                  delta_incise = -eta(i_next,j_next)
                  eta(i_next,j_next) = 0.d0
               endif
               call incise(i_next,j_next,delta_incise,sed_typ,sed_tim,top_lay,strat)

            endif
            eta_act(k+1) = eta(i_next,j_next)
         enddo

!****************************************************************
        ! Add new layer after avulsion and incision
         do i = 1,xmax
            do j = 1,(ymax+1)
               top_lay(i,j) = top_lay(i,j) + 1
               layer_now = top_lay(i,j)

               if (latbond(i,j) == 1) then
               ! Add a channel layer for the new active channel
                  sed_typ(i,j,layer_now) = 2 
                  sed_tim(i,j,layer_now) = t 
               else
               ! Add a new floodplain layer if the channel is deactivated
                  sed_typ(i,j,layer_now) = 1
                  sed_tim(i,j,layer_now) = t                  
               endif 
            enddo      
         enddo
!****************************************************************

      endif  ! end avulsion if
      k = k+1
   enddo  !end avulsion module

!**********************************************************************
  ! Update slopes (S_act) and distances (deltaL_act)         
   do j=1,num_act-1
      i_on = S_index(j)
      deltaL_act(j) = L(i_on)
      S_act(j) = (eta_act(j)-eta_act(j+1)) / L(i_on)                      
   enddo

!**********************************************************************
  ! PRINT DATA
   if (t-dt>tprint) then
       write(*,20) t
   20  format(1x,'time =',e15.6)

       do zz=1,xmax
          write(neta,30) (eta(zz,zzz),zzz=1,ymax+1)
          write(nlev,30) (levee(zz,zzz),zzz=1,ymax+1)
          write(nQs,30) (qsij(zz,zzz)*B_width,zzz=1,ymax+1)
          write(nH,30) (Hij(zz,zzz),zzz=1,ymax+1)
          write(nbond,40) (latbond(zz,zzz),zzz=1,ymax+1)
       enddo                         
   30  format(51e15.6) 
   40  format(51i15) 
       tprint = tprint+dtprint
   endif
!**********************************************************************   
    if (t>teq) then
   ! Compute the average H in each cell      
       if (t-dt>tprint2) then
          do zz = 1,xmax
             do zzz = 1,(ymax+1)
                if (Hij(zz,zzz) >0.d0) then
                   Havg(zz,zzz) = Havg(zz,zzz) + Hij(zz,zzz) 
                   Navg(zz,zzz) = Navg(zz,zzz) + 1.d0
                endif
             enddo
          enddo
          tprint2 = tprint2+dtprint2
       endif
 
   ! Output qt
       if (t-dt>tprint3) then
          do zz = 1,xmax
             if (latbond(zz,2) ==1) then
                qout2(i_out) = qsij(zz,2)*B_width
             endif
             
             if (latbond(zz,5) ==1) then
                qout5(i_out) = qsij(zz,5)*B_width
             endif

             if (latbond(zz,10) ==1) then                
                qout10(i_out) = qsij(zz,10)*B_width
             endif

             if (latbond(zz,20) ==1) then
                qout20(i_out) = qsij(zz,20)*B_width
             endif

             if (latbond(zz,25) ==1) then
                qout25(i_out) = qsij(zz,25)*B_width
             endif
 
             if (latbond(zz,30) ==1) then
                qout30(i_out) = qsij(zz,30)*B_width
             endif
   
             if (latbond(zz,40) ==1) then
                qout40(i_out) = qsij(zz,40)*B_width
             endif

             if (latbond(zz,50) ==1) then
                qout50(i_out) = qsij(zz,50)*B_width
             endif
          enddo
          tprint3 = tprint3+dtprint3
          i_out = i_out + 1
       endif
    endif
!**********************************************************************
  ! Clear variables and save old eta and strat change
   do zzz=1,(ymax+1)   
      do zz=1,xmax
         Hij(zz,zzz) = 0.d0
         qsij(zz,zzz) = 0.d0
         delta_eta(zz,zzz) = 0.d0
         delta_agg(zz,zzz) = 0.d0
      enddo
   enddo
   old_eta = eta
   old_eta_act = eta_act
   old_levee = levee
   old_levee_act = levee_act  
   
   do i = 1,num_cells
      chan(i) = 0.d0
      H(i) = 0.d0
      qt(i) = 0.d0
   enddo

enddo ! end time loop

!**********************************************************************
! Correction 4 (2026): add the deposition since the last avulsion
call strat3D(xmax,ymax,delta_strat,sed_typ,sed_tim,top_lay,strat)
nlay_max = maxval(top_lay(1:xmax,1:ymax+1))

do zz = 1,xmax
   do zzz = 1,(ymax+1)
      flag_layer = 1

      ! Correction 4 (2026): loop over this column's layers (was 1,num_avul)
      do kk = 1,top_lay(zz,zzz)
         if ((sed_typ(zz,zzz,kk)==1).and. (sed_tim(zz,zzz,kk)>teq)) then
            depth_over(zz,zzz) = depth_over(zz,zzz) + strat(zz,zzz,kk)   
            flag_layer = 1        
         elseif((sed_typ(zz,zzz,kk)==2).and. (sed_tim(zz,zzz,kk)>teq)) then
            depth_chan(zz,zzz) = depth_chan(zz,zzz) + strat(zz,zzz,kk)
            if (flag_layer == 1) then
               num_ch_lay(zz,zzz) = num_ch_lay(zz,zzz) + 1
               flag_layer = 0
            endif
         endif
      enddo

   enddo
enddo 

!**********************************************************************
do zz = 1,xmax
   do zzz = 1,(ymax+1)

      if (Navg(zz,zzz) >0.d0) then
         Havg(zz,zzz) = Havg(zz,zzz)/Navg(zz,zzz)
      else
         Havg(zz,zzz) = 0.d0
      endif

      if (visit_num(zz,zzz) > 0) then
         visit(zz,zzz) = visit(zz,zzz)/visit_num(zz,zzz)  
      else
         visit(zz,zzz) = 0.d0
      endif

      if (visit_num2(zz,zzz) > 0) then
         visit2(zz,zzz) = visit2(zz,zzz)/visit_num2(zz,zzz)         
      else
         visit2(zz,zzz) = 0.d0
      endif


   enddo
enddo  

!**********************************************************************
do zz=1,xmax
   write(ndepthch,70) (depth_chan(zz,zzz),zzz=1,ymax+1)
   write(ndepthov,70) (depth_over(zz,zzz),zzz=1,ymax+1)
   write(nHavg,70) (Havg(zz,zzz),zzz=1,ymax+1)

   write(nvisit,70) (visit(zz,zzz),zzz=1,ymax+1)
   write(nvisitn,75) (visit_num(zz,zzz),zzz=1,ymax+1)
   write(nvisit2,70) (visit2(zz,zzz),zzz=1,ymax+1)
   write(nvisitn2,75) (visit_num2(zz,zzz),zzz=1,ymax+1)   

   write(nchlay,75) (num_ch_lay(zz,zzz),zzz=1,ymax+1)
enddo   
70 format(51e15.6) 
75  format(51i15) 

!**********************************************************************
do k = 1,nvis
   do zz=1,xmax
      write(nTseries,76) (Tv_series(zz,zzz,k),zzz=1,ymax+1)
      write(nTseries2,76) (Tv_series2(zz,zzz,k),zzz=1,ymax+1)      
   enddo
enddo                       
              
76  format(51e15.6) 

!**********************************************************************
write(*,80) t
80  format(1x,'time =',e15.6)

!**********************************************************************
! Correction 4 (2026): write every layer (was 1,num_avul, which left out the last ones)
do k=1,nlay_max
   
   write(nstrat2,82) (strat(i,2,k),i=1,xmax)
   write(nsedtyp2,84) (sed_typ(i,2,k),i=1,xmax)
   write(nsedtim2,82) (sed_tim(i,2,k),i=1,xmax)
   
   write(nstrat5,82) (strat(i,5,k),i=1,xmax)
   write(nsedtyp5,84) (sed_typ(i,5,k),i=1,xmax)
   write(nsedtim5,82) (sed_tim(i,5,k),i=1,xmax)

   write(nstrat10,82) (strat(i,10,k),i=1,xmax)
   write(nsedtyp10,84) (sed_typ(i,10,k),i=1,xmax)
   write(nsedtim10,82) (sed_tim(i,10,k),i=1,xmax)
   
   write(nstrat20,82) (strat(i,20,k),i=1,xmax)
   write(nsedtyp20,84) (sed_typ(i,20,k),i=1,xmax)
   write(nsedtim20,82) (sed_tim(i,20,k),i=1,xmax)

   write(nstrat25,82) (strat(i,25,k),i=1,xmax)
   write(nsedtyp25,84) (sed_typ(i,25,k),i=1,xmax)
   write(nsedtim25,82) (sed_tim(i,25,k),i=1,xmax)
   
   write(nstrat30,82) (strat(i,30,k),i=1,xmax)
   write(nsedtyp30,84) (sed_typ(i,30,k),i=1,xmax)
   write(nsedtim30,82) (sed_tim(i,30,k),i=1,xmax)
   
   write(nstrat40,82) (strat(i,40,k),i=1,xmax)
   write(nsedtyp40,84) (sed_typ(i,40,k),i=1,xmax)
   write(nsedtim40,82) (sed_tim(i,40,k),i=1,xmax)
   
   write(nstrat50,82) (strat(i,50,k),i=1,xmax)
   write(nsedtyp50,84) (sed_typ(i,50,k),i=1,xmax)
   write(nsedtim50,82) (sed_tim(i,50,k),i=1,xmax)

enddo

82 format(71e20.10) 
84 format(71i15)



!**********************************************************************  
do zz=1,num_avul
   write(navul,90) (avul_out(zz,zzz),zzz=1,3)
enddo

90  format(3e20.10)

!**********************************************************************
do zz=1,i_out
   write(nqsout2,95) qout2(zz)
   write(nqsout5,95) qout5(zz)
   write(nqsout10,95) qout10(zz)
   write(nqsout20,95) qout20(zz)
   write(nqsout25,95) qout25(zz)
   write(nqsout30,95) qout30(zz)
   write(nqsout40,95) qout40(zz)
   write(nqsout50,95) qout50(zz)   
enddo

95  format(e20.10)


STOP
END
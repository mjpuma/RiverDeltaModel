subroutine slope5(i,j,latbond,xmax,eta,L,L_act,S)
implicit none
!@sum Subroutine to compute the slope of surrounding 
!@sum deposition of fine sediments
                                                        
!@auth M.J. Puma             
!  
integer*4, parameter :: prec10 = selected_real_kind(10)
integer*4, parameter :: xlim = 71 ! change output format
integer*4, parameter :: ylim = 51 ! if this is changed

real(kind = prec10) :: L_act
integer*4 i,j,xmax

real(kind = prec10),dimension(xlim,ylim) :: eta
integer*4,dimension(xlim,ylim) :: latbond
real(kind = prec10),dimension(5):: S,L

! Calculate slopes taking into account boundaries
if (i /= 1 .and. i/= xmax) then
   S(1) = (eta(i,j)-eta(i-1,j))/L(1) 
   S(2) = (eta(i,j)-eta(i-1,j+1))/L(2)
   S(3) = (eta(i,j)-eta(i,j+1))/L(3)
   S(4) = (eta(i,j)-eta(i+1,j+1))/L(4) 
   S(5) = (eta(i,j)-eta(i+1,j))/L(5)      

   ! Remove slope of active cells
   if(latbond(i-1,j) == 1) then
      S(1) = -999.d0
      L_act = L(1)
   endif  
   
   if(latbond(i-1,j+1) == 1) then
      S(2) = -999.d0
      L_act = L(2)
   endif    

   if(latbond(i,j+1) == 1) then
      S(3) = -999.d0
      L_act = L(3)
   endif    

   if(latbond(i+1,j+1) == 1) then
      S(4) = -999.d0
      L_act = L(4)
   endif    

   if(latbond(i+1,j) == 1)  then
      S(5) = -999.d0
      L_act = L(5)
   endif 
        
elseif (i == 1)then
   S(1) = -999.d0   
   S(2) = -999.d0
   S(3) = (eta(i,j)-eta(i,j+1))/L(3)
   S(4) = (eta(i,j)-eta(i+1,j+1))/L(4) 
   S(5) = (eta(i,j)-eta(i+1,j))/L(5)      

   ! Remove slope of active cells 
   if(latbond(i,j+1) == 1) then
      S(3) = -999.d0
      L_act = L(3)
   endif    

   if(latbond(i+1,j+1) == 1) then
      S(4) = -999.d0
      L_act = L(4)
   endif    

   if(latbond(i+1,j) == 1)  then
      S(5) = -999.d0
      L_act = L(5)
   endif 

elseif (i == xmax )then
   S(1) = (eta(i,j)-eta(i-1,j))/L(1) 
   S(2) = (eta(i,j)-eta(i-1,j+1))/L(2)
   S(3) = (eta(i,j)-eta(i,j+1))/L(3)
   S(4) =  -999.d0
   S(5) =  -999.d0     

   ! Remove slope of active cells    
   if(latbond(i-1,j) == 1) then
      S(1) = -999.d0
      L_act = L(1)
   endif  
      
   if(latbond(i-1,j+1) == 1) then
      S(2) = -999.d0
      L_act = L(2)
   endif    

   if(latbond(i,j+1) == 1) then
      S(3) = -999.d0
      L_act = L(3)
   endif 
     
endif ! end if to check bc


return
END
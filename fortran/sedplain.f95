subroutine sedplain(i,j,latbond,levee_maxj,H_maxj,delta_agg,delta_eta,subside,&
                    fine_dep,dt,I_tavg,eta,levee)
implicit none
!@sum Subroutine to compute the floodplain sedimentation
!@sum of fine sediments
                                                        
!@auth M.J. Puma             
!  
integer*4, parameter :: prec10 = selected_real_kind(10)
integer*4, parameter :: xlim = 71 ! change output format
integer*4, parameter :: ylim = 51 ! if this is changed

integer*4 i,j
integer*4,dimension(xlim,ylim) :: latbond

real(kind = prec10) dt,I_tavg,subside,subside_now,fine_dep
real(kind = prec10),dimension(ylim) :: levee_maxj,H_maxj
real(kind = prec10),dimension(xlim,ylim) :: delta_eta,delta_agg,&
                                            eta,levee

if (latbond(i,j) /= 1) then

   subside_now = subside*dt/I_tavg

   delta_agg(i,j) = fine_dep*dt

!$$$$$$    delta_agg(i,j) = fine_dep*(1+(levee_maxj(j)-eta(i,j))/H_maxj(j))*dt   

   delta_eta(i,j) = delta_agg(i,j) - subside_now
   eta(i,j) = eta(i,j) + delta_eta(i,j) 
   levee(i,j) = levee(i,j) + delta_eta(i,j) 

endif
                        
return
END



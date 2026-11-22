subroutine backwater(num_act,i_vec,j_vec,H_0,S_act,deltaL_act,&
                     ks,qw_zero,alpha_r,g,H,Hij)
implicit none
!@sum Subroutine to flow depths from the backwater formulation                                            
!@auth M.J. Puma 

integer*4, parameter :: prec10 = selected_real_kind(10)
integer*4, parameter :: xlim = 71
integer*4, parameter :: ylim = 51

real(kind = prec10),dimension(xlim,ylim) :: Hij

integer*4,dimension(xlim*ylim) :: i_vec,j_vec
real(kind = prec10),dimension(xlim*ylim) :: S_act,deltaL_act,H

integer*4 num_act,i,j,k,flag_norm
real(kind = prec10) ks,qw_zero,alpha_r,g,H_temp,H_norm,Fr2_p,&
                    Cf,fnp,Hpred,fn,fr2,deltaL_temp,S_temp,H_0

i = i_vec(num_act)
j = j_vec(num_act)
Hij(i,j) = H_0
H(num_act) = H_0
k = num_act
flag_norm = 0
do 
   if (k==1) EXIT
    i = i_vec(k-1)
    j = j_vec(k-1)
            
    H_temp = H(k) 
    S_temp = S_act(k-1)
    deltaL_temp = deltaL_act(k-1)

   ! Compute the normal flow depth
    H_norm = (ks**(1.d0/3.d0) * qw_zero**2.d0 / alpha_r**2.d0 / &
              g / S_temp) ** (3.d0/10.d0)

    if (flag_norm == 0) then
      ! Compute the elevation of channel water using backwater eq
       Fr2_p = qw_zero**2.d0/ g /(H_temp**3.d0)
       Cf = (1.d0/alpha_r**2.d0) * ((H_temp/ks)**(-1.d0/3.d0))
       fnp = (S_temp - Cf*Fr2_p) * deltaL_temp / (1.d0 - Fr2_p)
       Hpred = H_temp - fnp
            
       if(Hpred>0.d0) then
          Fr2 = qw_zero**2.d0/ g / (Hpred**3.d0)
          ! Correction 6 (2026): friction coefficient at the predicted depth
          Cf = (1.d0/alpha_r**2.d0) * ((Hpred/ks)**(-1.d0/3.d0))
          fn = (S_temp - Cf*Fr2) * deltaL_temp / (1.d0-Fr2) 
          H(k-1) = H_temp - 0.5d0*(fnp + fn)
       else
          H(k-1) = H_norm
          flag_norm = 1
       endif

      ! Check if backwater conditions hold       
       if (H(k-1) < H_norm) then
          H(k-1)= H_norm
          flag_norm = 1
       elseif(H(k-1)>H_temp) then
          H(k-1)= H_norm
          flag_norm = 1
       endif
    else
       H(k-1) = H_norm
    endif

   ! Assign flow depth to the matrix
    Hij(i,j) = H(k-1)  
    k=k-1
enddo

return
END
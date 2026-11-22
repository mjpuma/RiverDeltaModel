subroutine incise_H(num_act,H_0,S_act,deltaL_act,&
                     ks,qw_zero,alpha_r,g,chan)
implicit none
!@sum Subroutine to flow depths from the backwater formulation                                            
!@auth M.J. Puma 

integer*4, parameter :: prec10 = selected_real_kind(10)
integer*4, parameter :: xlim = 71
integer*4, parameter :: ylim = 51


real(kind = prec10),dimension(xlim*ylim) :: S_act,deltaL_act,chan

integer*4 num_act,k,flag_norm
real(kind = prec10) ks,qw_zero,alpha_r,g,H_temp,H_norm,Fr2_p,&
                    Cf,fnp,Hpred,fn,fr2,deltaL_temp,S_temp,H_0

chan(num_act) = H_0
k = num_act
flag_norm = 0

do 
   if (k==1) EXIT    
    H_temp = chan(k) 
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
          chan(k-1) = H_temp - 0.5d0*(fnp + fn)
       else
          chan(k-1) = H_norm
          flag_norm = 1
       endif

      ! Check if backwater conditions hold
       if (chan(k-1) < H_norm) then
          chan(k-1)= H_norm
          flag_norm = 1
       elseif(chan(k-1)>H_temp) then
          chan(k-1)= H_norm
          flag_norm = 1
       endif
    else
       chan(k-1) = H_norm
    endif
    k=k-1
enddo


return
END
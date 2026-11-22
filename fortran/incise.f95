subroutine incise(ii,jj,delta_incise,sed_typ,sed_tim,top_lay,strat)
implicit none
!@sum Keeps track of effect of incision on stratigraphy                                  
!@auth M.J. Puma 

integer*4, parameter :: prec10 = selected_real_kind(10)
integer*4, parameter :: xlim = 71
integer*4, parameter :: ylim = 51
integer*4, parameter :: zlim =  20000

integer*4,dimension(xlim,ylim) ::  top_lay

integer*4,dimension(xlim,ylim,zlim) ::  sed_typ
real(kind = prec10),dimension(xlim,ylim,zlim) ::  strat,sed_tim

integer*4 layernow,ii,jj
real(kind = prec10) elev_remain,delta_incise

layernow = top_lay(ii,jj)


! Erosion of sediment occurs
elev_remain = strat(ii,jj,layernow) + delta_incise
         
if (elev_remain > 0.d0)then  
  ! Just reduce elevation of the current layer
   strat(ii,jj,layernow) = strat(ii,jj,layernow)+delta_incise
else  
  ! Elevation loss > thickness of current layer
   elev_remain = delta_incise
   do 
      if (elev_remain > 0.d0) EXIT 
      elev_remain = strat(ii,jj,layernow)+elev_remain

      if(elev_remain > 0.d0)then
         strat(ii,jj,layernow) =  elev_remain
      else ! Remove layer   
         strat(ii,jj,layernow) = 0.d0
         sed_typ(ii,jj,layernow) = -1 
         sed_tim(ii,jj,layernow) = -1.d0 
         top_lay(ii,jj) = top_lay(ii,jj)-1
         layernow = top_lay(ii,jj)
      endif
               
      if (layernow == 1) then
         ! Correction 8 (2026): erosion into the initial deposit lowers it
         if (elev_remain < 0.d0) strat(ii,jj,1) = strat(ii,jj,1) + elev_remain
         EXIT
      endif

   enddo            
endif

      
return 
END




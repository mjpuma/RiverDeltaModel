subroutine strat3D(xmax,ymax,delta_strat,sed_typ,sed_tim,top_lay,strat)
implicit none
!@sum Subroutine to compute the stratigraphy
!@sum  of the modeled domain
                                                        
!@auth M.J. Puma 

integer*4, parameter :: prec10 = selected_real_kind(10)
integer*4, parameter :: xlim = 71
integer*4, parameter :: ylim = 51
integer*4, parameter :: zlim = 20000

integer*4,dimension(xlim,ylim) ::  top_lay
real(kind = prec10),dimension(xlim,ylim) ::  delta_strat

integer*4,dimension(xlim,ylim,zlim) ::  sed_typ
real(kind = prec10),dimension(xlim,ylim,zlim) ::  strat,sed_tim

integer*4 layernow,xmax,ymax,ii,jj
real(kind = prec10) elev_remain



do ii = 1,xmax
  do jj = 1,(ymax+1)
     layernow = top_lay(ii,jj)
     
     if(delta_strat(ii,jj)>0.d0)then 
        ! Deposition of sediment occurs   
         strat(ii,jj,layernow) = delta_strat(ii,jj)
     else 
        ! Erosion of sediment occurs

         elev_remain = delta_strat(ii,jj)
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

      delta_strat(ii,jj) = 0.d0

   enddo
enddo   


return 
END



subroutine bed_elev(xmax,ymax,eta)
implicit none
!@sum Subroutine to read the initial bed elevation
                                                      
!@auth M.J. Puma 

integer*4, parameter :: prec10 = selected_real_kind(10)
integer*4, parameter :: xlim = 71
integer*4, parameter :: ylim = 51


! Correction (2026): unit 12 is levout.txt in the main program; use another unit
integer*4, parameter :: neta_in = 90

real(kind = prec10),dimension(xlim,ylim) ::  eta
integer*4 xmax,ymax,i,j


!   Open file
open(unit=neta_in,file='eta_init.txt')

do i=1,xmax
   read(neta_in,*) (eta(i,j), j=1,(ymax+1))
enddo

close(unit=neta_in)
        
return
end
subroutine  tim_visit(t,xmax,ymax,latbond,flag_visit,last_t,visit,visit_num,Tv_series,&
                        flag_visit2,last_t2,visit2,visit_num2,Tv_series2)
implicit none
!@sum Keeps track time btw chnnel visits and number of visits                                 
!@auth M.J. Puma 

integer*4, parameter :: prec10 = selected_real_kind(10)
integer*4, parameter :: xlim = 71
integer*4, parameter :: ylim = 51
integer*4, parameter :: nvis = 100

integer*4,dimension(xlim,ylim) ::  latbond,flag_visit,visit_num,&
                                    flag_visit2,visit_num2
real(kind = prec10),dimension(xlim,ylim) :: last_t,visit,last_t2,visit2
real(kind = prec10),dimension(xlim,ylim,nvis) :: Tv_series,Tv_series2

integer*4 xmax,ymax,i,j,temp,temp2
real(kind = prec10) t


do i = 1,xmax
  do j = 1, (ymax+1)
    
   ! Calculates duration of channel visit to a cell
    if((latbond(i,j)==1).and.(flag_visit(i,j)==0))then
       last_t(i,j) = t
       flag_visit(i,j) = 1
    elseif((latbond(i,j) == 0).and.(flag_visit(i,j)==1))then
       visit_num(i,j) = visit_num(i,j) + 1
       temp = visit_num(i,j)       
       ! Correction (2026): the series holds nvis visits; it used to write past its end
       if (temp <= nvis) Tv_series(i,j,temp) = t - last_t(i,j)
       visit(i,j) = visit(i,j) + (t - last_t(i,j))

       flag_visit(i,j) = 0
    endif
    
   ! Calculates time btw channel visits to a cell
    if((latbond(i,j)==1).and.(flag_visit2(i,j)==0))then
       last_t2(i,j) = t
       flag_visit2(i,j) = 1
    elseif((latbond(i,j) == 0).and.(flag_visit2(i,j)==1))then
       flag_visit2(i,j) = 2
    elseif((latbond(i,j) == 1).and.(flag_visit2(i,j)==2))then          
       visit_num2(i,j) = visit_num2(i,j) + 1
       temp2 = visit_num2(i,j)       
       if (temp2 <= nvis) Tv_series2(i,j,temp2) = t - last_t2(i,j)
       visit2(i,j) = visit2(i,j) + (t - last_t2(i,j))

       flag_visit2(i,j) = 1
       last_t2(i,j) = t
    endif


  enddo
enddo


return
END
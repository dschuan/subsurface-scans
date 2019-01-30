import serial


ptsx=1
ptsy=1
intx=1
inty=1
xboundary = (ptsx - 1) * intx
yboundary = (ptsy - 1) * inty

ptsx2=2
ptsy2=2
intx2=1
inty2=1
xboundary2 = (ptsx2 - 1) * intx2
yboundary2 = (ptsy2 - 1) * inty2

spdx=10
spdy=10

spdy=(spdy*10/10)*2*60 #calculate the speed for y-axis to send to faulhaber motor
inty=(inty)*6000 #calculate the distance between points along y-axis
inty2=(inty2)*6000 #calculate the distance between points along y-axis

spdx=(spdx*10/10)*60 #calculate the speed for x-axis to send to faulhaber motor
intx=(intx)*3000 #calculate the distance between points along x-axis
intx2=(intx2)*3000 #calculate the distance between points along x-axis

spd1 = '1SP' #y-axis
spd2 = '2SP' #x-axis

a = '1la-' #Position header for Y-axis
c = '2la' #Position header for X-axis

Yspd = spd1+str(spdy)
Xspd = spd2+str(spdx)

############################################################
###########        Create Serial Object         ############
############################################################

# CREATE SERIAL OBJECT
s = serial('COM8') #create serial object
s.InputBufferSize = 8388608
fopen(s) #connect to serial object


############################################################
#############         Initialisation          ##############
############################################################

fprintf(s,'JMP2') #Homing Sequence/Initialise
pause(5)
fprintf(s,Xspd) #set X-axis speed to user's choice
fprintf(s,Yspd) #set Y-axis speed to user's choice

############################################################
#############            Counters             ##############
############################################################

b = 0                  # Initial Position for Y-Axis (Setup A)
d = 0                  # Initial Position for X-Axis (Setup A)
e = 0                  # Initial Position for Y-Axis (Setup B)
f = 0                  # Initial Position for X-Axis (Setup B)
msg = 0                # Counter for points reached, should be equal to X*Y points
zz = 0
yy = 1                 # Form name of file (Y-component)
xx = 1                 # Form name of file (X-component)
xx2 = 1                # Form name of file (X-component)
yy2 = 1                # Form name of file (Y-component)
ax = (ptsx) + 1          # Limit for x-axis of plot
ay = (ptsy) + 1          # Limit for y-axis of plot
ax2 = (ptsx2) + 1          # Limit for x-axis of plot
ay2 = (ptsy2) + 1          # Limit for y-axis of plot

############################################################
#############          Plot Graph             ##############
############################################################

clf
h = plot3(xx,yy,zz,'r*','EraseMode','none','MarkerSize',5)
title('Setup A')
xlabel('X-Axis')
ylabel('Y-Axis')
axis([0 ax 0 ay])
hold on
grid

figure
h2 = plot3(xx2,yy2,zz,'r*','EraseMode','none','MarkerSize',5)
title('Setup B')
xlabel('X-Axis')
ylabel('Y-Axis')
axis([0 ax2 0 ay2])
hold on
grid

xx = xx - 1
xx2 = xx2 - 1

############################################################
###########        Running of Positioner        ############
############################################################

############################################################
##################        Setup B        ###################
############################################################

for xa2 in range(1+ptsx2)
    xx2 = xx2 + 1
    # set(h2,'XData',xx2,'YData',yy2)
    # refreshdata(h2)
    # drawnow
    #pause(2)
    for ya2 in range(1+ptsy2)
        refresh

            ############################################################
            ################          Setup A          #################
            ############################################################

            for xa in range(ptsx+1)
                xx = xx + 1
                # set(h,'XData',xx,'YData',yy)
                # refreshdata(h)
                # drawnow
                #pause(2)
                for ya in range(1+ptsy)

                    ############################################################
                    #############         Saving of Data          ##############
                    ############################################################
                    #tic
                    xy = strcat(num2str(xx), num2str(yy))
                    xy2 = strcat(num2str(xx2),num2str(yy2))
                    msg = msg + 1#code for doing averaging
                    pause(3) # delay to allow averaging to complete
                    pause(2) # delay to allow saving of file
                    #toc
                    ############################################################
                    ##########      Running of Setup A (Y-Axis)      ###########
                    ############################################################

                    if mod(xa,2) ~= 0  #### Moving Positively ####
                        if ya ~= (ptsy)
                            yy = yy + 1
                            ya = ya + 1
                            b = b + inty
                            move1 = strcat(a,num2str(b))
#                             fprintf(s,move1) #move antenna by stated distance
#                             fprintf(s,'1M') #initiate move
                            pause(1)
                            set(h,'XData',xx,'YData',yy)
                            refreshdata(h)
                            drawnow
                            pause(4)
                        end
                    else              #### Moving Negatively ####
                        if ya ~= (ptsy)
                            yy = yy - 1
                            ya = ya + 1
                            b = b - inty
                            move1 = strcat(a,num2str(b))
#                             fprintf(s,move1) #move antenna by stated distance
#                             fprintf(s,'1M') #initiate move
                            pause(1)
                            set(h,'XData',xx,'YData',yy)
                            refreshdata(h)
                            drawnow
                            pause(4)
                        end
                    end
                end

                ############################################################
                ##########      Running of Setup A (X-Axis)      ###########
                ############################################################

                if xa ~= (ptsx)
                    xa = xa + 1
                    d = d + intx
                    move2 = strcat(c,num2str(d))
#                     fprintf(s,move2)
#                     fprintf(s,'2M')
                    pause(5)
                end
            end

            ############################################################
            ##########         Returning to 1st point        ###########
            ############################################################

            yy = 1
            xx = 0
            b = 0
            d = 0
#             fprintf(s,'1la0') #move antenna by stated distance
#             fprintf(s,'1M') #initiate move
#             fprintf(s,'2la0') #move antenna by stated distance
#             fprintf(s,'2M') #initiate move
            pause(8)

        ############################################################
        ##########      Running of Setup B (Y-Axis)      ###########
        ############################################################

        if mod(xa2,2) ~= 0  #### Moving Positively ####
            if ya2 ~= (ptsy2)
                yy2 = yy2 + 1
                ya2 = ya2 + 1
                # code for moving in y
                e = e + inty2
                move1 = strcat(a,num2str(e))
                fprintf(s,move1) #move antenna by stated distance
                fprintf(s,'1M') #initiate move
                pause(3)
                set(h2,'XData',xx2,'YData',yy2)
                refreshdata(h2)
                drawnow
                #pause(2)
            end
        else              #### Moving Negatively ####
            if ya2 ~= (ptsy2)
                yy2 = yy2 - 1
                ya2 = ya2 + 1
                # code for moving in y
                e = e - inty2
                move1 = strcat(a,num2str(e))
                fprintf(s,move1) #move antenna by stated distance
                fprintf(s,'1M') #initiate move
                pause(3)
                set(h2,'XData',xx2,'YData',yy2)
                refreshdata(h2)
                drawnow
                #pause(2)
            end
        end
    end

    ############################################################
    ##########      Running of Setup B (Y-Axis)      ###########
    ############################################################

    if xa2 ~= (ptsx2)
        xa2 = xa2 + 1
        # code for moving in x
        f = f + intx2
        move2 = strcat(c,num2str(f))
        fprintf(s,move2)
        fprintf(s,'2M')
        pause(2)
    end
end

############################################################
##########         Reinitialise and THE END      ###########
############################################################

fprintf(s,'JMP2')
disp(['Number of Points Measured = ' num2str(msg)])
fclose(s)
delete(s)
end
